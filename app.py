# 🧱 Chunk Daly — Layout-Aware Keyword→Chunk Matcher (Chonkie + Sentence-Transformers)
# - Upload CSV of keyword,url
# - Async crawl (≤10), cache HTML in SQLite, chunk with Chonkie → semantic fallback → full-page packing (no truncation)
# - Query/Document-aware embeddings (MXBAI, E5, BGE, GTE, Gemma, etc.)
# - Best-chunk cosine + whole-page cosine; Top-K Audit; cached embedder
# - Model-dimension safety (auto-purge stale embeddings on model switch)
# - Excel-friendly CSV export (UTF-8 with BOM) + .xlsx export option

import os
import io
import time
import sqlite3
import asyncio
from contextlib import closing
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

# --- Crawling ---
import httpx
import trafilatura

# --- Chunking (Chonkie) ---
try:
    from chonkie import LayoutAwareChunker, SemanticChunker
except Exception:
    LayoutAwareChunker = None
    SemanticChunker = None

# --- Embeddings via Sentence-Transformers ---
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfFolder

# --- Clean/pack utilities ---
import re, html

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Chunk Daly", layout="wide")
st.title("🧱 Chunk Daly — Layout-Aware Keyword→Chunk Matcher")

with st.sidebar:
    st.header("Configuration")
    db_path = st.text_input("SQLite DB path", value="chunks.db")
    crawl_concurrency = st.slider("Max concurrent crawls", 1, 10, 10)
    model_name = st.text_input(
        "Embedding model",
        value="mixedbread-ai/mxbai-embed-large-v1",  # you can switch to google/embeddinggemma-300m etc.
        help="Any Sentence-Transformers model (Hugging Face). Uses query/doc-specific encoding when available."
    )
    chunk_max_chars = st.number_input("Max characters per chunk", 500, 4000, 1400, 100)
    force_recrawl = st.toggle("Force recrawl URLs (refresh HTML cache)", value=False)
    rechunk_existing = st.toggle("Re-chunk existing pages (no recrawl)", value=False,
                                 help="Rebuild chunks from stored HTML even if chunks already exist.")
    force_rescore = st.toggle("Force rescore results (ignore cached results)", value=True)
    topk_audit = st.number_input("Audit: show Top-K candidates", min_value=1, max_value=10, value=5)

uploaded = st.file_uploader("Upload a CSV with columns: keyword,url", type=["csv"])
run_btn = st.button("Run matching")

# ======================
# Text cleanup + packing
# ======================
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    return s.strip()

def pack_chunks(chunks: List[str], min_chars=400, max_chars=1400) -> List[str]:
    packed, cur, cur_len = [], [], 0
    for ch in chunks:
        t = normalize_text(ch)
        if not t:
            continue
        if cur_len + len(t) + 1 <= max_chars:
            cur.append(t)
            cur_len += len(t) + 1
        else:
            if cur:
                blob = "\n\n".join(cur).strip()
                if blob:
                    packed.append(blob)
            cur, cur_len = [t], len(t)
        if cur_len >= min_chars:
            blob = "\n\n".join(cur).strip()
            if blob:
                packed.append(blob)
            cur, cur_len = [], 0
    if cur:
        blob = "\n\n".join(cur).strip()
        if blob:
            packed.append(blob)

    # Merge ultra-short leftovers into previous if possible
    result, buf = [], ""
    for p in packed:
        if len(p) < min_chars and buf and len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}"
        else:
            if buf:
                result.append(buf)
            buf = p
    if buf:
        result.append(buf)
    return result

# ======================
# SQLite helpers (with migration)
# ======================
DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS pages (
  url TEXT PRIMARY KEY,
  fetched_at INTEGER,
  status_code INTEGER,
  html BLOB
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT,
  chunk_index INTEGER,
  text TEXT,
  UNIQUE(url, chunk_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scope TEXT,      -- 'keyword' or 'chunk' or 'page'
  key TEXT,        -- keyword text OR url|chunk_index OR url (for page)
  dim INTEGER,
  vec BLOB,
  UNIQUE(scope, key)
);

CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  keyword TEXT,
  url TEXT,
  best_chunk_index INTEGER,
  best_score REAL,
  best_chunk_text TEXT
);
"""

def db_connect(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def db_init(conn: sqlite3.Connection):
    conn.executescript(DDL)
    # Add page_score if missing
    cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    if "page_score" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN page_score REAL")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_results_kw_url ON results(keyword, url)")

def db_write_page(conn: sqlite3.Connection, url: str, status_code: int, html_bytes: bytes):
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO pages (url, fetched_at, status_code, html) VALUES (?, ?, ?, ?)",
            (url, int(time.time()), status_code, html_bytes),
        )

def db_read_page_html(conn: sqlite3.Connection, url: str) -> Optional[str]:
    row = conn.execute("SELECT html FROM pages WHERE url=?", (url,)).fetchone()
    if not row:
        return None
    try:
        return row[0].decode("utf-8", errors="ignore")
    except Exception:
        return row[0]

def db_write_chunks(conn: sqlite3.Connection, url: str, chunks: List[str]):
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks (url, chunk_index, text) VALUES (?, ?, ?)",
            [(url, i, ch) for i, ch in enumerate(chunks)]
        )

def db_get_chunks(conn: sqlite3.Connection, url: str) -> List[Tuple[int, str]]:
    return conn.execute(
        "SELECT chunk_index, text FROM chunks WHERE url=? ORDER BY chunk_index ASC", (url,)
    ).fetchall()

def db_has_embedding(conn: sqlite3.Connection, scope: str, key: str) -> bool:
    return conn.execute("SELECT 1 FROM embeddings WHERE scope=? AND key=?", (scope, key)).fetchone() is not None

def db_write_embedding(conn: sqlite3.Connection, scope: str, key: str, vec: np.ndarray):
    vec = vec.astype(np.float32)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (scope, key, dim, vec) VALUES (?, ?, ?, ?)",
            (scope, key, vec.shape[0], vec.tobytes()),
        )

def db_read_embedding(conn: sqlite3.Connection, scope: str, key: str) -> Optional[np.ndarray]:
    row = conn.execute("SELECT dim, vec FROM embeddings WHERE scope=? AND key=?", (scope, key)).fetchone()
    if not row:
        return None
    dim, blob = row
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != dim:
        return None
    return arr

def db_write_result(conn: sqlite3.Connection, keyword: str, url: str, idx: int, score: float, text: str, page_score: Optional[float]):
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO results (keyword, url, best_chunk_index, best_score, best_chunk_text, page_score) VALUES (?, ?, ?, ?, ?, ?)",
            (keyword, url, idx, float(score), text, None if page_score is None else float(page_score)),
        )

def db_read_results(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT keyword, url AS landing_page, best_score AS cosine_similarity, "
        "page_score AS page_cosine_similarity, best_chunk_text AS chunk_text "
        "FROM results ORDER BY cosine_similarity DESC",
        conn
    )

# ======================
# Crawl (async, up to N)
# ======================
async def fetch_one(url: str) -> Tuple[str, int, str]:
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (Chunk-Daly/1.0)"}
        ) as client:
            r = await client.get(url)
            return url, r.status_code, r.text
    except Exception:
        return url, 0, ""

async def crawl_urls(urls: List[str], max_conc: int) -> Dict[str, Tuple[int, str]]:
    sem = asyncio.Semaphore(max_conc)
    out: Dict[str, Tuple[int, str]] = {}

    async def job(u: str):
        async with sem:
            url, code, text = await fetch_one(u)
            out[url] = (code, text)

    await asyncio.gather(*(job(u) for u in urls))
    return out

# ======================
# Extraction + Chunking
# ======================
def extract_main_text(html_src: str) -> str:
    try:
        txt = trafilatura.extract(
            html_src,
            include_comments=False,
            include_tables=False,
            favor_recall=True
        ) or ""
        return normalize_text(txt)
    except Exception:
        return ""

def layout_aware_chunk(html_src: str, max_chars_hint: int) -> List[str]:
    base_chunks: List[str] = []

    if LayoutAwareChunker is not None:
        try:
            la = LayoutAwareChunker(max_characters=max_chars_hint)
            pieces = list(la.chunk_html(html_src))
            if pieces:
                base_chunks = [getattr(p, "text", str(p)) for p in pieces if str(p).strip()]
        except Exception:
            base_chunks = []

    if not base_chunks:
        txt = extract_main_text(html_src)
        if SemanticChunker is not None and txt:
            try:
                sc = SemanticChunker(target_chunk_char_count=max_chars_hint)
                pieces = list(sc.chunk(txt))
                if pieces:
                    base_chunks = [getattr(p, "text", str(p)) for p in pieces if str(p).strip()]
            except Exception:
                base_chunks = [txt] if txt else []
        elif txt:
            paras = [p.strip() for p in txt.split("\n") if p.strip()]
            base_chunks = paras

    base_chunks = [normalize_text(c) for c in base_chunks if c and c.strip()]
    if not base_chunks:
        return []

    min_chars = max(350, int(max_chars_hint * 0.35))
    max_chars = max_chars_hint
    packed = pack_chunks(base_chunks, min_chars=min_chars, max_chars=max_chars)
    packed = [p for p in packed if len(p) >= int(min_chars * 0.6)]
    return packed

# ======================
# Embeddings (Sentence-Transformers, query/doc aware)
# ======================
class STEEmbedder:
    def __init__(self, model_name: str):
        token = os.getenv("HF_TOKEN") or HfFolder.get_token()
        try:
            try:
                self.model = SentenceTransformer(model_name, token=token, trust_remote_code=True)
            except TypeError:
                self.model = SentenceTransformer(model_name, use_auth_token=token, trust_remote_code=True)
        except Exception as e:
            msg = (
                "Could not initialize SentenceTransformer for this model.\n\n"
                "Tips:\n"
                "  • Ensure you've accepted the model license on Hugging Face (if gated).\n"
                "  • Run `huggingface-cli login` OR set HF_TOKEN environment variable.\n"
                f"Underlying error: {e}"
            )
            raise RuntimeError(msg)

    # unified encoding that tries task/prompt_name first, then robust prefixes
    def _encode_role(self, texts: List[str], role: str) -> np.ndarray:
        texts = [normalize_text(t) for t in texts]

        # Try model-native kwargs
        for kwargs in (
            {"task": "search_query"} if role == "query" else {"task": "search_document"},
            {"prompt_name": "query"} if role == "query" else {"prompt_name": "passage"},
            {"prompt_name": "document"} if role == "doc" else None,
        ):
            if not kwargs:
                continue
            try:
                vecs = self.model.encode(
                    texts,
                    batch_size=128,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                    **kwargs
                )
                return vecs.astype(np.float32)
            except TypeError:
                continue  # model doesn't support this kwarg

        # Fallback to prefixes (works for E5/BGE/GTE/MXBAI style models)
        prefixed = [f"query: {t}" if role == "query" else f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False
        )
        return vecs.astype(np.float32)

    def embed_query(self, texts: List[str]) -> np.ndarray:
        return self._encode_role(texts, role="query")

    def embed_doc(self, texts: List[str]) -> np.ndarray:
        return self._encode_role(texts, role="doc")

    # Back-compat if something calls .embed()
    def embed(self, texts: List[str]) -> np.ndarray:
        return self.embed_doc(texts)

    def dim(self) -> int:
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            return int(self.embed_doc(["probe"]).shape[1])

@st.cache_resource(show_spinner=False)
def get_embedder_cached(model_name: str) -> STEEmbedder:
    return STEEmbedder(model_name)

# ======================
# Dimension safety helpers
# ======================
def get_expected_dim(embedder: STEEmbedder) -> int:
    try:
        return int(embedder.model.get_sentence_embedding_dimension())
    except Exception:
        v = embedder.embed_doc(["__probe_dim__"])[0]
        return int(v.shape[0])

def purge_mismatched_embeddings(conn: sqlite3.Connection, expected_dim: int) -> int:
    """Delete embeddings not matching current model dim; clear results depending on them."""
    row = conn.execute("SELECT COUNT(*) FROM embeddings WHERE dim != ?", (expected_dim,)).fetchone()
    n = int(row[0]) if row else 0
    if n:
        with conn:
            conn.execute("DELETE FROM embeddings WHERE dim != ?", (expected_dim,))
            conn.execute("DELETE FROM results")
    return n

# ======================
# Cosine + whole-page embedding
# ======================
def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

def _mean_pool_unit_rows(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows
    avg = rows.mean(axis=0)
    n = np.linalg.norm(avg) + 1e-8
    return (avg / n).astype(np.float32)

def get_or_build_page_embedding(conn: sqlite3.Connection, embedder: STEEmbedder, url: str) -> Optional[np.ndarray]:
    v = db_read_embedding(conn, "page", url)
    if v is not None:
        exp = get_expected_dim(embedder)
        if int(v.shape[0]) != int(exp):
            v = None  # force rebuild
    if v is not None:
        return v

    url_chunks = db_get_chunks(conn, url)
    if url_chunks:
        chunk_keys = [f"{url}|{idx}" for idx, _ in url_chunks]
        vecs = []
        exp = get_expected_dim(embedder)
        for ck in chunk_keys:
            vv = db_read_embedding(conn, "chunk", ck)
            if vv is None or int(vv.shape[0]) != exp:
                idx = int(ck.split("|")[-1])
                text = normalize_text(dict(url_chunks)[idx])
                vv = embedder.embed_doc([text])[0]
                db_write_embedding(conn, "chunk", ck, vv)
            vecs.append(vv)
        mat = np.vstack(vecs).astype(np.float32)
        mat_n = l2_normalize(mat)
        page_vec = _mean_pool_unit_rows(mat_n)
        db_write_embedding(conn, "page", url, page_vec)
        return page_vec

    # Fallback: no chunks — embed extracted full text as a doc
    html_src = db_read_page_html(conn, url) or ""
    if not html_src:
        return None
    full_text = extract_main_text(html_src)
    if not full_text:
        return None
    v = embedder.embed_doc([full_text])[0]
    v = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    db_write_embedding(conn, "page", url, v)
    return v

def score_keyword_url(conn: sqlite3.Connection, embedder: STEEmbedder, keyword: str, url: str):
    """
    Returns (best_idx, sims, chunk_indices, chunk_texts, page_score)
      sims: float32 np.array cosine similarities per chunk (L2-normalized dot)
      page_score: cosine similarity between keyword and whole-page embedding
    """
    kw_vec = db_read_embedding(conn, "keyword", keyword)
    exp = get_expected_dim(embedder)
    if kw_vec is None or int(kw_vec.shape[0]) != exp:
        kw_vec = embedder.embed_query([keyword])[0]
        db_write_embedding(conn, "keyword", keyword, kw_vec)

    url_chunks = db_get_chunks(conn, url)
    if not url_chunks:
        page_vec = get_or_build_page_embedding(conn, embedder, url)
        if page_vec is None:
            return -1, np.zeros(0, dtype=np.float32), [], [], 0.0
        kw_vec_n = l2_normalize(kw_vec.reshape(1, -1))[0]
        page_score = float(np.dot(kw_vec_n, page_vec))
        return -1, np.zeros(0, dtype=np.float32), [], [], page_score

    chunk_map = dict(url_chunks)
    chunk_keys = [f"{url}|{idx}" for idx, _ in url_chunks]

    # Build chunk embeddings in order; re-embed if dim mismatch (as DOC)
    chunk_vecs = []
    for ck in chunk_keys:
        idx = int(ck.split("|")[-1])
        v = db_read_embedding(conn, "chunk", ck)
        if v is None or int(v.shape[0]) != exp:
            text = normalize_text(chunk_map[idx])
            v = embedder.embed_doc([text])[0]
            db_write_embedding(conn, "chunk", ck, v)
        chunk_vecs.append(v)
    chunk_mat = np.vstack(chunk_vecs).astype(np.float32)

    kw_vec_n = l2_normalize(kw_vec.reshape(1, -1))[0]
    chunk_mat_n = l2_normalize(chunk_mat)
    sims = (chunk_mat_n @ kw_vec_n.reshape(-1,)).astype(np.float32)

    # whole-page score
    page_vec = get_or_build_page_embedding(conn, embedder, url)
    page_score = 0.0
    if page_vec is not None:
        page_score = float(np.dot(kw_vec_n, page_vec))

    best_idx = int(np.argmax(sims)) if sims.size else -1
    chunk_indices = [int(ck.split("|")[-1]) for ck in chunk_keys]
    chunk_texts = [normalize_text(chunk_map[i]) for i in chunk_indices]
    return best_idx, sims, chunk_indices, chunk_texts, page_score

# ======================
# Pipeline
# ======================
async def crawl_and_cache(conn: sqlite3.Connection, urls: List[str], max_conc: int):
    if not urls:
        return
    results = await crawl_urls(urls, max_conc)
    for u, (code, html_text) in results.items():
        db_write_page(conn, u, code, html_text.encode("utf-8", errors="ignore") if html_text else b"")

def run_pipeline(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    max_conc: int,
    embedder: STEEmbedder,
    max_chars_hint: int,
    force_recrawl: bool,
    rechunk_existing: bool,
    force_rescore: bool
):
    # Clean input
    df = df.dropna(subset=["keyword", "url"]).copy()
    df["keyword"] = df["keyword"].str.strip()
    df["url"] = df["url"].str.strip()
    df = df[df["url"].str.startswith(("http://", "https://"))].drop_duplicates(subset=["keyword", "url"])
    needed_urls = sorted(set(df["url"].tolist()))

    # Ensure embedding dims in DB match the current model
    expected_dim = get_expected_dim(embedder)
    purged = purge_mismatched_embeddings(conn, expected_dim)
    if purged:
        st.warning(f"Chunk Daly: removed {purged} stale embeddings (dim mismatch). Recomputing with current model…")

    # Crawl (skip cached unless force)
    if not force_recrawl:
        cached = {u for (u,) in conn.execute("SELECT url FROM pages")}
        to_fetch = [u for u in needed_urls if u not in cached]
    else:
        to_fetch = needed_urls

    if to_fetch:
        st.info(f"Crawling {len(to_fetch)} URL(s)…")
        asyncio.run(crawl_and_cache(conn, to_fetch, max_conc))

    # Chunk (with rebuild logic)
    st.info("Chunking pages (layout-aware; full-page pack fallback)…")
    for u in needed_urls:
        # Inspect existing chunks
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(text)),0) FROM chunks WHERE url=?", (u,)
        ).fetchone()
        chunk_count = row[0] or 0
        total_len = row[1] or 0
        healthy = (chunk_count >= 2) or (total_len >= max_chars_hint * 1.5)

        if healthy and not force_recrawl and not rechunk_existing:
            pass
        else:
            html_src = db_read_page_html(conn, u) or ""
            chunks = layout_aware_chunk(html_src, max_chars_hint) if html_src else []
            if not chunks and html_src:
                text = extract_main_text(html_src)
                if text:
                    paras = [p.strip() for p in text.split("\n") if p.strip()]
                    min_chars = max(350, int(max_chars_hint * 0.35))
                    chunks = pack_chunks(paras, min_chars=min_chars, max_chars=max_chars_hint)

            with conn:
                conn.execute("DELETE FROM chunks WHERE url=?", (u,))
                conn.execute("DELETE FROM embeddings WHERE scope='chunk' AND key LIKE ?", (f"{u}|%",))
                conn.execute("DELETE FROM embeddings WHERE scope='page' AND key=?", (u,))  # invalidate page vector
            if chunks:
                db_write_chunks(conn, u, chunks)

    # Embed chunks (only missing) + ensure page embedding
    st.info("Embedding chunks…")
    for u in needed_urls:
        url_chunks = db_get_chunks(conn, u)
        if not url_chunks:
            continue
        missing_texts, missing_keys = [], []
        for idx, txt in url_chunks:
            key = f"{u}|{idx}"
            if not db_has_embedding(conn, "chunk", key):
                missing_texts.append(normalize_text(txt))
                missing_keys.append(key)
        if missing_texts:
            vecs = embedder.embed_doc(missing_texts)
            for k, v in zip(missing_keys, vecs):
                db_write_embedding(conn, "chunk", k, v)
        # build page vector from chunk vectors
        _ = get_or_build_page_embedding(conn, embedder, u)

    # Embed keywords (only missing) — as QUERIES
    st.info("Embedding keywords…")
    uniq_kws = sorted(set(df["keyword"].tolist()))
    missing_kws = [kw for kw in uniq_kws if not db_has_embedding(conn, "keyword", kw)]
    if missing_kws:
        kw_vecs = embedder.embed_query(missing_kws)
        for kw, v in zip(missing_kws, kw_vecs):
            db_write_embedding(conn, "keyword", kw, v)

    # Score best chunk + whole-page cosine
    st.info("Scoring best chunks…")
    for _, r in df.iterrows():
        kw, u = r["keyword"], r["url"]

        if not force_rescore and conn.execute(
            "SELECT 1 FROM results WHERE keyword=? AND url=?", (kw, u)
        ).fetchone():
            continue
        else:
            with conn:
                conn.execute("DELETE FROM results WHERE keyword=? AND url=?", (kw, u))

        best_idx, sims, chunk_indices, chunk_texts, page_score = score_keyword_url(conn, embedder, kw, u)

        if sims.size == 0 or best_idx < 0:
            db_write_result(conn, kw, u, -1, 0.0, "", page_score)
            continue

        argmax_idx = int(np.argmax(sims))
        if argmax_idx != best_idx:
            best_idx = argmax_idx

        chosen_chunk_idx = chunk_indices[best_idx]
        chosen_text = chunk_texts[best_idx]
        db_write_result(conn, kw, u, chosen_chunk_idx, float(sims[best_idx]), chosen_text, page_score)

# ======================
# Run matching (on click)
# ======================
if run_btn:
    if uploaded is None:
        st.error("Please upload a CSV first.")
    else:
        try:
            df_in = pd.read_csv(uploaded)
            cols = {c.lower(): c for c in df_in.columns}
            if "keyword" not in cols or "url" not in cols:
                st.error("CSV must include columns: keyword,url")
                df_in = None
            else:
                df_in = df_in.rename(columns={cols["keyword"]: "keyword", cols["url"]: "url"})
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_in = None

        if df_in is not None and not df_in.empty:
            with closing(db_connect(db_path)) as conn:
                db_init(conn)
                try:
                    embedder = get_embedder_cached(model_name)
                except Exception as e:
                    st.error(str(e)); st.stop()

                with st.spinner("Processing…"):
                    run_pipeline(
                        df=df_in,
                        conn=conn,
                        max_conc=crawl_concurrency,
                        embedder=embedder,
                        max_chars_hint=chunk_max_chars,
                        force_recrawl=force_recrawl,
                        rechunk_existing=rechunk_existing,
                        force_rescore=force_rescore
                    )
                st.success("Matching complete.")

# ======================
# Always show results + Audit
# ======================
with closing(db_connect(db_path)) as conn:
    db_init(conn)
    out_df = db_read_results(conn)

st.subheader("Results")
if out_df.empty:
    st.info("No results yet. Upload a CSV and click ‘Run matching’, then return here to audit.")
else:
    def preview(s: str, n=220):
        s = normalize_text(s).replace("\n", " ")
        return (s[:n] + "…") if len(s) > n else s

    out_df["chunk_preview"] = out_df["chunk_text"].apply(preview)
    st.dataframe(
        out_df[["keyword", "landing_page", "cosine_similarity", "page_cosine_similarity", "chunk_preview"]],
        use_container_width=True
    )

    # --- Excel-friendly CSV (UTF-8 with BOM) ---
    csv_bom_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download results as CSV (UTF-8)",
        data=csv_bom_bytes,
        file_name="chunk_daly_results.csv",
        mime="text/csv"
    )

    # --- Optional: Excel .xlsx export ---
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Results")
    st.download_button(
        "Download results as Excel (.xlsx)",
        data=xbuf.getvalue(),
        file_name="chunk_daly_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")
st.subheader("🔎 Audit a keyword → URL match (Top-K)")
if out_df.empty:
    st.caption("Run matching first to populate results.")
else:
    kw_choices = sorted(out_df["keyword"].unique().tolist())
    kw_pick = st.selectbox("Keyword", kw_choices, key="audit_kw")
    urls_for_kw = out_df.loc[out_df["keyword"] == kw_pick, "landing_page"].unique().tolist()
    u_pick = st.selectbox("URL", urls_for_kw, key="audit_url")

    if st.button("Run audit", key="run_audit"):
        try:
            embedder = get_embedder_cached(model_name)
        except Exception as e:
            st.error(str(e)); st.stop()

        with closing(db_connect(db_path)) as audit_conn:
            # ensure dims are clean for audits too
            purge_mismatched_embeddings(audit_conn, get_expected_dim(embedder))
            best_idx, sims, chunk_indices, chunk_texts, page_score = score_keyword_url(audit_conn, embedder, kw_pick, u_pick)

        if sims.size == 0:
            st.warning("No chunks or embeddings found for this pair.")
        else:
            order = np.argsort(-sims)
            top = order[:topk_audit]
            rows = []
            for rank, j in enumerate(top, start=1):
                txt = chunk_texts[j]
                rows.append({
                    "rank": rank,
                    "chunk_index": int(chunk_indices[j]),
                    "cosine_similarity": float(sims[j]),
                    "chunk_preview": (txt.replace("\n", " ")[:220] + "…") if len(txt) > 220 else txt.replace("\n", " "),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(f"Whole-page cosine: {page_score:.3f}  •  Arg-max chunk index: {int(chunk_indices[int(np.argmax(sims))])}")
