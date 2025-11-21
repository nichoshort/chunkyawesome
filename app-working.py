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

# --- Embeddings: EmbeddingGemma via Sentence Transformers ---
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfFolder

# --- Clean/pack utilities ---
import re, html

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Chunk Daly", layout="wide")
st.title("🔎 Chunk Daly")

with st.sidebar:
    st.header("Configuration")
    db_path = st.text_input("SQLite DB path", value="chunks.db")
    crawl_concurrency = st.slider("Max concurrent crawls", 1, 10, 10)
    model_name = st.text_input(
        "Embedding model",
        value="google/embeddinggemma-300m",
        help="Loads via sentence-transformers. Requires HF access to google/embeddinggemma-300m."
    )
    chunk_max_chars = st.number_input("Max characters per chunk", 500, 4000, 1400, 100)
    force_recrawl = st.toggle("Force recrawl URLs (refresh HTML cache)", value=False)
    rechunk_existing = st.toggle("Re-chunk existing pages (no recrawl)", value=False,
                                 help="Rebuild chunks from stored HTML even if chunks already exist.")

uploaded = st.file_uploader("Upload a CSV with columns: keyword,url", type=["csv"])
run_btn = st.button("Run matching")

# ======================
# Text cleanup + packing
# ======================
def normalize_text(s: str) -> str:
    """Fix HTML entities + whitespace while preserving paragraphs."""
    if not s:
        return ""
    s = html.unescape(s)
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    return s.strip()

def pack_chunks(chunks: List[str], min_chars=400, max_chars=1400) -> List[str]:
    """
    Merge adjacent small chunks into ~paragraph-sized passages.
    Targets [min_chars, max_chars] when possible.
    """
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
# SQLite helpers
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
  scope TEXT,      -- 'keyword' or 'chunk'
  key TEXT,        -- keyword text OR url|chunk_index
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
  best_chunk_text TEXT,
  UNIQUE(keyword, url)
);
"""

def db_connect(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def db_init(conn: sqlite3.Connection):
    conn.executescript(DDL)

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

def db_write_result(conn: sqlite3.Connection, keyword: str, url: str, idx: int, score: float, text: str):
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO results (keyword, url, best_chunk_index, best_score, best_chunk_text) VALUES (?, ?, ?, ?, ?)",
            (keyword, url, idx, float(score), text),
        )

def db_read_results(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT keyword, url AS landing_page, best_score AS cosine_similarity, best_chunk_text AS chunk_text "
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
            headers={"User-Agent": "Mozilla/5.0 (Qforia-Chonkie/1.0)"}
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
    """
    Try LayoutAwareChunker on HTML; fallback to Semantic chunking on cleaned text;
    then pack into paragraphs so we don't end up with tiny one-sentence chunks.
    """
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
# Embeddings (EmbeddingGemma via Sentence Transformers)
# ======================
class STEEmbedder:
    def __init__(self, model_name: str):
        # Try token from env or HF cache (after `huggingface-cli login`)
        token = os.getenv("HF_TOKEN") or HfFolder.get_token()
        try:
            try:
                self.model = SentenceTransformer(model_name, token=token, trust_remote_code=True)
            except TypeError:
                self.model = SentenceTransformer(model_name, use_auth_token=token, trust_remote_code=True)
        except Exception as e:
            msg = (
                "Could not initialize SentenceTransformer for a (possibly gated) model.\n\n"
                "Tips:\n"
                "  • Ensure you've accepted the license & have access on Hugging Face.\n"
                "  • Run `huggingface-cli login` OR set HF_TOKEN environment variable.\n"
                f"Underlying error: {e}"
            )
            raise RuntimeError(msg)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        vecs = self.model.encode(
            [normalize_text(t) for t in texts],
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False
        )
        return vecs.astype(np.float32)

# ======================
# Cosine similarity helpers
# ======================
def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

def best_match_for_keyword(keyword_vec: np.ndarray, chunk_vecs: np.ndarray) -> Tuple[int, float]:
    if chunk_vecs.size == 0:
        return -1, 0.0
    sims = (chunk_vecs @ keyword_vec.reshape(-1, 1)).ravel()
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])

# ======================
# Pipeline
# ======================
def run_pipeline(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    max_conc: int,
    embedder: STEEmbedder,
    max_chars_hint: int,
    force_recrawl: bool,
    rechunk_existing: bool
):
    # Clean input
    df = df.dropna(subset=["keyword", "url"]).copy()
    df["keyword"] = df["keyword"].str.strip()
    df["url"] = df["url"].str.strip()
    df = df[df["url"].str.startswith(("http://", "https://"))].drop_duplicates(subset=["keyword", "url"])
    needed_urls = sorted(set(df["url"].tolist()))

    # Crawl (skip cached unless force)
    if not force_recrawl:
        cached = {u for (u,) in conn.execute("SELECT url FROM pages")}
        to_fetch = [u for u in needed_urls if u not in cached]
    else:
        to_fetch = needed_urls

    if to_fetch:
        st.info(f"Crawling {len(to_fetch)} URL(s)…")
        results = asyncio.run(crawl_urls(to_fetch, max_conc))
        for u, (code, html_text) in results.items():
            db_write_page(conn, u, code, html_text.encode("utf-8", errors="ignore") if html_text else b"")

    # Chunk (with rebuild logic)
    st.info("Chunking pages (layout-aware; full-page pack fallback)…")
    for u in needed_urls:
        # Inspect existing chunks
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(text)),0) FROM chunks WHERE url=?", (u,)
        ).fetchone()
        chunk_count = row[0] or 0
        total_len = row[1] or 0

        # HEALTHY if already ≥2 chunks OR total text is sizable.
        healthy = (chunk_count >= 2) or (total_len >= max_chars_hint * 1.5)

        if healthy and not force_recrawl and not rechunk_existing:
            # keep existing chunks
            pass
        else:
            # Rebuild chunks from stored HTML (no recrawl needed)
            html_src = db_read_page_html(conn, u) or ""
            chunks = layout_aware_chunk(html_src, max_chars_hint) if html_src else []

            # Full-page pack fallback (NO TRUNCATION)
            if not chunks and html_src:
                text = extract_main_text(html_src)
                if text:
                    paras = [p.strip() for p in text.split("\n") if p.strip()]
                    min_chars = max(350, int(max_chars_hint * 0.35))
                    chunks = pack_chunks(paras, min_chars=min_chars, max_chars=max_chars_hint)

            # If still nothing, keep as empty
            with conn:
                conn.execute("DELETE FROM chunks WHERE url=?", (u,))
                conn.execute("DELETE FROM embeddings WHERE scope='chunk' AND key LIKE ?", (f"{u}|%",))
            if chunks:
                db_write_chunks(conn, u, chunks)

    # Embed chunks (only missing)
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
            vecs = embedder.embed(missing_texts)
            for k, v in zip(missing_keys, vecs):
                db_write_embedding(conn, "chunk", k, v)

    # Embed keywords (only missing)
    st.info("Embedding keywords…")
    uniq_kws = sorted(set(df["keyword"].tolist()))
    missing_kws = [kw for kw in uniq_kws if not db_has_embedding(conn, "keyword", kw)]
    if missing_kws:
        kw_vecs = embedder.embed(missing_kws)
        for kw, v in zip(missing_kws, kw_vecs):
            db_write_embedding(conn, "keyword", kw, v)

    # Score best chunk per (keyword, url) — dense cosine
    st.info("Scoring best chunks…")
    for _, r in df.iterrows():
        kw, u = r["keyword"], r["url"]
        # Skip if already computed
        if conn.execute("SELECT 1 FROM results WHERE keyword=? AND url=?", (kw, u)).fetchone():
            continue

        kw_vec = db_read_embedding(conn, "keyword", kw)
        url_chunks = db_get_chunks(conn, u)
        if kw_vec is None or not url_chunks:
            db_write_result(conn, kw, u, -1, 0.0, "")
            continue

        chunk_map = dict(url_chunks)
        chunk_keys = [f"{u}|{idx}" for idx, _ in url_chunks]
        chunk_vecs = []
        for ck in chunk_keys:
            v = db_read_embedding(conn, "chunk", ck)
            if v is None:
                idx = int(ck.split("|")[-1])
                text = normalize_text(chunk_map[idx])
                v = embedder.embed([text])[0]
                db_write_embedding(conn, "chunk", ck, v)
            chunk_vecs.append(v)
        chunk_mat = np.vstack(chunk_vecs).astype(np.float32)

        # cosine after L2-normalization
        kw_vec_n = l2_normalize(kw_vec.reshape(1, -1))[0]
        chunk_mat_n = l2_normalize(chunk_mat)
        sims = (chunk_mat_n @ kw_vec_n.reshape(-1,)).astype(np.float32)
        best_idx = int(np.argmax(sims))
        best_key = chunk_keys[best_idx]
        best_index = int(best_key.split("|")[-1])
        best_text = normalize_text(chunk_map.get(best_index, ""))

        db_write_result(conn, kw, u, best_idx, float(sims[best_idx]), best_text)

# ======================
# UI flow
# ======================
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        cols = {c.lower(): c for c in df_in.columns}
        if "keyword" not in cols or "url" not in cols:
            st.error("CSV must include columns: keyword,url")
            df_in = None
        else:
            df_in = df_in.rename(columns={cols["keyword"]: "keyword", cols["url"]: "url"})
            st.write("Sample of input:")
            st.dataframe(df_in.head(20))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df_in = None
else:
    df_in = None

if run_btn:
    if df_in is None or df_in.empty:
        st.error("Please upload a valid CSV first.")
    else:
        with closing(db_connect(db_path)) as conn:
            db_init(conn)
            try:
                embedder = STEEmbedder(model_name)
            except Exception as e:
                st.error(str(e))
                st.stop()

            with st.spinner("Processing…"):
                run_pipeline(
                    df=df_in,
                    conn=conn,
                    max_conc=crawl_concurrency,
                    embedder=embedder,
                    max_chars_hint=chunk_max_chars,
                    force_recrawl=force_recrawl,
                    rechunk_existing=rechunk_existing
                )

            st.success("Done.")
            out_df = db_read_results(conn)

            # Add a clean preview column for table
            def preview(s: str, n=220):
                s = normalize_text(s).replace("\n", " ")
                return (s[:n] + "…") if len(s) > n else s

            out_df["chunk_preview"] = out_df["chunk_text"].apply(preview)

            st.subheader("Results")
            st.dataframe(
                out_df[["keyword", "landing_page", "cosine_similarity", "chunk_preview"]],
                use_container_width=True
            )

            # Expanders for full text
            st.markdown("---")
            st.subheader("Full Matches")
            for _, row in out_df.iterrows():
                with st.expander(f"{row['keyword']} — {row['landing_page']} (score: {row['cosine_similarity']:.3f})"):
                    st.write(row["chunk_text"])

            # Export CSV
            csv_buf = io.StringIO()
            out_df.to_csv(csv_buf, index=False)
            st.download_button(
                "Download results as CSV",
                data=csv_buf.getvalue(),
                file_name="keyword_chunk_matches.csv",
                mime="text/csv"
            )

            st.caption("Columns: keyword, landing_page, cosine_similarity, chunk_text, chunk_preview")
