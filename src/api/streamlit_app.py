"""PixSearch — Streamlit demo UI (Pehia-style aesthetic).

Run with:
    PYTHONPATH=. streamlit run src/api/streamlit_app.py

Backends supported (auto-detected from indexes/ folder):
  classical_combo  — color histogram + HOG + ORB (image query only)
  resnet50         — ResNet-50 frozen features   (image query only)
  clip             — CLIP ViT-B/32               (image OR text query)
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.classical.extractors import get_extractor
from src.index.faiss_index import FAISSIndex


st.set_page_config(
    page_title="PixSearch — Visual Search Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS — soft pastel gradient, glass cards, magazine-style typography
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  /* Hide Streamlit chrome */
  #MainMenu, footer, header {visibility: hidden;}

  /* Soft gradient background — lavender into pink into peach */
  .stApp {
    background:
      radial-gradient(circle at 0% 0%, rgba(229, 204, 255, 0.6) 0%, transparent 50%),
      radial-gradient(circle at 100% 0%, rgba(255, 209, 220, 0.5) 0%, transparent 50%),
      radial-gradient(circle at 100% 100%, rgba(255, 222, 200, 0.5) 0%, transparent 50%),
      radial-gradient(circle at 0% 100%, rgba(204, 229, 255, 0.4) 0%, transparent 50%),
      linear-gradient(135deg, #f6f0ff 0%, #fef0f5 50%, #fff5ed 100%);
    background-attachment: fixed;
    color: #2d2540;
    font-family: 'Inter', -apple-system, sans-serif;
  }

  .block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1500px !important;
  }

  /* Headings */
  h1, h2, h3, h4 {
    color: #2d2540;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.02em;
  }

  /* ────────────────────────────────────────────────────────────────────
     Top header strip — brand + post-ads-style accent button
  ──────────────────────────────────────────────────────────────────── */
  .top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
    padding: 0 8px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-logo {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, #c179e8, #ff7eb3, #ffa07a, #c179e8);
    box-shadow: 0 4px 16px rgba(193, 121, 232, 0.3);
  }
  .brand-name {
    font-size: 22px;
    font-weight: 800;
    color: #2d2540;
    letter-spacing: -0.02em;
  }
  .accent-pill {
    background: linear-gradient(135deg, #b478f5 0%, #ff7eb3 100%);
    color: #fff;
    padding: 8px 22px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    box-shadow: 0 4px 18px rgba(180, 120, 245, 0.35);
  }

  /* ────────────────────────────────────────────────────────────────────
     Section labels — small uppercase
  ──────────────────────────────────────────────────────────────────── */
  .section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #9b8eb5;
    margin-bottom: 14px;
  }

  /* ────────────────────────────────────────────────────────────────────
     Sidebar — soft white panel
  ──────────────────────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.5);
  }
  section[data-testid="stSidebar"] .stMarkdown { color: #2d2540; }
  section[data-testid="stSidebar"] h3 { color: #2d2540; font-weight: 700; }
  .stSelectbox label, .stTextInput label, .stRadio label, .stSlider label {
    color: #6b5d8a !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }

  /* Selectbox styling */
  div[data-baseweb="select"] > div {
    background: #fff !important;
    border: 1px solid rgba(180, 120, 245, 0.15) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(45, 37, 64, 0.04);
  }

  /* Stat pills (top bar) */
  .stat-pill {
    display: inline-flex;
    align-items: center;
    background: #fff;
    border-radius: 100px;
    padding: 6px 16px;
    margin-right: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #6b5d8a;
    box-shadow: 0 2px 12px rgba(45, 37, 64, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.8);
  }
  .stat-pill b { color: #2d2540; font-weight: 700; }
  .stat-pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: linear-gradient(135deg, #b478f5, #ff7eb3);
    display: inline-block;
    margin-right: 8px;
  }
  .clip-tag {
    background: linear-gradient(135deg, #b478f5, #ff7eb3);
    color: #fff;
    border-radius: 100px;
    padding: 2px 10px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-left: 8px;
    text-transform: uppercase;
  }

  /* ────────────────────────────────────────────────────────────────────
     Query card — large white card on the left
  ──────────────────────────────────────────────────────────────────── */
  .query-card {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(45, 37, 64, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.9);
    margin-bottom: 16px;
  }
  .query-title {
    font-size: 24px;
    font-weight: 800;
    color: #2d2540;
    margin-bottom: 4px;
  }
  .query-meta {
    color: #9b8eb5;
    font-size: 12px;
    margin-bottom: 16px;
  }

  /* File uploader skin */
  [data-testid="stFileUploader"] {
    background: linear-gradient(180deg, #fff 0%, #faf5ff 100%);
    border: 1.5px dashed rgba(180, 120, 245, 0.3);
    border-radius: 14px;
    padding: 12px;
  }
  [data-testid="stFileUploader"] section { border: none !important; }

  /* Text input skin */
  .stTextInput > div > div > input {
    background: #fff !important;
    border: 1px solid rgba(180, 120, 245, 0.2) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    color: #2d2540 !important;
  }

  /* ────────────────────────────────────────────────────────────────────
     Result cards — like the "trending" listings in the mockup
  ──────────────────────────────────────────────────────────────────── */
  .result-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 4px 20px rgba(45, 37, 64, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.95);
    margin-bottom: 14px;
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(180, 120, 245, 0.15);
  }
  .result-rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f0e6ff, #ffe0eb);
    color: #8b5cd6;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 3px 10px;
    border-radius: 100px;
    text-transform: uppercase;
  }
  .trending-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ffa07a, #ff7eb3);
    color: #fff;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 0.15em;
    padding: 3px 10px;
    border-radius: 100px;
    text-transform: uppercase;
    margin-left: 6px;
  }
  .result-label {
    color: #2d2540;
    font-size: 14px;
    font-weight: 700;
    margin-top: 8px;
    text-transform: capitalize;
  }
  /* "Star rating" — purple bar that fills based on cosine score */
  .score-bar-wrapper {
    background: rgba(180, 120, 245, 0.08);
    border-radius: 100px;
    height: 4px;
    margin: 6px 0 4px 0;
    overflow: hidden;
  }
  .score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #b478f5, #ff7eb3);
    border-radius: 100px;
  }
  .score-text {
    color: #9b8eb5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }
  .score-text b { color: #8b5cd6; font-weight: 700; }

  /* ────────────────────────────────────────────────────────────────────
     Right rail — vertical mini-cards for stats (like "International Ads")
  ──────────────────────────────────────────────────────────────────── */
  .rail-card {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 4px 16px rgba(45, 37, 64, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.9);
  }
  .rail-icon {
    width: 40px; height: 40px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 18px;
    color: #fff;
  }
  .rail-icon.purple { background: linear-gradient(135deg, #b478f5, #8b5cd6); }
  .rail-icon.pink   { background: linear-gradient(135deg, #ff7eb3, #f06292); }
  .rail-icon.peach  { background: linear-gradient(135deg, #ffa07a, #ff8c64); }
  .rail-icon.blue   { background: linear-gradient(135deg, #7eb8ff, #5c9eff); }
  .rail-label {
    font-size: 10px;
    color: #9b8eb5;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2px;
  }
  .rail-value {
    font-size: 14px;
    font-weight: 700;
    color: #2d2540;
  }
  .rail-section-title {
    font-size: 13px;
    font-weight: 700;
    color: #2d2540;
    margin-bottom: 12px;
    margin-top: 8px;
  }

  /* Highlighted (rank 1) result card — special look */
  .top-result-card {
    background: linear-gradient(135deg, #f7eaff 0%, #ffe9f1 100%);
    border-radius: 20px;
    padding: 16px;
    box-shadow: 0 12px 36px rgba(180, 120, 245, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.95);
    margin-bottom: 18px;
  }

  /* Radio button */
  div[role="radiogroup"] label {
    background: #fff;
    border-radius: 10px;
    padding: 8px 12px !important;
    margin-right: 8px;
    border: 1px solid rgba(180, 120, 245, 0.15);
    color: #2d2540 !important;
  }

  /* Slider */
  .stSlider > div > div > div > div { background: linear-gradient(90deg, #b478f5, #ff7eb3) !important; }

  /* About panel */
  .about-panel {
    background: linear-gradient(135deg, #f7eaff 0%, #ffe9f1 100%);
    border-radius: 16px;
    padding: 16px;
    font-size: 12px;
    color: #6b5d8a;
    line-height: 1.6;
    margin-top: 16px;
  }
  .about-panel b { color: #2d2540; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading index…")
def load_index(index_dir: str) -> FAISSIndex:
    return FAISSIndex.load(index_dir)


@st.cache_resource(show_spinner="Loading ResNet-50…")
def load_resnet():
    from src.deep.resnet import ResNet50Embedder
    return ResNet50Embedder()


@st.cache_resource(show_spinner="Loading CLIP ViT-B/32…")
def load_clip():
    from src.deep.clip_embedder import CLIPEmbedder
    return CLIPEmbedder()


def find_indexes(root: str = "indexes") -> dict[str, str]:
    root_path = Path(root)
    if not root_path.exists():
        return {}
    return {
        c.name: str(c) for c in sorted(root_path.iterdir())
        if c.is_dir() and (c / "index.faiss").exists()
    }


def embed_image_query(pil_image: Image.Image, index: FAISSIndex) -> np.ndarray:
    backend = index.backend_name
    if backend == "resnet50":
        return load_resnet().extract_from_pil(pil_image)
    if backend == "clip":
        return load_clip().embed_image_pil(pil_image)
    if backend.startswith("classical_"):
        ext_name = backend.replace("classical_", "")
        extractor = get_extractor(ext_name)
        rgb = np.array(pil_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return extractor.extract(bgr)
    raise ValueError(f"Unknown backend: {backend}")


def embed_text_query(text: str) -> np.ndarray:
    return load_clip().embed_text(text)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    # ── Top header — brand on the left, accent pill on the right
    st.markdown(
        """
        <div class="top-header">
          <div class="brand">
            <div class="brand-logo"></div>
            <div class="brand-name">PixSearch</div>
          </div>
          <div class="accent-pill">Visual Search Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar
    with st.sidebar:
        st.markdown("### Configuration")

        indexes = find_indexes()
        if not indexes:
            st.error(
                "No indexes found in `indexes/`.\n\nBuild one first:\n"
                "```\npython scripts/build_index.py \\\n"
                "  --data data/corel1k --backend clip \\\n"
                "  --out indexes/clip_1k\n```"
            )
            st.stop()

        chosen = st.selectbox("Retrieval backend", list(indexes.keys()))
        top_k = st.slider("Results to show", 4, 20, 10, 2)

        st.markdown(
            """
            <div class="about-panel">
              <b>PixSearch</b> compares classical (histogram + HOG + ORB)
              and deep learning (ResNet-50, CLIP) feature extractors over
              the Wang Corel dataset.<br><br>
              <b>CLIP</b> enables <b>text-to-image search</b> — type a
              description and find matching photos without any labels.<br><br>
              Vectors indexed with <b>FAISS</b> for sub-millisecond
              nearest-neighbor search.<br><br>
              <span style="color:#9b8eb5;">Muhammad Haroon · SZABIST</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    index = load_index(indexes[chosen])
    is_clip = (index.backend_name == "clip")

    # ── Three-column layout: query (left), results (center), rail (right)
    col_query, col_results, col_rail = st.columns([1.0, 2.0, 0.9], gap="medium")

    # ===========================================================
    # LEFT COLUMN — query card
    # ===========================================================
    embed_ms = 0.0
    query_vec = None
    query_label = ""

    with col_query:
        clip_tag = '<span class="clip-tag">text + image</span>' if is_clip else ""
        st.markdown(
            f"""
            <div class="section-label">// query</div>
            <div class="query-card">
              <div class="query-title">New Search{clip_tag}</div>
              <div class="query-meta">backend: <b>{index.backend_name}</b> · dim {index.dim} · {index.size:,} indexed</div>
            """,
            unsafe_allow_html=True,
        )

        if is_clip:
            query_mode = st.radio(
                "Query type",
                ["📷  Image", "✍️  Text"],
                horizontal=True,
                label_visibility="collapsed",
            )
        else:
            query_mode = "📷  Image"

        if "Image" in query_mode:
            uploaded = st.file_uploader(
                "Drop an image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
            )
            if uploaded:
                pil_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
                st.image(pil_image, use_container_width=True)
                t0 = time.time()
                query_vec = embed_image_query(pil_image, index)
                embed_ms = (time.time() - t0) * 1000
                query_label = uploaded.name
            else:
                st.info("Upload an image to begin.", icon="📁")
        else:
            text_input = st.text_input(
                "Describe what you're looking for",
                placeholder="e.g. a horse running in a field",
                label_visibility="collapsed",
            )
            if text_input and text_input.strip():
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg,#f7eaff,#ffe9f1);
                                border-radius:12px;padding:14px 16px;margin-top:8px;
                                font-style:italic;color:#6b5d8a;">
                      🔍 &nbsp; "{text_input}"
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                t0 = time.time()
                query_vec = embed_text_query(text_input.strip())
                embed_ms = (time.time() - t0) * 1000
                query_label = f'"{text_input}"'
            else:
                st.info("Type a description to search.", icon="✍️")

        st.markdown("</div>", unsafe_allow_html=True)

    # ===========================================================
    # CENTER COLUMN — results
    # ===========================================================
    with col_results:
        if query_vec is None:
            st.markdown(
                """
                <div class="section-label">// results</div>
                <div class="query-card" style="text-align:center;padding:60px 24px;">
                  <div style="font-size:48px;margin-bottom:16px;">🔎</div>
                  <div style="color:#9b8eb5;font-size:14px;">
                    Submit a query on the left to see the most similar images.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Run search
        t0 = time.time()
        hits = index.search(query_vec, k=top_k)
        search_ms = (time.time() - t0) * 1000
        total_ms = embed_ms + search_ms

        st.markdown('<div class="section-label">// most similar</div>', unsafe_allow_html=True)

        # Top result — special highlighted card (like the "Top 3 Rated" in mockup)
        if hits:
            top = hits[0]
            st.markdown('<div class="top-result-card">', unsafe_allow_html=True)
            top_cols = st.columns([1, 2], gap="medium")
            with top_cols[0]:
                try:
                    img = Image.open(top.image_path).convert("RGB")
                    st.image(img, use_container_width=True)
                except Exception:
                    pass
            with top_cols[1]:
                st.markdown(
                    f"""
                    <div style="padding:8px 4px;">
                      <span class="result-rank-badge">rank 01</span>
                      <span class="trending-badge">★ best match</span>
                      <div style="font-size:22px;font-weight:800;color:#2d2540;margin-top:14px;text-transform:capitalize;">
                        {top.label_name}
                      </div>
                      <div style="color:#9b8eb5;font-size:13px;margin-top:6px;">
                        Cosine similarity: <b style="color:#8b5cd6;">{top.score:.4f}</b>
                      </div>
                      <div class="score-bar-wrapper" style="height:6px;margin-top:14px;">
                        <div class="score-bar-fill" style="width:{max(0, top.score)*100:.1f}%;"></div>
                      </div>
                      <div style="margin-top:18px;">
                        <span class="stat-pill"><span class="dot"></span>embed {embed_ms:.1f} ms</span>
                        <span class="stat-pill"><span class="dot"></span>search {search_ms:.2f} ms</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # Remaining results — grid of 4 per row (cleaner than 5)
        rest = hits[1:]
        n_cols = 4
        for row_start in range(0, len(rest), n_cols):
            row_hits = rest[row_start:row_start + n_cols]
            cols = st.columns(n_cols, gap="small")
            for col, hit in zip(cols, row_hits):
                with col:
                    try:
                        img = Image.open(hit.image_path).convert("RGB")
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"load failed")

                    score_pct = max(0, hit.score) * 100
                    st.markdown(
                        f"""
                        <div style="padding:6px 2px 14px 2px;">
                          <span class="result-rank-badge">rank {hit.rank:02d}</span>
                          <div class="result-label">{hit.label_name}</div>
                          <div class="score-bar-wrapper">
                            <div class="score-bar-fill" style="width:{score_pct:.1f}%;"></div>
                          </div>
                          <div class="score-text">cos = <b>{hit.score:.4f}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ===========================================================
    # RIGHT RAIL — performance stats and class breakdown
    # ===========================================================
    with col_rail:
        st.markdown('<div class="section-label">// performance</div>', unsafe_allow_html=True)

        if query_vec is None:
            st.markdown(
                """
                <div class="rail-card">
                  <div class="rail-icon purple">⏱</div>
                  <div>
                    <div class="rail-label">embed time</div>
                    <div class="rail-value">— ms</div>
                  </div>
                </div>
                <div class="rail-card">
                  <div class="rail-icon pink">⚡</div>
                  <div>
                    <div class="rail-label">search time</div>
                    <div class="rail-value">— ms</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="rail-card">
                  <div class="rail-icon purple">⏱</div>
                  <div>
                    <div class="rail-label">embed time</div>
                    <div class="rail-value">{embed_ms:.1f} ms</div>
                  </div>
                </div>
                <div class="rail-card">
                  <div class="rail-icon pink">⚡</div>
                  <div>
                    <div class="rail-label">search time</div>
                    <div class="rail-value">{search_ms:.2f} ms</div>
                  </div>
                </div>
                <div class="rail-card">
                  <div class="rail-icon peach">📊</div>
                  <div>
                    <div class="rail-label">total latency</div>
                    <div class="rail-value">{total_ms:.1f} ms</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Class distribution of the top-K hits — bonus insight
            from collections import Counter
            counts = Counter(h.label_name for h in hits)
            st.markdown(
                '<div class="rail-section-title" style="margin-top:24px;">Top-K class mix</div>',
                unsafe_allow_html=True,
            )
            colors = ["purple", "pink", "peach", "blue"]
            emojis = {"africa":"🌍","beach":"🏖","buildings":"🏛","buses":"🚌",
                      "dinosaurs":"🦖","elephants":"🐘","flowers":"🌸",
                      "horses":"🐎","mountains":"⛰","food":"🍽"}
            for i, (name, count) in enumerate(counts.most_common(5)):
                emoji = emojis.get(name, "•")
                color = colors[i % len(colors)]
                st.markdown(
                    f"""
                    <div class="rail-card">
                      <div class="rail-icon {color}">{emoji}</div>
                      <div>
                        <div class="rail-label">{name}</div>
                        <div class="rail-value">{count} / {len(hits)}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
