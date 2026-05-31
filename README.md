# PixSearch — Content-Based Image Retrieval

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)
![FAISS](https://img.shields.io/badge/FAISS-CPU-5A5868.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)
![HF Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow.svg)

A visual search engine that retrieves images by visual similarity, not by filename or tags. Three feature backends — a classical pipeline (HSV + HOG + ORB), a frozen ResNet-50, and CLIP for text queries — feed a single FAISS index and a Streamlit UI.

**Live demo:** https://huggingface.co/spaces/haroon8124/pixsearch

<!-- Add a screenshot of the running app here. Suggested filename: docs/screenshot.png -->
<!-- ![PixSearch UI](docs/screenshot.png) -->

## Features

- **Image-to-image search** using either a classical pipeline (color histogram + HOG + ORB) or a frozen ResNet-50 backbone.
- **Text-to-image search** using OpenAI CLIP (ViT-B/32), zero-shot.
- **Sub-millisecond retrieval** via FAISS `IndexFlatIP` over L2-normalised embeddings (cosine similarity).
- **Streamlit UI** with backend selector, top-K slider, and per-query latency readout.
- **Reproducible evaluation** with Precision@K, Recall@K, and mean Average Precision on Wang Corel-1K.

## Results

Wang Corel-1K, K = 10, 1,000 images split across 10 semantic classes.

| Backend                       | P@10 | R@10 | mAP  |
|-------------------------------|:----:|:----:|:----:|
| Classical (HSV + HOG + ORB)   | 0.56 | 0.056 | 0.49 |
| ResNet-50 (frozen, 2048-d)    | 0.86 | 0.086 | 0.80 |

ResNet features beat the classical baseline by **+30% absolute on P@10**. CLIP is evaluated separately on text-to-image queries.

## Architecture

```
Image / text query
       │
       ▼
 Feature extractor   (classical | resnet | clip)
       │
       ▼
 L2 normalise → FAISS IndexFlatIP  (inner product = cosine)
       │
       ▼
 Top-K results + similarity scores → Streamlit UI
```

## Project Structure

```
pixsearch/
├── app.py              # Streamlit entry point
├── src/                # Feature extractors, indexer, query logic
├── scripts/            # Index-building scripts
├── indexes/            # Saved FAISS indexes (.faiss + .npy metadata)
├── data/demo/          # Demo images bundled with the app
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/haroon-ai1/Visual-Search-Engine-pixsearch.git
cd Visual-Search-Engine-pixsearch
```

### 2. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get the dataset

Download the Wang Corel-1K dataset (10 classes × 100 images each) and place it under:

```
data/corel1k/<class_name>/*.jpg
```

### 4. Build an index

Pick the backend you want and run the corresponding command:

```bash
# Classical pipeline (HSV + HOG + ORB)
python scripts/build_index.py --data data/corel1k --backend classical --out indexes/classical

# ResNet-50 deep features
python scripts/build_index.py --data data/corel1k --backend resnet --out indexes/resnet

# CLIP (ViT-B/32)
python scripts/build_index.py --data data/corel1k --backend clip --out indexes/clip
```

### 5. Run the app

```bash
PYTHONPATH=. streamlit run app.py
```

The app opens at `http://localhost:8501`. Pick a backend in the sidebar, upload an image or type a text query (CLIP only), and adjust top-K.

## Tech Stack

- **Deep learning:** PyTorch, torchvision (ResNet-50), open_clip (ViT-B/32)
- **Classical CV:** OpenCV, scikit-image, NumPy
- **Vector search:** FAISS (CPU)
- **UI:** Streamlit + custom CSS
- **Evaluation:** scikit-learn (P@K, R@K, mAP)
- **Deployment:** Hugging Face Spaces (Docker runtime)

## Evaluation

To reproduce the numbers above:

```bash
python scripts/evaluate.py --index indexes/resnet --data data/corel1k --k 10
```

This computes P@K, R@K, and mAP using the class folder as the relevance signal.

## Author

**Muhammad Haroon**
B.S. Artificial Intelligence, SZABIST Islamabad
[LinkedIn](https://linkedin.com/in/haroon-ai) · [GitHub](https://github.com/haroon-ai1)

Built as the final project for AICL-3602 Computer Vision Lab, Spring 2026.

## License

MIT. See [`LICENSE`](LICENSE) for details.
