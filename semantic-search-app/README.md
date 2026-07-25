# Nearest — Semantic Similarity Search

A complete, runnable web app around your Word2Vec + cosine-similarity search
notebook: a FastAPI backend that ports the notebook's pipeline into an API,
and a custom front end to search and upload datasets from the browser.

```
semantic-search-app/
├── backend/
│   ├── main.py            FastAPI app (routes, static file serving)
│   ├── search_engine.py   Preprocessing, Word2Vec training, search logic
│   └── artifacts/         Trained model cache (created automatically)
├── frontend/
│   ├── index.html
│   └── assets/
│       ├── style.css
│       └── app.js
├── data/
│   └── sample_dataset.csv Small demo dataset (cloud-security articles)
└── requirements.txt
```

## 1. Setup (VS Code)

Open this folder in VS Code, then in its integrated terminal:

```bash
# create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

**If `pip install` tries to compile pandas/numpy from source** (you'll see
`meson`, `vswhere.exe`, or "Microsoft Visual Studio" in the error), it means
pip couldn't find a prebuilt wheel for your Python version and is falling
back to building from source — which needs Visual Studio's C++ build tools.
The fix is almost always to use a Python version with good wheel coverage
(3.11 or 3.12), not to install those build tools:

```bash
# check what's installed
py -0                     # Windows: lists installed Python versions

# create the venv with a specific version, e.g. 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you only have a very new Python (3.13+) installed, download 3.12 from
python.org first — brand-new Python releases often don't have wheels yet
for scientific packages like pandas, numpy, and gensim.

## 2. Run

```bash
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

On first launch the app automatically trains on the bundled
`data/sample_dataset.csv` so you have something to search immediately.
The trained model, embeddings, and dataframe are cached under
`backend/artifacts/`, so restarting the server doesn't retrain from scratch.

## 3. Use your own data

In the UI, drag a CSV onto the **"Bring your own dataset"** panel (or click
to browse). Your CSV needs three columns — column names are case-insensitive:

| Title | Content | Keywords |
|---|---|---|
| Short headline for the document | The full text to search over | Comma-separated tags |

Uploading retrains the Word2Vec model on your data and replaces the sample
dataset. This can take anywhere from under a second (dozens of rows) to a
minute or two (tens of thousands of rows).

## 4. API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/status` | `{ ready, num_documents }` |
| `POST` | `/api/upload` | multipart `file` field (CSV) → trains a new model |
| `POST` | `/api/search` | JSON `{ query, top_k }` → ranked results |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I secure cloud storage?", "top_k": 5}'
```

## How it matches your notebook

`backend/search_engine.py` mirrors the notebook's pipeline exactly:

1. **Clean** — lowercase, strip URLs/HTML/punctuation (`clean_text`)
2. **Combine** — `title + keywords + content` → `combined_text`
3. **Tokenize** — simple regex + `.split()`
4. **Train** — `gensim.models.Word2Vec(sg=1, vector_size=100, window=5, epochs=20)`
5. **Embed** — mean of word vectors per document
6. **Search** — cosine similarity between the query vector and every document vector

The only behavioral change: `min_count` is set to `1` instead of `2` so the
model still works on small demo/test datasets — bump it back up in
`search_engine.py` if you're training on a large, noisy corpus.

## Notes

- No external API calls or API keys — everything (model training + inference)
  runs locally via `gensim` and `scikit-learn`.
- To retrain from scratch, delete the contents of `backend/artifacts/` and
  restart the server, or just upload a new CSV.
