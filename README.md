# Sentiment-Analysis (scaffold)

Purpose
- Build and compare sentiment classifiers (negative / neutral / positive) on social media comments.
- Baseline: TF-IDF + LogisticRegression
- Embeddings: averaged static embeddings (glove-twitter-25) + PyTorch MLP

Project structure
- data/            (data & caches)
- models/          (saved models)
- reports/         (evaluation outputs)
- notebooks/       (experiment.ipynb)
- src/             (project source)
  - preprocess.py
  - data_loader.py
  - train_baseline.py
  - train_embeddings.py
  - evaluate.py
  - utils.py (optional)
- requirements.txt

Quick setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt