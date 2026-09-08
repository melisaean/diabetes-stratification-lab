# Diabetes Stratification Lab

End-to-end ML pipeline that discovers patient personas from the UCI Diabetes 130-US Hospitals dataset, maps them in an interactive browser, and lets clinicians explore "Neural Twins" — the five most similar historical cases for any patient.

## Quick Start

```bash
# Clone
git clone https://github.com/melisaean/diabetes-stratification-lab.git
cd diabetes-stratification-lab

# Create venv (Python 3.11+)
python3.11 -m venv .venv && source .venv/bin/activate

# Install
pip install -e ".[all]"

# Fetch data + run pipeline (downloads ~180 MB on first run)
make data
make pipeline

# Start server
make serve          # http://localhost:8000/ui
```

## What's Inside

| Stage | What it does | Key files |
|-------|-------------|-----------|
| **Data** | Fetches UCI dataset, drops high-null columns, maps ICD-9 codes | `scripts/fetch_data.py`, `src/personas/data.py` |
| **Features** | Ordinal + frequency encoding → MinMaxScaler → engineered matrix | `src/personas/features.py` |
| **Autoencoder** | 256→128→64→32 latent bottleneck; MSE + KL loss; early stopping | `src/personas/models.py` |
| **Clustering** | KMeans on latent space → persona labels + KNN twin index | `src/personas/clustering.py` |
| **Projection** | UMAP 2D embedding for the interactive map | `src/personas/dimensionality.py` |
| **Serving** | FastAPI + Plotly scatter + "Neural Twins" + new-patient inference | `app/` |
| **Evaluation** | Reconstruction metrics, cluster sweeps, bootstrap stability, persona profiles | `src/personas/evaluation.py` |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/map` | Cached Plotly scatter JSON (UMAP 2D projection, coloured by persona) |
| `GET` | `/persona/{id}` | Aggregate stats for a persona (avg stay, meds, readmission risk) |
| `GET` | `/patient/{idx}/twins` | 5 nearest clinical neighbours (Neural Twins) |
| `POST` | `/encode` | New-patient inference → persona + readmission risk + twins |
| `GET` | `/ui` | Interactive single-page browser UI |
| `GET` | `/health` | Health check |

## Make Targets

```bash
make data          # Fetch raw UCI dataset
make pipeline      # Run full ML pipeline
make evaluate      # Run evaluation harness
make serve         # Start FastAPI server on :8000
make test          # Run pytest
make lint          # Ruff check
make typecheck     # Mypy
make build         # Build wheel
make clean         # Remove all generated outputs
```

## Docker

```bash
docker compose up --build    # http://localhost:8000/ui
```

## Project Structure

```
diabetes-stratification-lab/
├── src/personas/          # Core ML library
├── app/                   # FastAPI serving layer
├── scripts/               # CLI entry points
├── tests/                 # Unit tests
├── data/                  # Raw (gitignored) + serve-assets (committed)
├── outputs/               # Model artefacts
├── Makefile               # Build automation
├── Dockerfile             # Multi-stage production image
└── pyproject.toml         # Package + deps
```

## Tech Stack

- **Python 3.11+**, PyTorch, scikit-learn, UMAP-learn, Plotly, FastAPI, Pydantic v2
- **Dev**: pytest, ruff, mypy
- **Infra**: Docker, GitHub Actions CI

## License

MIT