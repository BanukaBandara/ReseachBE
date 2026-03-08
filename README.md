# FoundHere_Mobile_Application_API

## Run (Backend + ML together)

This repo includes a Python Growth Monitoring ML model under `growthMonitoring_ml/`.
The Node backend auto-starts the local ML API on `http://127.0.0.1:8001` when you run the server, so you do **not** need to run the model separately.

### 1) Install Node deps

`npm install`

### 2) Install Python deps (one time)

From the repo root:

`python -m pip install -r growthMonitoring_ml/requirements.txt`

### 3) Start dev server

`npm run dev`

### Environment

- `PORT` (default `3001`)
- `GROWTH_PYTHON_API_URL` (default `http://127.0.0.1:8001/predict`)
- Optional:
	- `START_GROWTH_ML=true|false` (force enable/disable auto-start)
	- `PYTHON_BIN` (override python executable, e.g. `py`)
	- `GROWTH_ML_CHECKPOINT` (path to `.pt` file)
