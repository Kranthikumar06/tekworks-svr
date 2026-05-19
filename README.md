# SVR on California Housing

This folder contains a Jupyter notebook that trains a Support Vector Regression (SVR) model on the California housing dataset from scikit-learn and saves the trained model as a pickle file.

## Files
- svr.ipynb: Data loading, outlier handling, model training, and evaluation.
- svr_model.pkl: Trained SVR model saved from the notebook.
- requirements.txt: Python dependencies for running the notebook.

## Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
Open the notebook and run the cells in order:

```bash
jupyter notebook svr.ipynb
```

If you use VS Code, open the notebook and run all cells.

## Notes
- The notebook uses `fetch_california_housing()` from scikit-learn; no external dataset download is needed.
- The model is saved to `svr_model.pkl` in the same folder.
