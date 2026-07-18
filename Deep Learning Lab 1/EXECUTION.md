# Execution Instructions

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo>
```

## 2. Set up a Python environment

It's recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Get the dataset

The notebook first tries to download the dataset directly from the UCI repository. If that fails (e.g. blocked network, restricted environment), it automatically falls back to the local file `data_banknote_authentication.csv` included in this repo — no action needed either way.

If you want to fetch it manually instead:
```bash
# Download from UCI
curl -O https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip
unzip banknote+authentication.zip
```

## 5. Run the notebook

```bash
jupyter notebook DL_Lab_1.ipynb
```

Run all cells from top to bottom (**Kernel → Restart & Run All**) to reproduce every result in order — EDA plots, preprocessing, model training, evaluation metrics, learning-rate comparison, and the scikit-learn comparison.

## 6. Generating report-ready figures

Each plotting cell automatically saves a `.eps` file into the working directory (e.g. `training_error.eps`, `confusion_matrix.eps`) at 600 dpi with bold axis labels/ticks/legends, styled for direct use in a LaTeX report.

If you're embedding these in Overleaf and run into compile timeouts (common on the free tier for large raster plots like the confusion matrix, heatmap, and pairplot), convert them to compressed PDFs first:

```bash
for f in *.eps; do
  gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
     -dPDFSETTINGS=/ebook -dDownsampleColorImages=true -dColorImageResolution=300 \
     -sOutputFile="${f%.eps}.pdf" "$f"
done
```
Then reference the `.pdf` files in your `.tex` document instead of the `.eps` files.

## 7. Expected runtime

The full notebook (including the from-scratch perceptron training, the 3-way learning-rate comparison, and the sklearn comparison) runs in well under a minute on a standard laptop CPU — no GPU required.

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again inside the active virtual environment |
| Plots show wrong font instead of Times New Roman | Times New Roman isn't installed on most Linux systems; the notebook automatically falls back to Liberation Serif (a metrically identical free substitute) and prints which font it used |
| UCI dataset URL unreachable | The notebook automatically falls back to the bundled `data_banknote_authentication.csv` |
