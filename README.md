# Machine Learning Models for Predicting RPA Project Duration

Code for the Master's dissertation **"Modelos de Machine Learning para
Previsão de Duração de Projetos de RPA"** (Diogo Ribeiro Santos, MSc in
Systems Engineering, University of Minho, 2026).

The study frames the prediction of *Robotic Process Automation* project
duration as an **ordinal classification** problem (5 duration classes),
comparing Random Forest, multinomial Logistic Regression, SVM (RBF) and
Ordinal Logistic Regression (*proportional odds*, implemented directly in
`src/ordinal_model.py`) on a database of n = 52
real projects, using stratified cross-validation (k = 3) and metrics suited
to ordinal problems (class-distance MAE and Quadratic Weighted Kappa) in
addition to standard metrics (accuracy and weighted F1).

## Data

**The original database is not included in this repository**, as it contains
confidential information from the company where the study was conducted. The
pipeline expects the file `Original.xlsx` in `data/raw/`; without it, the
scripts abort with an explanatory message. The full code is nevertheless
published to document the methodology and allow its application to analogous
data.

## Language note

The code (docstrings, comments, console output) is written in English.
Generated artefacts — figure labels, titles and LaTeX table captions — are
intentionally in **Portuguese**, so that they match the figures and tables
of the dissertation, which is written in Portuguese.

## Structure

```
├── data/
│   ├── raw/            # Original.xlsx (not versioned)
│   └── processed/      # intermediate versions V2..V9 (generated)
├── outputs/
│   ├── figures/        # univariate/ bivariate/ modeling/
│   └── tables/         # LaTeX tables (.tex) and result CSVs
└── src/
    ├── config.py           # paths, constants, plot style
    ├── latex_utils.py      # LaTeX table generation (shared)
    ├── evaluation.py       # metrics + cross-validation (shared)
    ├── 01_preprocess.py    # Original.xlsx -> V2..V9
    ├── 02_univariate.py    # univariate analysis (dissertation §3.3)
    ├── 03_bivariate.py     # bivariate analysis + selection (§3.5)
    ├── 04_modeling.py      # modelling and evaluation (§3.6 and Ch. 4)
    ├── 05_extra_figures.py # complementary charts for Chapter 4
    ├── ordinal_model.py    # proportional-odds ordinal regression (shared)
    └── app.py              # interactive prediction app (Chapter 5)
```

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts from the repository root, in numbered order:

```bash
python src/01_preprocess.py
python src/02_univariate.py
python src/03_bivariate.py
python src/04_modeling.py
python src/05_extra_figures.py
```

### Note on ordering (V8 → bivariate → V9)

The final preprocessing stage (generation of V9) depends on the **results of
the bivariate analysis**, which in turn runs on V8. The list of variables
removed in V9 is fixed in `01_preprocess.py` (`POST_BIVARIATE_DROPS`),
reflecting the results reported in the dissertation — which is why the
commands above work in a single pass.

To reproduce the study step by step (and verify that list independently):

```bash
python src/01_preprocess.py --stop-at-v8   # generates only V2..V8
python src/03_bivariate.py                 # analysis on V8
python src/01_preprocess.py                # generates V9
python src/04_modeling.py
```

## Interactive prediction app

`src/app.py` is a [Streamlit](https://streamlit.io) application (the Chapter 5 deliverable) where you enter the characteristics of a new RPA project and obtain the predicted duration class, with per-class probabilities, using the Ordinal Logistic Regression model. The model is retrained on the full V9 dataset at startup (sub-second), so no serialised model is shipped.

```bash
python src/01_preprocess.py   # once, to generate data/processed/V9
streamlit run src/app.py
```

The app UI is in Portuguese, matching the dissertation and the organisational context of the study.

## Reproducibility

All stochastic procedures (cross-validation, Random Forest, chart jitter)
use fixed seeds (`RANDOM_STATE = 42` in `src/config.py`), so results are
deterministically reproducible on the same input data.

## License

The code is distributed under the MIT license (see `LICENSE`). The database
is not covered by this license and is not distributed.
