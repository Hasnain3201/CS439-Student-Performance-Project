# CS439 Student Performance Project

Predicting high-school students' final grades (G3) using demographic factors, study habits, attendance, and early-term grades from the UCI Student Performance dataset. Built for CS439 to emphasize transparent, interpretable modeling and responsible evaluation.

## Project Overview
- Goal: forecast the final grade (G3) so instructors can intervene early with targeted support.
- Dataset: 649 Portuguese high-school students, 53 features (categorical and numeric), no missing values. Source: UCI ML Repository (student-por.csv included).
- Approach: thorough EDA, feature engineering to capture effort and momentum, and a reproducible preprocessing + modeling pipeline in scikit-learn.
- Models compared: Linear Regression, Ridge, Lasso, Gradient Boosting. Lasso (alpha=0.1) performed best with MAE ≈ 0.71 and R² ≈ 0.87 on the held-out test split.
- Evaluation: 80/20 train-test split with 5-fold cross-validation; residual diagnostics; subgroup error checks by sex and school to watch for bias.

## Repository Contents
- `student_performance.ipynb` — main notebook with EDA, feature engineering, modeling, diagnostics, and fairness checks.
- `student_performance.ipynb.zip` — zipped copy of the notebook (course submission backup).
- `student-por.csv` — UCI Student Performance dataset (Portuguese course section).
- `finalreport.pdf` — full written report detailing methods, results, and findings.
- `Project_guidelines_CS_439.pdf` — course project instructions.

## Key Methods and Features
- Data prep: ColumnTransformer + Pipeline to one-hot encode categoricals and standardize numerics to avoid scale-driven bias in linear models.
- Feature engineering highlights:
  - Study-time-to-attendance ratio to reward consistent class engagement.
  - Combined G1/G2 performance score to smooth early-term noise.
  - G2–G1 delta to capture academic momentum.
- Model selection: prioritized interpretability and simplicity; Lasso chosen for strong performance and built-in feature selection.
- Reliability checks: predicted vs. actual plots, residual analysis for patterns, and subgroup MAE/MSE comparisons (sex, school) to flag disparities.

## How to Run the Notebook
1) Requirements: Python 3 with `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, and `jupyter` installed. Example:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install pandas numpy scikit-learn seaborn matplotlib jupyter
   ```
2) Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook student_performance.ipynb
   ```
3) Ensure `student-por.csv` stays in the project root. Run all cells to reproduce the analysis, visualizations, and metrics.

## What You’ll See in the Notebook
- EDA: histograms of key numerics (grades, study time, absences), correlation heatmaps, scatterplots, and boxplots to spot outliers and trends.
- Modeling workflow: train/test split, cross-validation, and comparison of linear baselines versus tree-based Gradient Boosting.
- Results: Lasso chosen as the final model (test MAE ≈ 0.71, R² ≈ 0.87); coefficients inspected for interpretability and feature importance.
- Fairness checks: subgroup error and signed error comparisons by sex and school; no meaningful disparity observed in the tested groups.

## Reproducibility Notes
- Randomness: scikit-learn splits seeded in the notebook to keep results stable.
- Data integrity: dataset has no missing values; categorical encodings and scaling are applied consistently through the pipeline to prevent leakage.
- Transparency: coefficients from the linear models are inspected to explain how features drive predictions and to keep the approach classroom-ready.
