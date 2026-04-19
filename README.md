# 2026dsa_finalproject_group8
## Data Engineering
This project prepares training and testing data for downstream machine learning models for the final course project.

### Main Files
- `code/CRSS8030_Final_project_group8_data_engineering.qmd` — main data engineering workflow
- `code/CRSS8030_Final_project_group8_data_engineering.html` — rendered report

### Folder Structure
- `data/` — raw training and testing input files
- `output/final_model_files/` — final model-ready datasets
- `output/weather_files/` — weather support and intermediate weather files
- `output/metadata_files/` — cleaned trait, soil, and metadata files
- `output/summary_files/` — summary and documentation files

### Modeling Notes
- Response variable: `yield_mg_ha`
- Training data contain observed yield
- Testing data retain matching predictors but do not contain observed yield for modeling
- Main model-ready datasets are base, yearly weather, and monthly weather

### Partner Handoff
Use `output/final_model_files/` for downstream modeling. Supporting files are organized into weather, metadata, and summary subfolders.

### Large Files
Large generated monthly-weather CSVs were excluded from GitHub due to file size limits and should be shared separately if needed.


### Model training

#### 1. Preprocessing (tidymodels recipe)
- Flagged unseen factor levels (`step_novel`)
- Lumped rare categories below 0.5% into "other" (`step_other`)
- Imputed missing numeric values with median, categorical with mode
- One-hot encoded all nominal predictors (`step_dummy`)
- Removed near-zero-variance columns (`step_nzv`)
- Dropped raw date columns (`date_planted`, `date_harvested`) — or optionally engineered them into numeric features: day-of-year planted, day-of-year harvested, and season length in days

#### 2. Cross-Validation
- 10-fold cross-validation stratified on `yield_mg_ha`

#### 3. Hyperparameter Tuning
- 100-point Latin Hypercube search grid per model
- ANOVA racing (`tune_race_anova`) to eliminate poor candidates early and reduce compute time
- Metrics tracked: RMSE, R², MAE across all folds

#### 4. Model Selection
Six selection strategies were evaluated per model:

| Strategy | Description |
|---|---|
| `best_rmse` | Lowest mean RMSE |
| `best_rmse_pct_loss` | Simpler model within 1% RMSE loss |
| `best_rmse_one_std_err` | Simpler model within 1 SE of best RMSE |
| `best_r2` | Highest mean R² ← **final selection** |
| `best_r2_pct_loss` | Simpler model within 1% R² loss |
| `best_r2_one_std_err` | Simpler model within 1 SE of best R² |

Final models were selected using the **`best_r2`** strategy.

#### 5. Final Fit & Evaluation
- 80/20 stratified split of training data for final validation
- Models fit on the 80% training portion via `last_fit()`
- RMSE, R², and MAE reported for both training and validation sets
- Final models refit on 100% of training data to generate held-out test predictions

---

## Outputs (per model run)

| File | Description |
|---|---|
| `00_run_summary.txt` | Timestamp, data size, final hyperparameters, metrics |
| `01_yield_distribution.png` | Target variable histogram |
| `02_missing_data.png / .csv` | Missing data bar chart and table |
| `04_xgb/lgbm_hyperparam_grid.csv` | Full 100-point search grid |
| `04_xgb_grid_viz.png` | Latin hypercube grid visualization |
| `05_xgb/lgbm_race_plot.png` | ANOVA racing elimination plot |
| `05_xgb/lgbm_tune_metrics_all.csv` | All CV tuning results |
| `06_xgb/lgbm_best_hyperparams_all_strategies.csv` | All 6 selection strategies compared |
| `07_final_chosen_hyperparams.csv` | Final hyperparameters used per model |
| `08_train_val_distribution.png` | Density plot of 80/20 split |
| `09_validation_metrics.csv` | RMSE, R², MAE on validation set |
| `10_training_metrics.csv` | RMSE, R², MAE on training set |
| `11_metrics_summary_train_vs_val.csv` | Combined metrics comparison |
| `11a_rmse_comparison.png` | RMSE bar chart — train vs validation |
| `11b_r2_comparison.png` | R² bar chart — train vs validation |
| `12_xgb/lgbm_pred_vs_obs.png` | Predicted vs observed scatter plot |
| `12_xgb/lgbm_residuals.png` | Residuals vs fitted plot |
| `12_xgb/lgbm_val_predictions.csv` | Raw validation predictions |
| `13_xgb/lgbm_vip.png` | Top 20 variable importance plot |
| `13_xgb/lgbm_variable_importance.csv` | Full ranked variable importance |
| `14_test_predictions.csv` | Final held-out test predictions |

---