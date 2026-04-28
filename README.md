------------------------------------------------------------------------

------------------------------------------------------------------------

# 2026dsa_finalproject_group8

## Project Overview

This project prepares clean, reproducible training and testing datasets for the CRSS 8030 final course project and uses them for downstream machine learning prediction of corn yield.

The workflow was divided into two main components:

-   **Data engineering** — cleaning, merging, feature engineering, weather integration, and export of model-ready datasets
-   **Modeling** — training and tuning XGBoost and LightGBM models across multiple feature sets to predict `yield_mg_ha`

------------------------------------------------------------------------

## Main Files

### Data Engineering

-   `code/CRSS8030_Final_project_group8_data_engineering.qmd` — main data engineering workflow
-   `code/CRSS8030_Final_project_group8_data_engineering.html` — rendered report

### Modeling

#### Original / general modeling scripts

-   `code/yield_xgb_lgbm.R` — original main modeling workflow
-   `code/yield_prediction_ml.qmd` — modeling notebook / supporting workflow
-   `code/yield_prediction_ml_script.R` — extracted script version

#### Revised cluster-ready modeling scripts

-   `code/hybrid_yield_xgb.R` — parent-only XGBoost model
-   `code/hybrid_yield_lgbm.R` — parent-only LightGBM model
-   `code/hybridplus_yield_xgb.R` — hybrid-plus XGBoost model
-   `code/hybridplus_yield_lgbm.R` — hybrid-plus LightGBM model

------------------------------------------------------------------------

## Folder Structure

-   `data/` — raw training and testing input files
-   `output/final_model_files/` — final model-ready datasets
-   `output/weather_files/` — weather support and intermediate weather files
-   `output/metadata_files/` — cleaned trait, soil, and metadata files
-   `output/summary_files/` — summary and documentation files

### Additional modeling output folders

-   `output/monthly_gs_parent_xgb/` — parent-only XGBoost outputs
-   `output/monthly_gs_parent_lgbm/` — parent-only LightGBM outputs
-   `output/monthly_gs_hybridplus_xgb/` — hybrid-plus XGBoost outputs
-   `output/monthly_gs_hybridplus_lgbm/` — hybrid-plus LightGBM outputs

------------------------------------------------------------------------

## Data Engineering Summary

The data engineering workflow:

-   cleaned and standardized site names and previous crop labels
-   merged training trait, metadata, and soil files
-   merged testing submission, metadata, and soil files
-   aggregated training trait data to the **year × site × hybrid** level to match the submission structure
-   added external weather features
-   exported multiple model-ready datasets for downstream machine learning

------------------------------------------------------------------------

## Modeling Notes

-   **Response variable:** `yield_mg_ha`
-   Training data contains observed yield
-   Testing data retain matching predictors, but do not contain observed yield
-   Main model-ready datasets include:
    -   **Base/Core**
    -   **Yearly weather**
    -   **Monthly weather**
    -   **Monthly GS** (monthly growing-season subset)

------------------------------------------------------------------------

## Feature Set Meaning

-   **Base/Core** — cleaned internal predictors only (year, site, hybrid, previous_crop, coordinates, and soil variables)
-   **Yearly weather** — base predictors plus yearly weather summaries
-   **Monthly weather** — base predictors plus monthly weather summaries across all months
-   **Monthly GS** — revised monthly weather feature set restricted to biologically relevant corn growing-season months

### Additional revised hybrid feature sets

#### Parent-only

The parent-only models replaced raw hybrid identity with hybrid-derived parent features:

-   `parent_a`
-   `parent_b`
-   `hybrid_has_slash`

These were used to reduce prediction collapse at the environment level and improve within-site hybrid differentiation.

#### Hybrid-plus

The hybrid-plus models extended the parent-only approach by retaining exact hybrid identity and adding:

-   `hybrid`
-   `parent_a`
-   `parent_b`
-   `parent_pair`
-   `hybrid_has_slash`

This was tested to improve hybrid-specific prediction while keeping biologically interpretable parent structure.

------------------------------------------------------------------------

## Partner Handoff

Use `output/final_model_files/` for downstream modeling inputs. Supporting files are organized into weather, metadata, and summary subfolders.

For the revised Monthly GS modeling runs, use the cluster-generated output folders listed above.

------------------------------------------------------------------------

## Large Files

Large generated monthly-weather CSVs were excluded from GitHub due to file size limits and should be shared separately if needed.

------------------------------------------------------------------------

## Model Training Workflow

### 1. Preprocessing (tidymodels recipe)

The revised cluster-ready models used a tidymodels recipe that:

-   flagged unseen factor levels with `step_novel()`
-   lumped rare `previous_crop` levels with `step_other()`
-   imputed missing numeric values with median
-   imputed missing categorical values with mode
-   one-hot encoded nominal predictors with `step_dummy()`
-   removed near-zero-variance predictors with `step_nzv()`

### 2. Cross-Validation

Revised hybrid-plus tuning runs used:

-   **10-fold cross-validation**
-   stratified on `yield_mg_ha`

### 3. Hyperparameter Tuning

Revised hybrid-plus tuning runs used:

-   **50-point Latin Hypercube search grid** per model
-   metrics tracked: **RMSE, R², and MAE**
-   separate model runs for:
    -   XGBoost
    -   LightGBM

### 4. Final Validation

-   final **80/20 stratified split** of training data
-   fitted with `last_fit()`
-   validation metrics reported for both models
-   final models refit on 100% of training data to generate held-out test predictions

------------------------------------------------------------------------

## Final Model Comparison

| Feature Set | Model | Validation RMSE | Validation R² | Notes |
|---------------|--------------:|--------------:|--------------:|---------------|
| Old Monthly GS | XGBoost | 1.817750 | 0.628279 | Original Monthly GS model |
| Old Monthly GS | LightGBM | 1.817811 | 0.628255 | Very similar to XGBoost |
| Parent-only | XGBoost | 1.788056 | 0.640334 | Added parent-derived hybrid features |
| Parent-only | LightGBM | 1.787938 | 0.640382 | Slightly better than XGBoost |
| Hybrid-plus | XGBoost | 1.787927 | 0.640382 | Added raw hybrid + parent-pair features |
| **Hybrid-plus** | **LightGBM** | **1.787780** | **0.640443** | **Best overall model** |

### Summary of results

-   Both **parent-only** and **hybrid-plus** feature engineering improved model performance relative to the original Monthly GS models.
-   Validation RMSE decreased from approximately **1.818** to **1.788**.
-   Validation R² increased from approximately **0.628** to **0.640**.
-   The best-performing final model was **Hybrid-plus LightGBM**, although differences among the best models were very small.

------------------------------------------------------------------------

## Interpretation of Prediction Behavior

A major issue in earlier modeling was that predictions tended to collapse at the **environment/site level**, giving many hybrids within a site the same or nearly the same predicted value.

The parent-only and hybrid-plus feature engineering steps improved this behavior by allowing the models to distinguish small groups of hybrids within sites based on shared parent structure. However, even in the revised runs, environment remained a strong driver of prediction, which is expected for yield.

Thus, the revised models improved:

-   overall validation performance
-   biological interpretability of within-site prediction groups
-   hybrid differentiation relative to the original environment-heavy models

------------------------------------------------------------------------

## Example Output Naming Structure

| File | Description |
|------------------------------------|------------------------------------|
| `01_yield_distribution.png` | Target variable histogram |
| `02_missing_data.png / .csv` | Missing data plot and table |
| `03_predictor_count.txt` | Number of predictors after preprocessing |
| `04_xgb/lgbm_hyperparam_grid.csv` | Full tuning grid |
| `05_xgb/lgbm_tune_metrics_all.csv` | All CV tuning results |
| `07_xgb/lgbm_final_best.csv` | Final chosen hyperparameters |
| `08_train_val_distribution.png` | Density plot of 80/20 split |
| `09_xgb/lgbm_validation_metrics.csv` | RMSE, R², MAE on validation set |
| `12_xgb/lgbm_pred_vs_obs.png` | Predicted vs observed scatter plot |
| `12_xgb/lgbm_val_predictions.csv` | Raw validation predictions |
| `13_xgb/lgbm_vip.png` | Top variable importance plot |
| `14_test_predictions.csv` | Final held-out test predictions |
| `14_site_prediction_summary.csv` | Number of unique predictions per site |
| `xgb/lgbm_*_fit.rds` | Saved fitted model object |

------------------------------------------------------------------------

## Current Recommendation

Based on the final comparisons in this repository, the recommended final model is:

### **Hybrid-plus LightGBM**

-   Best validation RMSE
-   Best validation R²
-   Same general within-site differentiation pattern as the other revised models
-   Most complete final feature set tested
