# 2026dsa_finalproject_group8

## Project Overview

This project was completed for the CRSS 8030 final course project and focuses on predicting corn yield using reproducible data engineering, machine learning, and interactive visualization.

The workflow included three connected components:

-   **Data engineering** — cleaning, merging, feature engineering, weather integration, and export of model-ready datasets
-   **Modeling** — training and tuning XGBoost and LightGBM models across multiple feature sets to predict `yield_mg_ha`
-   **Shiny app development** — building an interactive app to communicate the data, model performance, and 2024 prediction results

The overall goal was to prepare clean training and testing datasets, evaluate multiple machine learning feature sets, and generate predictions for the 2024 test environments.

------------------------------------------------------------------------

## Main Files

### Data Engineering

-   `code/CRSS8030_Final_project_group8_data_engineering.qmd` — main data engineering workflow
-   `code/CRSS8030_Final_project_group8_data_engineering.html` — rendered report

### Modeling

#### Original / general modeling workflows

-   `code/yield_xgb_lgbm.R` — original main modeling workflow
-   `code/yield_prediction_ml.qmd` — modeling notebook / supporting workflow
-   `code/yield_prediction_ml_script.R` — extracted script version

#### Revised cluster-ready modeling scripts

-   `code/hybrid_yield_xgb.R` — parent-only XGBoost model
-   `code/hybrid_yield_lgbm.R` — parent-only LightGBM model
-   `code/hybridplus_yield_xgb.R` — hybrid-plus XGBoost model
-   `code/hybridplus_yield_lgbm.R` — hybrid-plus LightGBM model

### Shiny App

-   `app.R` — main deployed Shiny application
-   `code/app.R` — development version of the app
-   Live app: `https://harimarasini.shinyapps.io/CornVision/`

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

-   Cleaned and standardized site names and previous crop labels
-   Merged training trait, metadata, and soil files
-   Merged testing submission, metadata, and soil files
-   Aggregated training trait data to the **year × site × hybrid** level to match the final submission structure
-   Added external weather features
-   Exported multiple model-ready datasets for downstream machine learning

------------------------------------------------------------------------

## Modeling Notes

-   **Response variable:** `yield_mg_ha`
-   Training data contain observed yield
-   Testing data retain matching predictors but do not contain observed yield
-   Main model-ready datasets included:
    -   **Base/Core**
    -   **Yearly weather**
    -   **Monthly weather**
    -   **Monthly GS** (monthly growing-season subset)

------------------------------------------------------------------------

## Feature Set Meaning

### Base/Core

Cleaned internal predictors only, including: - `year` - `site` - `hybrid` - `previous_crop` - coordinates - soil variables

### Yearly Weather

Base predictors plus yearly weather summaries.

### Monthly Weather

Base predictors plus monthly weather summaries across all months.

### Monthly GS

A revised monthly weather feature set restricted to biologically relevant corn growing-season months.

------------------------------------------------------------------------

## Revised Hybrid Feature Sets

### Parent-only

The parent-only models replaced raw hybrid identity with hybrid-derived parent structure:

-   `parent_a`
-   `parent_b`
-   `hybrid_has_slash`

This feature set was designed to reduce prediction collapse at the environment level while improving within-site hybrid differentiation in a relatively simple and interpretable way.

### Hybrid-plus

The hybrid-plus models extended the parent-only approach by retaining exact hybrid identity and adding:

-   `hybrid`
-   `parent_a`
-   `parent_b`
-   `parent_pair`
-   `hybrid_has_slash`

This feature set was tested to improve hybrid-specific prediction further while preserving interpretable parent structure. However, the gain over parent-only was very small.

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

### 2. Cross-Validation and Tuning Strategy

Two rounds of revised modeling were conducted.

#### Parent-only models

-   **5-fold cross-validation**
-   **30-point Latin Hypercube search grid**
-   separate runs for:
    -   XGBoost
    -   LightGBM

#### Hybrid-plus models

-   **10-fold cross-validation**
-   **50-point Latin Hypercube search grid**
-   separate runs for:
    -   XGBoost
    -   LightGBM

Metrics tracked for all runs: - **RMSE** - **R²** - **MAE**

### 3. Final Validation

For each revised model:

-   a final **80/20 stratified split** of the training data was used
-   models were fitted with `last_fit()`
-   validation metrics were recorded
-   final models were refit on 100% of the training data to generate held-out test predictions

------------------------------------------------------------------------

## Final Model Comparison

| Feature Set | Model | CV / Grid | Validation RMSE | Validation R² | Notes |
|------------|-----------:|-----------:|-----------:|-----------:|------------|
| Old Monthly GS | XGBoost | original workflow | 1.817750 | 0.628279 | Original Monthly GS model |
| Old Monthly GS | LightGBM | original workflow | 1.817811 | 0.628255 | Very similar to XGBoost |
| Parent-only | XGBoost | 5-fold / 30-grid | 1.788056 | 0.640334 | Added parent-derived hybrid features |
| **Parent-only** | **LightGBM** | **5-fold / 30-grid** | **1.787938** | **0.640382** | **Selected final model** |
| Hybrid-plus | XGBoost | 10-fold / 50-grid | 1.787927 | 0.640382 | Added raw hybrid + parent-pair features |
| Hybrid-plus | LightGBM | 10-fold / 50-grid | 1.787780 | 0.640443 | Slight numerical improvement, but minimal practical gain |

------------------------------------------------------------------------

## Summary of Results

-   Both **parent-only** and **hybrid-plus** feature engineering improved model performance relative to the original Monthly GS models.
-   The **parent-only** models were tuned using **5-fold cross-validation** and a **30-point grid**.
-   The **hybrid-plus** models were tuned using **10-fold cross-validation** and a **50-point grid**.
-   Validation RMSE decreased from approximately **1.818** to **1.788**.
-   Validation R² increased from approximately **0.628** to **0.640**.
-   Although the hybrid-plus models showed a very small numerical improvement, the gain was minimal relative to the added complexity.
-   The final selected model was **parent-only LightGBM**, which provided strong performance with a simpler and more interpretable feature set.

------------------------------------------------------------------------

## Interpretation of Prediction Behavior

A major issue in earlier modeling was that predictions tended to collapse at the **environment/site** level, giving many hybrids within a site the same or nearly the same predicted value.

The parent-only and hybrid-plus feature engineering steps improved this behavior by allowing the models to distinguish small groups of hybrids within sites based on shared parent structure. However, environment remained a strong driver of prediction, which is expected for yield.

Thus, the revised models improved:

-   overall validation performance
-   biological interpretability of within-site prediction groups
-   hybrid differentiation relative to the original environment-heavy models

------------------------------------------------------------------------

## Shiny App Summary

To communicate the final project results in an interactive format, we developed **CornVision**, an R Shiny application that visualizes the data, modeling workflow, and 2024 prediction outputs.

The app includes:

-   Project overview and key data statistics
-   Yield exploratory data analysis
-   Weather and soil visualizations
-   Site-level exploration
-   Variable importance plots
-   Model performance summaries
-   Interactive 2024 prediction displays

The app was built to make the machine learning workflow more interpretable and accessible by allowing users to explore both the input data and the final prediction outputs through an interactive interface.

### Live App

`https://harimarasini.shinyapps.io/CornVision/`

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

## Partner Handoff

Use `output/final_model_files/` for downstream modeling inputs. Supporting files are organized into weather, metadata, and summary subfolders.

For the revised Monthly GS modeling runs, use the cluster-generated output folders listed above.

------------------------------------------------------------------------

## Large Files

Large generated monthly-weather CSVs were excluded from GitHub due to file size limits and should be shared separately if needed.

------------------------------------------------------------------------

## Final Model Recommendation

Based on the final comparisons in this repository, the selected final model is:

### **Parent-only LightGBM**

Why this model was chosen:

-   Strong validation performance
-   Lower complexity than hybrid-plus
-   Improved within-site differentiation compared with the original Monthly GS model
-   Simpler and more interpretable feature set
-   Minimal practical loss relative to hybrid-plus despite nearly identical performance
