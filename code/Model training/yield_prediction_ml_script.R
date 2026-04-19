library(tidyverse)
library(tidymodels)
library(xgboost)
library(bonsai)        # tidymodels engine for LightGBM
library(lightgbm)
library(finetune)      # racing (ANOVA) tuning
library(vip)           # variable importance
library(ggplot2)
library(patchwork)

tidymodels_prefer()
set.seed(2026)

train_path <- "/Users/hm/Library/CloudStorage/OneDrive-UniversityofGeorgia/class_notes/CRSS_8030/2026dsa_finalproject_group8/output/final_model_files/training_model_base_final.csv"
test_path  <- "/Users/hm/Library/CloudStorage/OneDrive-UniversityofGeorgia/class_notes/CRSS_8030/2026dsa_finalproject_group8/output/final_model_files/testing_model_base_final.csv"

train_raw <- read_csv(train_path)
test_raw  <- read_csv(test_path)

# Quick look
glimpse(train_raw)
cat("\nTraining rows :", nrow(train_raw),
    "\nTesting rows  :", nrow(test_raw))

# Target distribution
p1 <- ggplot(train_raw, aes(x = yield_mg_ha)) +
  geom_histogram(bins = 60, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  labs(title = "Yield distribution (training)", x = "yield_mg_ha", y = "count") +
  theme_bw()

# Missing data summary
miss_tbl <- train_raw |>
  summarise(across(everything(), ~ mean(is.na(.)))) |>
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") |>
  filter(pct_missing > 0) |>
  arrange(desc(pct_missing))

p2 <- ggplot(miss_tbl, aes(x = reorder(variable, pct_missing), y = pct_missing)) +
  geom_col(fill = "#d7191c") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Missing data (training)", x = NULL, y = "% missing") +
  theme_bw()

p1 + p2

base_recipe <- recipe(yield_mg_ha ~ ., data = train_raw) |>
  # Handle unseen factor levels
  step_novel(all_nominal_predictors()) |>
  # Lump rare hybrids / sites / previous crops (< 0.5 % of training data)
  step_other(all_nominal_predictors(), threshold = 0.005) |>
  # Impute
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  # Encode categoricals
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  # Remove near-zero variance columns created by lumping
  step_nzv(all_predictors())

# Verify recipe on small bake
prep(base_recipe, training = train_raw) |>
  bake(new_data = head(train_raw, 5)) |>
  glimpse()

cv_folds <- vfold_cv(
  train_raw,
  v       = 5,
  repeats = 2,
  strata  = yield_mg_ha,
  breaks  = 4   # quartile bins for stratification
)

cv_folds

xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune(),
  learn_rate     = tune()
) |>
  set_engine("xgboost",
             nthread         = parallel::detectCores() - 1,
             counts          = FALSE,   # mtry is a proportion
             early_stopping_rounds = 50
  ) |>
  set_mode("regression")

lgbm_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune(),
  learn_rate     = tune()
) |>
  set_engine("lightgbm",
             num_threads     = parallel::detectCores() - 1,
             counts          = FALSE
  ) |>
  set_mode("regression")

xgb_wf  <- workflow() |> add_recipe(base_recipe) |> add_model(xgb_spec)
lgbm_wf <- workflow() |> add_recipe(base_recipe) |> add_model(lgbm_spec)

# XGBoost grid
xgb_grid <- grid_latin_hypercube(
  trees(          range = c(300L,  2000L)),
  tree_depth(     range = c(3L,    10L)),
  min_n(          range = c(5L,    40L)),
  loss_reduction( range = c(-5,    2)),   # log10 scale internally
  sample_size = sample_prop(range = c(0.5, 1.0)),
  mtry        = mtry(range = c(0.3,  1.0)),   # proportion (counts=FALSE)
  learn_rate( range = c(-3,   -1)),           # log10 scale
  size = 40
)

# LightGBM grid
lgbm_grid <- grid_latin_hypercube(
  trees(          range = c(300L,  2000L)),
  tree_depth(     range = c(3L,    10L)),
  min_n(          range = c(5L,    40L)),
  loss_reduction( range = c(-5,    2)),
  sample_size = sample_prop(range = c(0.5, 1.0)),
  mtry        = mtry(range = c(0.3,  1.0)),
  learn_rate( range = c(-3,   -1)),
  size = 40
)

library(doParallel)
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 1))
registerDoParallel(cl)
cat("Workers registered:", getDoParWorkers(), "\n")

xgb_race <- tune_race_anova(
  xgb_wf,
  resamples  = cv_folds,
  grid       = xgb_grid,
  metrics    = metric_set(rmse, rsq, mae),
  control    = control_race(
    verbose        = TRUE,
    save_pred      = FALSE,
    save_workflow  = FALSE,
    burn_in        = 5,          # evaluate all candidates for 5 folds before racing
    alpha          = 0.05
  )
)

autoplot(xgb_race)

lgbm_race <- tune_race_anova(
  lgbm_wf,
  resamples  = cv_folds,
  grid       = lgbm_grid,
  metrics    = metric_set(rmse, rsq, mae),
  control    = control_race(
    verbose        = TRUE,
    save_pred      = FALSE,
    save_workflow  = FALSE,
    burn_in        = 5,
    alpha          = 0.05
  )
)

autoplot(lgbm_race)

# Best XGBoost parameters
xgb_best <- select_best(xgb_race, metric = "rmse")
xgb_best

# Best LightGBM parameters
lgbm_best <- select_best(lgbm_race, metric = "rmse")
lgbm_best

# CV performance comparison
xgb_cv_metrics  <- collect_metrics(xgb_race) |>
  filter(.metric == "rmse") |>
  slice_min(mean, n = 1) |>
  mutate(model = "XGBoost")

lgbm_cv_metrics <- collect_metrics(lgbm_race) |>
  filter(.metric == "rmse") |>
  slice_min(mean, n = 1) |>
  mutate(model = "LightGBM")

bind_rows(xgb_cv_metrics, lgbm_cv_metrics) |>
  select(model, mean, std_err, n) |>
  rename(CV_RMSE = mean) |>
  knitr::kable(digits = 4, caption = "Best CV RMSE per model")

# Finalise workflows
xgb_final_wf  <- finalize_workflow(xgb_wf,  xgb_best)
lgbm_final_wf <- finalize_workflow(lgbm_wf, lgbm_best)

# Fit on full training set
xgb_fit  <- fit(xgb_final_wf,  data = train_raw)
lgbm_fit <- fit(lgbm_final_wf, data = train_raw)

cat("Models fitted on full training data ✓\n")

xgb_preds  <- predict(xgb_fit,  new_data = test_raw) |>
  rename(xgb_pred  = .pred)

lgbm_preds <- predict(lgbm_fit, new_data = test_raw) |>
  rename(lgbm_pred = .pred)

# Note: test set has no yield_mg_ha column — we assess on training CV only.
# If a held-out truth column is available in test data, bind and evaluate below.

if ("yield_mg_ha" %in% names(test_raw)) {
  results_tbl <- test_raw |>
    select(yield_mg_ha) |>
    bind_cols(xgb_preds, lgbm_preds)

  eval_metrics <- metric_set(rmse, rsq, mae)

  xgb_test_met <- results_tbl |>
    eval_metrics(truth = yield_mg_ha, estimate = xgb_pred) |>
    mutate(model = "XGBoost")

  lgbm_test_met <- results_tbl |>
    eval_metrics(truth = yield_mg_ha, estimate = lgbm_pred) |>
    mutate(model = "LightGBM")

  test_perf <- bind_rows(xgb_test_met, lgbm_test_met) |>
    select(model, .metric, .estimate) |>
    pivot_wider(names_from = .metric, values_from = .estimate)

  print(knitr::kable(test_perf, digits = 4,
        caption = "Test set performance"))

  # Observed vs Predicted plots
  p_xgb <- ggplot(results_tbl, aes(x = yield_mg_ha, y = xgb_pred)) +
    geom_point(alpha = 0.3, size = 0.8, color = "#2c7bb6") +
    geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 0.8) +
    labs(title = "XGBoost — Observed vs Predicted",
         x = "Observed yield (Mg/ha)", y = "Predicted yield (Mg/ha)") +
    theme_bw()

  p_lgbm <- ggplot(results_tbl, aes(x = yield_mg_ha, y = lgbm_pred)) +
    geom_point(alpha = 0.3, size = 0.8, color = "#1a9641") +
    geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 0.8) +
    labs(title = "LightGBM — Observed vs Predicted",
         x = "Observed yield (Mg/ha)", y = "Predicted yield (Mg/ha)") +
    theme_bw()

  print(p_xgb + p_lgbm)

} else {
  message("Test set does not contain yield_mg_ha; skipping test-set evaluation.")
  cat("XGBoost predictions (first 6):\n")
  print(head(xgb_preds))
  cat("\nLightGBM predictions (first 6):\n")
  print(head(lgbm_preds))
}

# XGBoost VIP
p_vip_xgb <- xgb_fit |>
  extract_fit_parsnip() |>
  vip(num_features = 20, geom = "col",
      aesthetics = list(fill = "#2c7bb6", color = "white")) +
  labs(title = "XGBoost — Top 20 Features") +
  theme_bw()

# LightGBM VIP
p_vip_lgbm <- lgbm_fit |>
  extract_fit_parsnip() |>
  vip(num_features = 20, geom = "col",
      aesthetics = list(fill = "#1a9641", color = "white")) +
  labs(title = "LightGBM — Top 20 Features") +
  theme_bw()

p_vip_xgb / p_vip_lgbm

out_dir <- dirname(test_path)

# Save predictions
predictions_out <- bind_cols(test_raw, xgb_preds, lgbm_preds)
write_csv(predictions_out,
          file.path(out_dir, "test_predictions.csv"))

# Save fitted models
saveRDS(xgb_fit,  file.path(out_dir, "xgb_final_fit.rds"))
saveRDS(lgbm_fit, file.path(out_dir, "lgbm_final_fit.rds"))

cat("Saved predictions and models to:", out_dir, "\n")

stopCluster(cl)
sessionInfo()

knitr::purl("yield_prediction_ml.qmd", output = "yield_prediction_ml_script.R", documentation = 0)
