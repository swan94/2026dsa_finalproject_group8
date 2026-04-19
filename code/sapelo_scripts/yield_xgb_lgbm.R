# ============================================================
#  Corn Yield Prediction — XGBoost & LightGBM
#  CRSS 8030 | Final Project Group 8 | 2026
# ============================================================


# ── Step 1: Load Libraries ───────────────────────────────────
library(tidymodels)   # Core ML framework
library(finetune)     # tune_race_anova()
library(vip)          # Variable importance plots
library(xgboost)      # XGBoost engine
library(bonsai)       # LightGBM engine for tidymodels
library(lightgbm)     # LightGBM backend
library(tidyverse)    # Data wrangling & visualization
library(doParallel)   # Parallel processing


# ── Step 2: Load Data ────────────────────────────────────────
train_path <- "/home/hm64666/final_project/data/training/training_model_base_final.csv"
test_path  <- "/home/hm64666/final_project/data/testing/testing_model_base_final.csv"

train_raw <- read_csv(train_path)
test_raw  <- read_csv(test_path)

glimpse(train_raw)
glimpse(test_raw)


# ── Step 3: Exploratory Plots ────────────────────────────────

# Target variable distribution
ggplot(train_raw, aes(x = yield_mg_ha)) +
  geom_histogram(bins = 60, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  labs(title = "Distribution of Corn Yield (Training Set)",
       x = "Yield (Mg/ha)", y = "Count") +
  theme_bw()

# Missing data summary
train_raw %>%
  summarise(across(everything(), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>%
  filter(pct_missing > 0) %>%
  arrange(desc(pct_missing)) %>%
  ggplot(aes(x = reorder(variable, pct_missing), y = pct_missing)) +
  geom_col(fill = "#d7191c") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Missing Data — Training Set",
       x = NULL, y = "% Missing") +
  theme_bw()


# ── Step 4: Pre-processing with Recipe ──────────────────────
#
#  Steps chosen:
#    step_novel        — flags new factor levels seen only in test set
#    step_other        — lumps rare hybrids/sites (< 0.5%) into "other"
#    step_impute_median — fills missing numeric values (soil pH, OM, K, P)
#    step_impute_mode  — fills missing categorical values
#    step_dummy        — one-hot encode all nominal predictors
#    step_nzv          — removes near-zero-variance columns after encoding
#
#  No normalization needed — tree models are scale-invariant.

yield_recipe <- recipe(yield_mg_ha ~ ., data = train_raw) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors())

yield_recipe

# Prep and inspect
yield_prep <- yield_recipe %>% prep()
yield_prep

# Check number of predictors after preprocessing (needed for mtry range)
n_preds <- ncol(bake(yield_prep, new_data = NULL, all_predictors()))
cat("Predictors after preprocessing:", n_preds, "\n")


# ── Step 5: Cross-Validation Setup ──────────────────────────
#
#  10-fold CV, stratified on yield_mg_ha (matches instructor's v = 10 style).
#  Stratification on a numeric target uses quartile bins automatically.

set.seed(235)
resampling_foldcv <- vfold_cv(
  train_raw,
  v      = 10,
  strata = yield_mg_ha
)

resampling_foldcv
resampling_foldcv$splits[[1]]


# ── Step 6: Model Specifications ────────────────────────────

# --- XGBoost ---
xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune(),
  learn_rate     = tune()
) %>%
  set_engine("xgboost",
             nthread               = parallel::detectCores() - 1,
             early_stopping_rounds = 50) %>%
  set_mode("regression")

xgb_spec

# --- LightGBM ---
lgbm_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune(),
  learn_rate     = tune()
) %>%
  set_engine("lightgbm",
             num_threads = parallel::detectCores() - 1) %>%
  set_mode("regression")

lgbm_spec


# ── Step 7: Hyperparameter Grids (Latin Hypercube) ───────────
#
#  mtry range must be integers (column counts), so we derive the
#  upper bound from n_preds computed after prep().
#  grid_space_filling() replaces the deprecated grid_latin_hypercube().

xgb_grid <- grid_space_filling(
  trees(          range = c(300L,  2000L)),
  tree_depth(     range = c(3L,    10L)),
  min_n(          range = c(5L,    40L)),
  loss_reduction( range = c(-5,    2)),
  sample_size   = sample_prop(range = c(0.5, 1.0)),
  mtry(           range = c(3L, as.integer(ceiling(n_preds * 0.8)))),
  learn_rate(     range = c(-3,   -1)),
  size = 100,
  type = "latin_hypercube"
)

lgbm_grid <- grid_space_filling(
  trees(          range = c(300L,  2000L)),
  tree_depth(     range = c(3L,    10L)),
  min_n(          range = c(5L,    40L)),
  loss_reduction( range = c(-5,    2)),
  sample_size   = sample_prop(range = c(0.5, 1.0)),
  mtry(           range = c(3L, as.integer(ceiling(n_preds * 0.8)))),
  learn_rate(     range = c(-3,   -1)),
  size = 100,
  type = "latin_hypercube"
)

# Visualize the grid (like instructor's ggplot of grid)
ggplot(data = xgb_grid,
       aes(x = tree_depth, y = min_n)) +
  geom_point(aes(color = factor(round(learn_rate, 3)),
                 size  = trees),
             alpha = 0.5, show.legend = FALSE) +
  labs(title = "XGBoost Hyperparameter Grid (Latin Hypercube)",
       x = "Tree Depth", y = "Min N") +
  theme_bw()


# ── Step 8: Tune — ANOVA Racing ──────────────────────────────

set.seed(76544)
registerDoParallel(cores = parallel::detectCores() - 1)

# --- XGBoost tuning ---
xgb_res <- tune_race_anova(
  object       = xgb_spec,
  preprocessor = yield_recipe,
  resamples    = resampling_foldcv,
  grid         = xgb_grid,
  metrics      = metric_set(rmse, rsq, mae),
  control      = control_race(save_pred = TRUE,
                              verbose   = TRUE)
)

stopImplicitCluster()

# Race plot — XGBoost
plot_race(xgb_res)

# Re-register for LightGBM
registerDoParallel(cores = parallel::detectCores() - 1)

# --- LightGBM tuning ---
lgbm_res <- tune_race_anova(
  object       = lgbm_spec,
  preprocessor = yield_recipe,
  resamples    = resampling_foldcv,
  grid         = lgbm_grid,
  metrics      = metric_set(rmse, rsq, mae),
  control      = control_race(save_pred = TRUE,
                              verbose   = TRUE)
)

stopImplicitCluster()

# Race plot — LightGBM
plot_race(lgbm_res)


# ── Step 9: Select Best Models ───────────────────────────────
#  Mirrors instructor's six-strategy approach for both models.

# ---- XGBoost ----
xgb_best_rmse <- xgb_res %>%
  select_best(metric = "rmse") %>%
  mutate(source = "xgb_best_rmse")

xgb_best_rmse_pct_loss <- xgb_res %>%
  select_by_pct_loss("min_n", metric = "rmse", limit = 1) %>%
  mutate(source = "xgb_best_rmse_pct_loss")

xgb_best_rmse_one_std_err <- xgb_res %>%
  select_by_one_std_err(metric = "rmse", trees) %>%
  mutate(source = "xgb_best_rmse_one_std_err")

xgb_best_r2 <- xgb_res %>%
  select_best(metric = "rsq") %>%
  mutate(source = "xgb_best_r2")

xgb_best_r2_pct_loss <- xgb_res %>%
  select_by_pct_loss("min_n", metric = "rsq", limit = 1) %>%
  mutate(source = "xgb_best_r2_pct_loss")

xgb_best_r2_one_std_err <- xgb_res %>%
  select_by_one_std_err(metric = "rsq", trees) %>%
  mutate(source = "xgb_best_r2_one_std_err")

# Compare XGBoost candidates
xgb_best_rmse %>%
  bind_rows(xgb_best_rmse_pct_loss,
            xgb_best_rmse_one_std_err,
            xgb_best_r2,
            xgb_best_r2_pct_loss,
            xgb_best_r2_one_std_err) %>%
  dplyr::select(source, everything())

# ---- LightGBM ----
lgbm_best_rmse <- lgbm_res %>%
  select_best(metric = "rmse") %>%
  mutate(source = "lgbm_best_rmse")

lgbm_best_rmse_pct_loss <- lgbm_res %>%
  select_by_pct_loss("min_n", metric = "rmse", limit = 1) %>%
  mutate(source = "lgbm_best_rmse_pct_loss")

lgbm_best_rmse_one_std_err <- lgbm_res %>%
  select_by_one_std_err(metric = "rmse", trees) %>%
  mutate(source = "lgbm_best_rmse_one_std_err")

lgbm_best_r2 <- lgbm_res %>%
  select_best(metric = "rsq") %>%
  mutate(source = "lgbm_best_r2")

lgbm_best_r2_pct_loss <- lgbm_res %>%
  select_by_pct_loss("min_n", metric = "rsq", limit = 1) %>%
  mutate(source = "lgbm_best_r2_pct_loss")

lgbm_best_r2_one_std_err <- lgbm_res %>%
  select_by_one_std_err(metric = "rsq", trees) %>%
  mutate(source = "lgbm_best_r2_one_std_err")

# Compare LightGBM candidates
lgbm_best_rmse %>%
  bind_rows(lgbm_best_rmse_pct_loss,
            lgbm_best_rmse_one_std_err,
            lgbm_best_r2,
            lgbm_best_r2_pct_loss,
            lgbm_best_r2_one_std_err) %>%
  dplyr::select(source, everything())


# ── Step 10: Final Specifications ────────────────────────────
#  Use best_r2 (highest R²) as the final selection criterion.

xgb_final_spec <- boost_tree(
  trees          = xgb_best_r2$trees,
  tree_depth     = xgb_best_r2$tree_depth,
  min_n          = xgb_best_r2$min_n,
  loss_reduction = xgb_best_r2$loss_reduction,
  sample_size    = xgb_best_r2$sample_size,
  mtry           = xgb_best_r2$mtry,
  learn_rate     = xgb_best_r2$learn_rate
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

xgb_final_spec

lgbm_final_spec <- boost_tree(
  trees          = lgbm_best_r2$trees,
  tree_depth     = lgbm_best_r2$tree_depth,
  min_n          = lgbm_best_r2$min_n,
  loss_reduction = lgbm_best_r2$loss_reduction,
  sample_size    = lgbm_best_r2$sample_size,
  mtry           = lgbm_best_r2$mtry,
  learn_rate     = lgbm_best_r2$learn_rate
) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

lgbm_final_spec


# ── Step 11: Final Fit on Full Training Data ─────────────────
#  Note: test_raw has no yield_mg_ha, so we use last_fit() on the
#  training split only for validation, then predict on test_raw separately.

# We create an internal 80/20 split from train_raw for last_fit() validation
set.seed(931735)
yield_split <- initial_split(train_raw, prop = 0.8, strata = yield_mg_ha)
yield_train <- training(yield_split)
yield_test  <- testing(yield_split)

# Distribution check (mirrors instructor's density plot)
ggplot() +
  geom_density(data = yield_train,
               aes(x = yield_mg_ha), color = "red") +
  geom_density(data = yield_test,
               aes(x = yield_mg_ha), color = "blue") +
  labs(title = "Yield Distribution — Train (red) vs Validation (blue)",
       x = "Yield (Mg/ha)") +
  theme_bw()

set.seed(10)

xgb_final_fit <- last_fit(
  xgb_final_spec,
  yield_recipe,
  split = yield_split
)

lgbm_final_fit <- last_fit(
  lgbm_final_spec,
  yield_recipe,
  split = yield_split
)


# ── Step 12: Evaluate on Validation Set ──────────────────────

cat("=== XGBoost — Validation Metrics ===\n")
xgb_final_fit %>% collect_metrics()

cat("\n=== LightGBM — Validation Metrics ===\n")
lgbm_final_fit %>% collect_metrics()

# Collect predictions
xgb_preds_val  <- xgb_final_fit  %>% collect_predictions()
lgbm_preds_val <- lgbm_final_fit %>% collect_predictions()


# ── Step 13: Evaluate on Training Set ────────────────────────

# XGBoost training metrics
xgb_final_spec %>%
  fit(yield_mg_ha ~ .,
      data = bake(yield_prep, yield_train)) %>%
  augment(new_data = bake(yield_prep, yield_train)) %>%
  rmse(yield_mg_ha, .pred) %>%
  bind_rows(
    xgb_final_spec %>%
      fit(yield_mg_ha ~ .,
          data = bake(yield_prep, yield_train)) %>%
      augment(new_data = bake(yield_prep, yield_train)) %>%
      rsq(yield_mg_ha, .pred)
  ) %>%
  mutate(model = "XGBoost — Training")

# LightGBM training metrics
lgbm_final_spec %>%
  fit(yield_mg_ha ~ .,
      data = bake(yield_prep, yield_train)) %>%
  augment(new_data = bake(yield_prep, yield_train)) %>%
  rmse(yield_mg_ha, .pred) %>%
  bind_rows(
    lgbm_final_spec %>%
      fit(yield_mg_ha ~ .,
          data = bake(yield_prep, yield_train)) %>%
      augment(new_data = bake(yield_prep, yield_train)) %>%
      rsq(yield_mg_ha, .pred)
  ) %>%
  mutate(model = "LightGBM — Training")


# ── Step 14: Predicted vs Observed Plots ─────────────────────

# XGBoost
xgb_preds_val %>%
  ggplot(aes(x = yield_mg_ha, y = .pred)) +
  geom_point(alpha = 0.3, color = "#2c7bb6", size = 0.8) +
  geom_abline() +
  geom_smooth(method = "lm") +
  labs(title = "XGBoost — Predicted vs Observed (Validation)",
       x = "Observed Yield (Mg/ha)",
       y = "Predicted Yield (Mg/ha)") +
  theme_bw()

# LightGBM
lgbm_preds_val %>%
  ggplot(aes(x = yield_mg_ha, y = .pred)) +
  geom_point(alpha = 0.3, color = "#1a9641", size = 0.8) +
  geom_abline() +
  geom_smooth(method = "lm") +
  labs(title = "LightGBM — Predicted vs Observed (Validation)",
       x = "Observed Yield (Mg/ha)",
       y = "Predicted Yield (Mg/ha)") +
  theme_bw()


# ── Step 15: Variable Importance ─────────────────────────────

# XGBoost VIP
xgb_final_spec %>%
  fit(yield_mg_ha ~ .,
      data = bake(yield_prep, yield_train)) %>%
  vi() %>%
  mutate(Variable = fct_reorder(Variable, Importance)) %>%
  slice_max(Importance, n = 20) %>%
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#2c7bb6") +
  scale_x_continuous(expand = c(0, 0)) +
  labs(title = "XGBoost — Variable Importance (Top 20)", y = NULL) +
  theme_bw()

# LightGBM VIP
lgbm_final_spec %>%
  fit(yield_mg_ha ~ .,
      data = bake(yield_prep, yield_train)) %>%
  vi() %>%
  mutate(Variable = fct_reorder(Variable, Importance)) %>%
  slice_max(Importance, n = 20) %>%
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#1a9641") +
  scale_x_continuous(expand = c(0, 0)) +
  labs(title = "LightGBM — Variable Importance (Top 20)", y = NULL) +
  theme_bw()


# ── Step 16: Predict on Held-Out Test Set ────────────────────
#  test_raw has no yield column — these predictions are for submission.

# Fit final models on ALL training data
xgb_full_fit <- xgb_final_spec %>%
  fit(yield_mg_ha ~ .,
      data = bake(yield_prep, train_raw))

lgbm_full_fit <- lgbm_final_spec %>%
  fit(yield_mg_ha ~ .,
      data = bake(yield_prep, train_raw))

# Generate predictions
xgb_test_preds  <- predict(xgb_full_fit,
                           new_data = bake(yield_prep, test_raw,
                                           all_predictors())) %>%
  rename(xgb_pred = .pred)

lgbm_test_preds <- predict(lgbm_full_fit,
                           new_data = bake(yield_prep, test_raw,
                                           all_predictors())) %>%
  rename(lgbm_pred = .pred)

# Combine and save
test_predictions <- test_raw %>%
  bind_cols(xgb_test_preds, lgbm_test_preds)

out_dir <- dirname(test_path)
write_csv(test_predictions, file.path(out_dir, "test_predictions.csv"))
cat("Predictions saved to:", file.path(out_dir, "test_predictions.csv"), "\n")

# ── Done ─────────────────────────────────────────────────────
