
# ── Step 1: Load Libraries ───────────────────────────────────
library(tidymodels)   # Core ML framework
library(finetune)     # tune_race_anova()
library(vip)          # Variable importance plots
library(xgboost)      # XGBoost engine
library(bonsai)       # LightGBM engine for tidymodels
library(lightgbm)     # LightGBM backend
library(tidyverse)    # Data wrangling & visualization
library(doParallel)   # Parallel processing


# ── Output Directory & Model Tag ─────────────────────────────
#  Separate folder and prefix from monthly_weather (mw),
#  yearly_weather (yw), core (cm), and base model runs.
#
#  model_tag  → prefix on every filename  (e.g. mg_01_...)
#  out_dir    → dedicated subfolder

model_tag <- "mg"     # mg = monthly growing season
out_dir   <- "/home/hm64666/final_project/outputs/monthly_gs"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Helper: save a ggplot (prefixed filename, consistent size)
save_plot <- function(plot_obj, filename, w = 8, h = 5) {
  full_name <- paste0(model_tag, "_", filename)
  ggsave(
    filename = file.path(out_dir, full_name),
    plot     = plot_obj,
    width    = w,
    height   = h,
    dpi      = 300
  )
  message("Plot saved: ", full_name)
}

# Helper: save a data frame as CSV (prefixed filename)
save_csv <- function(df, filename) {
  full_name <- paste0(model_tag, "_", filename)
  write_csv(df, file.path(out_dir, full_name))
  message("CSV  saved: ", full_name)
}


# ── Step 2: Load Data ────────────────────────────────────────
# FIXED: corrected both paths to the monthly_gs data files
train_path <- "/home/hm64666/final_project/data/training/training_model_monthly_gs_final.csv"
test_path  <- "/home/hm64666/final_project/data/testing/testing_model_monthly_gs_final.csv"

train_raw <- read_csv(train_path)
test_raw  <- read_csv(test_path)

# Save data glimpse
sink(file.path(out_dir, paste0(model_tag, "_data_glimpse.txt")))
cat("=== Training Data (Monthly Growing Season) ===\n"); glimpse(train_raw)
cat("\n=== Test Data (Monthly Growing Season) ===\n");    glimpse(test_raw)
sink()
message("Text saved: ", model_tag, "_data_glimpse.txt")


# ── Step 3: Exploratory Plots ────────────────────────────────

# 3a. Target variable distribution
p_yield_dist <- ggplot(train_raw, aes(x = yield_mg_ha)) +
  geom_histogram(bins = 60, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  labs(title = "Monthly GS — Distribution of Corn Yield (Training Set)",
       x = "Yield (Mg/ha)", y = "Count") +
  theme_bw()
save_plot(p_yield_dist, "01_yield_distribution.png")

# 3b. Missing data summary
missing_df <- train_raw %>%
  summarise(across(everything(), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>%
  filter(pct_missing > 0) %>%
  arrange(desc(pct_missing))

if (nrow(missing_df) > 0) {
  p_missing <- missing_df %>%
    ggplot(aes(x = reorder(variable, pct_missing), y = pct_missing)) +
    geom_col(fill = "#d7191c") +
    coord_flip() +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "Monthly GS — Missing Data (Training Set)", x = NULL, y = "% Missing") +
    theme_bw()
  save_plot(p_missing, "02_missing_data.png", w = 8, h = max(4, nrow(missing_df) * 0.35))
} else {
  message("No missing data found — skipping missing-data plot.")
}

save_csv(missing_df, "02_missing_data_table.csv")


# ── Step 4: Pre-processing with Recipe ──────────────────────
# FIXED: step_rm() added to drop date columns (date_planted, date_harvested)
# if present — these are Date type and crash XGBoost/LightGBM at tuning time.
yield_recipe <- recipe(yield_mg_ha ~ ., data = train_raw) %>%
  step_rm(any_of(c("date_planted", "date_harvested"))) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors())

yield_prep <- yield_recipe %>% prep()

n_preds <- ncol(bake(yield_prep, new_data = NULL, all_predictors()))
cat("Predictors after preprocessing:", n_preds, "\n")

write_lines(
  paste("Monthly GS — Predictors after preprocessing:", n_preds),
  file.path(out_dir, paste0(model_tag, "_03_predictor_count.txt"))
)


# ── Step 5: Cross-Validation Setup ──────────────────────────
set.seed(235)
resampling_foldcv <- vfold_cv(train_raw, v = 10, strata = yield_mg_ha)


# ── Step 6: Model Specifications ────────────────────────────

xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(), min_n = tune(),
  loss_reduction = tune(), sample_size = tune(),
  mtry = tune(), learn_rate = tune()
) %>%
  set_engine("xgboost",
             nthread               = parallel::detectCores() - 1,
             early_stopping_rounds = 50) %>%
  set_mode("regression")

lgbm_spec <- boost_tree(
  trees = tune(), tree_depth = tune(), min_n = tune(),
  loss_reduction = tune(), sample_size = tune(),
  mtry = tune(), learn_rate = tune()
) %>%
  set_engine("lightgbm",
             num_threads = parallel::detectCores() - 1) %>%
  set_mode("regression")


# ── Step 7: Hyperparameter Grids (Latin Hypercube) ───────────
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

save_csv(xgb_grid,  "04_xgb_hyperparam_grid.csv")
save_csv(lgbm_grid, "04_lgbm_hyperparam_grid.csv")

p_xgb_grid <- ggplot(data = xgb_grid, aes(x = tree_depth, y = min_n)) +
  geom_point(aes(color = factor(round(learn_rate, 3)), size = trees),
             alpha = 0.5, show.legend = FALSE) +
  labs(title = "Monthly GS — XGBoost Hyperparameter Grid (Latin Hypercube)",
       x = "Tree Depth", y = "Min N") +
  theme_bw()
save_plot(p_xgb_grid, "04_xgb_grid_viz.png")


# ── Step 8: Tune — ANOVA Racing ──────────────────────────────
set.seed(76544)
registerDoParallel(cores = parallel::detectCores() - 1)

xgb_res <- tune_race_anova(
  object       = xgb_spec,
  preprocessor = yield_recipe,
  resamples    = resampling_foldcv,
  grid         = xgb_grid,
  metrics      = metric_set(rmse, rsq, mae),
  control      = control_race(save_pred = TRUE, verbose = TRUE)
)
stopImplicitCluster()

p_xgb_race <- plot_race(xgb_res)
save_plot(p_xgb_race, "05_xgb_race_plot.png")
save_csv(collect_metrics(xgb_res), "05_xgb_tune_metrics_all.csv")

registerDoParallel(cores = parallel::detectCores() - 1)

lgbm_res <- tune_race_anova(
  object       = lgbm_spec,
  preprocessor = yield_recipe,
  resamples    = resampling_foldcv,
  grid         = lgbm_grid,
  metrics      = metric_set(rmse, rsq, mae),
  control      = control_race(save_pred = TRUE, verbose = TRUE)
)
stopImplicitCluster()

p_lgbm_race <- plot_race(lgbm_res)
save_plot(p_lgbm_race, "05_lgbm_race_plot.png")
save_csv(collect_metrics(lgbm_res), "05_lgbm_tune_metrics_all.csv")


# ── Step 9: Select Best Models ───────────────────────────────

# ---- XGBoost ----
xgb_best_rmse             <- xgb_res %>% select_best(metric = "rmse")                             %>% mutate(source = "xgb_best_rmse")
xgb_best_rmse_pct_loss    <- xgb_res %>% select_by_pct_loss("min_n", metric = "rmse", limit = 1) %>% mutate(source = "xgb_best_rmse_pct_loss")
xgb_best_rmse_one_std_err <- xgb_res %>% select_by_one_std_err(metric = "rmse", trees)            %>% mutate(source = "xgb_best_rmse_one_std_err")
xgb_best_r2               <- xgb_res %>% select_best(metric = "rsq")                              %>% mutate(source = "xgb_best_r2")
xgb_best_r2_pct_loss      <- xgb_res %>% select_by_pct_loss("min_n", metric = "rsq",  limit = 1) %>% mutate(source = "xgb_best_r2_pct_loss")
xgb_best_r2_one_std_err   <- xgb_res %>% select_by_one_std_err(metric = "rsq",  trees)            %>% mutate(source = "xgb_best_r2_one_std_err")

xgb_candidates <- xgb_best_rmse %>%
  bind_rows(xgb_best_rmse_pct_loss, xgb_best_rmse_one_std_err,
            xgb_best_r2, xgb_best_r2_pct_loss, xgb_best_r2_one_std_err) %>%
  dplyr::select(source, everything())
save_csv(xgb_candidates, "06_xgb_best_hyperparams_all_strategies.csv")

# ---- LightGBM ----
lgbm_best_rmse             <- lgbm_res %>% select_best(metric = "rmse")                             %>% mutate(source = "lgbm_best_rmse")
lgbm_best_rmse_pct_loss    <- lgbm_res %>% select_by_pct_loss("min_n", metric = "rmse", limit = 1) %>% mutate(source = "lgbm_best_rmse_pct_loss")
lgbm_best_rmse_one_std_err <- lgbm_res %>% select_by_one_std_err(metric = "rmse", trees)            %>% mutate(source = "lgbm_best_rmse_one_std_err")
lgbm_best_r2               <- lgbm_res %>% select_best(metric = "rsq")                              %>% mutate(source = "lgbm_best_r2")
lgbm_best_r2_pct_loss      <- lgbm_res %>% select_by_pct_loss("min_n", metric = "rsq",  limit = 1) %>% mutate(source = "lgbm_best_r2_pct_loss")
lgbm_best_r2_one_std_err   <- lgbm_res %>% select_by_one_std_err(metric = "rsq",  trees)            %>% mutate(source = "lgbm_best_r2_one_std_err")

lgbm_candidates <- lgbm_best_rmse %>%
  bind_rows(lgbm_best_rmse_pct_loss, lgbm_best_rmse_one_std_err,
            lgbm_best_r2, lgbm_best_r2_pct_loss, lgbm_best_r2_one_std_err) %>%
  dplyr::select(source, everything())
save_csv(lgbm_candidates, "06_lgbm_best_hyperparams_all_strategies.csv")


# ── Step 10: Final Specifications (best_r2 strategy) ─────────

xgb_final_spec <- boost_tree(
  trees          = xgb_best_r2$trees,
  tree_depth     = xgb_best_r2$tree_depth,
  min_n          = xgb_best_r2$min_n,
  loss_reduction = xgb_best_r2$loss_reduction,
  sample_size    = xgb_best_r2$sample_size,
  mtry           = xgb_best_r2$mtry,
  learn_rate     = xgb_best_r2$learn_rate
) %>% set_engine("xgboost") %>% set_mode("regression")

lgbm_final_spec <- boost_tree(
  trees          = lgbm_best_r2$trees,
  tree_depth     = lgbm_best_r2$tree_depth,
  min_n          = lgbm_best_r2$min_n,
  loss_reduction = lgbm_best_r2$loss_reduction,
  sample_size    = lgbm_best_r2$sample_size,
  mtry           = lgbm_best_r2$mtry,
  learn_rate     = lgbm_best_r2$learn_rate
) %>% set_engine("lightgbm") %>% set_mode("regression")

final_hyperparams <- bind_rows(
  xgb_best_r2  %>% mutate(model = "XGBoost",  selection = "best_r2"),
  lgbm_best_r2 %>% mutate(model = "LightGBM", selection = "best_r2")
) %>% dplyr::select(model, selection, everything())
save_csv(final_hyperparams, "07_final_chosen_hyperparams.csv")


# ── Step 11: Train / Validation Split ────────────────────────
set.seed(931735)
yield_split <- initial_split(train_raw, prop = 0.8, strata = yield_mg_ha)
yield_train <- training(yield_split)
yield_test  <- testing(yield_split)

p_split_dist <- ggplot() +
  geom_density(data = yield_train, aes(x = yield_mg_ha, color = "Train (80%)")) +
  geom_density(data = yield_test,  aes(x = yield_mg_ha, color = "Validation (20%)")) +
  scale_color_manual(values = c("Train (80%)" = "red", "Validation (20%)" = "blue")) +
  labs(title = "Monthly GS — Yield Distribution: Train vs Validation Split",
       x = "Yield (Mg/ha)", color = NULL) +
  theme_bw() +
  theme(legend.position = "top")
save_plot(p_split_dist, "08_train_val_distribution.png")


# ── Step 12: Final Fits ───────────────────────────────────────
set.seed(10)
xgb_final_fit  <- last_fit(xgb_final_spec,  yield_recipe, split = yield_split)
lgbm_final_fit <- last_fit(lgbm_final_spec, yield_recipe, split = yield_split)


# ── Step 13: Validation Metrics ──────────────────────────────

xgb_val_metrics  <- xgb_final_fit  %>% collect_metrics() %>% mutate(model = "XGBoost")
lgbm_val_metrics <- lgbm_final_fit %>% collect_metrics() %>% mutate(model = "LightGBM")

val_metrics_combined <- bind_rows(xgb_val_metrics, lgbm_val_metrics) %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

save_csv(val_metrics_combined, "09_validation_metrics.csv")
cat("\n=== Monthly GS — Validation Metrics ===\n")
print(val_metrics_combined)


# ── Step 14: Training-set Metrics ────────────────────────────

compute_train_metrics <- function(spec, prep_obj, train_df, model_name) {
  fit_obj <- spec %>% fit(yield_mg_ha ~ ., data = bake(prep_obj, train_df))
  aug     <- augment(fit_obj, new_data = bake(prep_obj, train_df))
  bind_rows(
    rmse(aug, yield_mg_ha, .pred),
    rsq( aug, yield_mg_ha, .pred),
    mae( aug, yield_mg_ha, .pred)
  ) %>% mutate(model = model_name, set = "Training")
}

xgb_train_metrics  <- compute_train_metrics(xgb_final_spec,  yield_prep, yield_train, "XGBoost")
lgbm_train_metrics <- compute_train_metrics(lgbm_final_spec, yield_prep, yield_train, "LightGBM")

train_metrics_combined <- bind_rows(xgb_train_metrics, lgbm_train_metrics) %>%
  dplyr::select(model, set, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

save_csv(train_metrics_combined, "10_training_metrics.csv")

# Combined train vs validation summary
metrics_summary <- bind_rows(
  train_metrics_combined,
  val_metrics_combined %>% mutate(set = "Validation")
) %>% dplyr::select(model, set, rmse, rsq, mae) %>%
  arrange(model, set)

save_csv(metrics_summary, "11_metrics_summary_train_vs_val.csv")
cat("\n=== Monthly GS — Metrics Summary (Train vs Validation) ===\n")
print(metrics_summary)

# Bar chart: RMSE
p_rmse_bar <- metrics_summary %>%
  ggplot(aes(x = model, y = rmse, fill = set)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_text(aes(label = round(rmse, 4)),
            position = position_dodge(0.6), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = c("Training" = "#2c7bb6", "Validation" = "#d7191c")) +
  labs(title = "Monthly GS — RMSE: Train vs Validation",
       x = NULL, y = "RMSE (Mg/ha)", fill = NULL) +
  theme_bw() + theme(legend.position = "top")
save_plot(p_rmse_bar, "11a_rmse_comparison.png", w = 6, h = 5)

# Bar chart: R²
p_r2_bar <- metrics_summary %>%
  ggplot(aes(x = model, y = rsq, fill = set)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_text(aes(label = round(rsq, 4)),
            position = position_dodge(0.6), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = c("Training" = "#2c7bb6", "Validation" = "#d7191c")) +
  labs(title = "Monthly GS — R²: Train vs Validation",
       x = NULL, y = "R²", fill = NULL) +
  theme_bw() + theme(legend.position = "top")
save_plot(p_r2_bar, "11b_r2_comparison.png", w = 6, h = 5)


# ── Step 15: Predicted vs Observed Plots ─────────────────────

xgb_preds_val  <- xgb_final_fit  %>% collect_predictions()
lgbm_preds_val <- lgbm_final_fit %>% collect_predictions()

p_xgb_pred <- xgb_preds_val %>%
  ggplot(aes(x = yield_mg_ha, y = .pred)) +
  geom_point(alpha = 0.3, color = "#2c7bb6", size = 0.8) +
  geom_abline() + geom_smooth(method = "lm") +
  labs(title = "Monthly GS — XGBoost: Predicted vs Observed (Validation)",
       x = "Observed Yield (Mg/ha)", y = "Predicted Yield (Mg/ha)") +
  theme_bw()
save_plot(p_xgb_pred, "12_xgb_pred_vs_obs.png")

p_lgbm_pred <- lgbm_preds_val %>%
  ggplot(aes(x = yield_mg_ha, y = .pred)) +
  geom_point(alpha = 0.3, color = "#1a9641", size = 0.8) +
  geom_abline() + geom_smooth(method = "lm") +
  labs(title = "Monthly GS — LightGBM: Predicted vs Observed (Validation)",
       x = "Observed Yield (Mg/ha)", y = "Predicted Yield (Mg/ha)") +
  theme_bw()
save_plot(p_lgbm_pred, "12_lgbm_pred_vs_obs.png")

save_csv(xgb_preds_val  %>% mutate(model = "XGBoost"),  "12_xgb_val_predictions.csv")
save_csv(lgbm_preds_val %>% mutate(model = "LightGBM"), "12_lgbm_val_predictions.csv")

# Residual plots
p_xgb_resid <- xgb_preds_val %>%
  mutate(residual = yield_mg_ha - .pred) %>%
  ggplot(aes(x = .pred, y = residual)) +
  geom_point(alpha = 0.3, color = "#2c7bb6", size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Monthly GS — XGBoost: Residuals vs Fitted (Validation)",
       x = "Fitted (Predicted) Yield", y = "Residual") +
  theme_bw()
save_plot(p_xgb_resid, "12_xgb_residuals.png")

p_lgbm_resid <- lgbm_preds_val %>%
  mutate(residual = yield_mg_ha - .pred) %>%
  ggplot(aes(x = .pred, y = residual)) +
  geom_point(alpha = 0.3, color = "#1a9641", size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Monthly GS — LightGBM: Residuals vs Fitted (Validation)",
       x = "Fitted (Predicted) Yield", y = "Residual") +
  theme_bw()
save_plot(p_lgbm_resid, "12_lgbm_residuals.png")


# ── Step 16: Variable Importance ─────────────────────────────

xgb_fit_for_vip  <- xgb_final_spec  %>% fit(yield_mg_ha ~ ., data = bake(yield_prep, yield_train))
lgbm_fit_for_vip <- lgbm_final_spec %>% fit(yield_mg_ha ~ ., data = bake(yield_prep, yield_train))

xgb_vi <- xgb_fit_for_vip %>% vi() %>%
  mutate(Variable = fct_reorder(Variable, Importance))
save_csv(xgb_vi, "13_xgb_variable_importance.csv")

p_xgb_vip <- xgb_vi %>%
  slice_max(Importance, n = 20) %>%
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#2c7bb6") +
  scale_x_continuous(expand = c(0, 0)) +
  labs(title = "Monthly GS — XGBoost: Variable Importance (Top 20)", y = NULL) +
  theme_bw()
save_plot(p_xgb_vip, "13_xgb_vip.png", h = 6)

lgbm_vi <- lgbm_fit_for_vip %>% vi() %>%
  mutate(Variable = fct_reorder(Variable, Importance))
save_csv(lgbm_vi, "13_lgbm_variable_importance.csv")

p_lgbm_vip <- lgbm_vi %>%
  slice_max(Importance, n = 20) %>%
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#1a9641") +
  scale_x_continuous(expand = c(0, 0)) +
  labs(title = "Monthly GS — LightGBM: Variable Importance (Top 20)", y = NULL) +
  theme_bw()
save_plot(p_lgbm_vip, "13_lgbm_vip.png", h = 6)


# ── Step 17: Predict on Held-Out Test Set ────────────────────

xgb_full_fit  <- xgb_final_spec  %>% fit(yield_mg_ha ~ ., data = bake(yield_prep, train_raw))
lgbm_full_fit <- lgbm_final_spec %>% fit(yield_mg_ha ~ ., data = bake(yield_prep, train_raw))

xgb_test_preds  <- predict(xgb_full_fit,  new_data = bake(yield_prep, test_raw, all_predictors())) %>% rename(xgb_pred  = .pred)
lgbm_test_preds <- predict(lgbm_full_fit, new_data = bake(yield_prep, test_raw, all_predictors())) %>% rename(lgbm_pred = .pred)

test_predictions <- test_raw %>% bind_cols(xgb_test_preds, lgbm_test_preds)
save_csv(test_predictions, "14_test_predictions.csv")


# ── Step 18: Master Summary File ─────────────────────────────

sink(file.path(out_dir, paste0(model_tag, "_00_run_summary.txt")))

cat("============================================================\n")
cat(" Corn Yield Prediction — Run Summary\n")
cat(" Model      : Monthly Growing Season Weather (monthly_gs)\n")
cat(" CRSS 8030  | Final Project Group 8 | 2026\n")
cat(" Output dir :", out_dir, "\n")
cat(" Generated  :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")

cat("--- Data ---\n")
cat("  Training file         :", basename(train_path), "\n")
cat("  Testing file          :", basename(test_path),  "\n")
cat("  Training rows         :", nrow(train_raw), "\n")
cat("  Test rows             :", nrow(test_raw),  "\n")
cat("  Predictors (post-prep):", n_preds, "\n\n")

cat("--- CV Setup ---\n")
cat("  10-fold CV, stratified on yield_mg_ha\n\n")

cat("--- Final Hyperparameters (best_r2 strategy) ---\n")
cat("  XGBoost:\n")
cat(sprintf("    trees=%d  tree_depth=%d  min_n=%d\n",
    xgb_best_r2$trees, xgb_best_r2$tree_depth, xgb_best_r2$min_n))
cat(sprintf("    loss_reduction=%.5f  sample_size=%.3f\n",
    xgb_best_r2$loss_reduction, xgb_best_r2$sample_size))
cat(sprintf("    mtry=%d  learn_rate=%.5f\n\n",
    xgb_best_r2$mtry, xgb_best_r2$learn_rate))
cat("  LightGBM:\n")
cat(sprintf("    trees=%d  tree_depth=%d  min_n=%d\n",
    lgbm_best_r2$trees, lgbm_best_r2$tree_depth, lgbm_best_r2$min_n))
cat(sprintf("    loss_reduction=%.5f  sample_size=%.3f\n",
    lgbm_best_r2$loss_reduction, lgbm_best_r2$sample_size))
cat(sprintf("    mtry=%d  learn_rate=%.5f\n\n",
    lgbm_best_r2$mtry, lgbm_best_r2$learn_rate))

cat("--- Metrics Summary ---\n")
print(as.data.frame(metrics_summary), row.names = FALSE)
cat("\n")

cat("--- Output Files Written ---\n")
for (f in sort(list.files(out_dir))) cat(" ", f, "\n")

cat("\n============================================================\n")
sink()
message("Summary saved: ", model_tag, "_00_run_summary.txt")

cat("\n✓ All Monthly GS outputs written to:", out_dir, "\n")

# ── Done ─────────────────────────────────────────────────────
