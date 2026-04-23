# ============================================================
# Corn Yield Prediction — Monthly GS Parent Model
# Cluster-ready version
# Uses parent-based hybrid features and drops raw hybrid
# ============================================================

# Step 1: Load Libraries
library(tidyverse)
library(tidymodels)
library(finetune)
library(vip)
library(xgboost)
library(bonsai)
library(lightgbm)
library(doParallel)
library(stringr)

tidymodels_prefer()
set.seed(2026)

# Step 2: File Paths
# Adjust these paths for your cluster environment
train_path <- "output/final_model_files/training_model_monthly_gs_final.csv"
test_path  <- "output/final_model_files/testing_model_monthly_gs_final.csv"

out_dir <- "output/monthly_gs_parent_cluster"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Step 3: Logging Helper
log_file <- file.path(out_dir, "run_log.txt")

log_msg <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  write(msg, file = log_file, append = TRUE)
}

write("RUN START", file = log_file)

# Step 4: Read Data
log_msg("Step 4: reading data")
train_raw <- read_csv(train_path, show_col_types = FALSE)
test_raw  <- read_csv(test_path, show_col_types = FALSE)

log_msg("train rows = ", nrow(train_raw), ", cols = ", ncol(train_raw))
log_msg("test rows = ", nrow(test_raw), ", cols = ", ncol(test_raw))

# Step 5: Hybrid Parent Feature Engineering
log_msg("Step 5: adding parent features")

add_hybrid_parent_features <- function(df) {
  split_mat <- str_split_fixed(as.character(df$hybrid), "/", 2)
  
  df %>%
    mutate(
      hybrid = as.character(hybrid),
      parent_a = if_else(
        str_detect(hybrid, "/"),
        str_trim(split_mat[, 1]),
        str_trim(hybrid)
      ),
      parent_b = if_else(
        str_detect(hybrid, "/"),
        str_trim(split_mat[, 2]),
        "none"
      ),
      hybrid_has_slash = if_else(str_detect(hybrid, "/"), "yes", "no")
    )
}

train_raw <- add_hybrid_parent_features(train_raw)
test_raw  <- add_hybrid_parent_features(test_raw)

write_csv(
  train_raw %>%
    select(hybrid, parent_a, parent_b, hybrid_has_slash) %>%
    head(20),
  file.path(out_dir, "parent_feature_preview.csv")
)

# Step 6: EDA Outputs
log_msg("Step 6: writing EDA outputs")

p_yield <- ggplot(train_raw, aes(x = yield_mg_ha)) +
  geom_histogram(bins = 60, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  labs(title = "Yield distribution (training)",
       x = "yield_mg_ha", y = "count") +
  theme_bw()

ggsave(file.path(out_dir, "01_yield_distribution.png"), p_yield, width = 8, height = 6)

miss_tbl <- train_raw %>%
  summarise(across(everything(), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>%
  filter(pct_missing > 0) %>%
  arrange(desc(pct_missing))

write_csv(miss_tbl, file.path(out_dir, "02_missing_data_table.csv"))

p_missing <- ggplot(miss_tbl, aes(x = reorder(variable, pct_missing), y = pct_missing)) +
  geom_col(fill = "#d7191c") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Missing data (training)", x = NULL, y = "% missing") +
  theme_bw()

ggsave(file.path(out_dir, "02_missing_data.png"), p_missing, width = 8, height = 6)

# Step 7: Recipe
# Key revision:
# Drop raw hybrid to avoid recipe blow-up
# Keep parent_a, parent_b, hybrid_has_slash
log_msg("Step 7: building recipe")

yield_recipe <- recipe(yield_mg_ha ~ ., data = train_raw) %>%
  step_rm(hybrid) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(previous_crop, threshold = 0.005) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors())

log_msg("Step 8: prepping recipe")
yield_prep <- prep(yield_recipe, training = train_raw)

n_preds <- ncol(bake(yield_prep, new_data = NULL, all_predictors()))
log_msg("Predictors after preprocessing: ", n_preds)

write_lines(
  paste("Predictors after preprocessing:", n_preds),
  file.path(out_dir, "03_predictor_count.txt")
)

# Step 8: Cross-Validation 
log_msg("Step 9: building CV folds")

set.seed(235)
cv_folds <- vfold_cv(
  train_raw,
  v = 10,
  strata = yield_mg_ha
)

# Step 9: Model Specs
log_msg("Step 10: defining models")

xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune(),
  learn_rate     = tune()
) %>%
  set_engine(
    "xgboost",
    nthread = max(1, parallel::detectCores() - 1),
    early_stopping_rounds = 50
  ) %>%
  set_mode("regression")

lgbm_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune(),
  learn_rate     = tune()
) %>%
  set_engine(
    "lightgbm",
    num_threads = max(1, parallel::detectCores() - 1)
  ) %>%
  set_mode("regression")

# Step 10: Grids
log_msg("Step 11: creating tuning grids")

set.seed(345)

xgb_grid <- grid_space_filling(
  trees(range = c(300L, 2000L)),
  tree_depth(range = c(3L, 10L)),
  min_n(range = c(5L, 40L)),
  loss_reduction(range = c(-5, 2)),
  sample_size = sample_prop(range = c(0.5, 1.0)),
  mtry(range = c(3L, as.integer(ceiling(n_preds * 0.8)))),
  learn_rate(range = c(-3, -1)),
  size = 100,
  type = "latin_hypercube"
)

lgbm_grid <- grid_space_filling(
  trees(range = c(300L, 2000L)),
  tree_depth(range = c(3L, 10L)),
  min_n(range = c(5L, 40L)),
  loss_reduction(range = c(-5, 2)),
  sample_size = sample_prop(range = c(0.5, 1.0)),
  mtry(range = c(3L, as.integer(ceiling(n_preds * 0.8)))),
  learn_rate(range = c(-3, -1)),
  size = 100,
  type = "latin_hypercube"
)

write_csv(xgb_grid, file.path(out_dir, "04_xgb_hyperparam_grid.csv"))
write_csv(lgbm_grid, file.path(out_dir, "04_lgbm_hyperparam_grid.csv"))

p_grid <- ggplot(xgb_grid, aes(x = tree_depth, y = min_n)) +
  geom_point(aes(color = factor(round(learn_rate, 3)), size = trees),
             alpha = 0.5, show.legend = FALSE) +
  labs(title = "XGBoost Hyperparameter Grid",
       x = "Tree Depth", y = "Min N") +
  theme_bw()

ggsave(file.path(out_dir, "04_xgb_grid_viz.png"), p_grid, width = 8, height = 6)

#  Step 11: Tune Models
log_msg("Step 12: tuning XGBoost")
registerDoParallel(cores = max(1, parallel::detectCores() - 1))

set.seed(76544)
xgb_res <- tune_race_anova(
  object       = xgb_spec,
  preprocessor = yield_recipe,
  resamples    = cv_folds,
  grid         = xgb_grid,
  metrics      = metric_set(rmse, rsq, mae),
  control      = control_race(save_pred = TRUE, verbose = TRUE)
)

stopImplicitCluster()

write_csv(collect_metrics(xgb_res), file.path(out_dir, "05_xgb_tune_metrics_all.csv"))

log_msg("Step 13: tuning LightGBM")
registerDoParallel(cores = max(1, parallel::detectCores() - 1))

set.seed(76545)
lgbm_res <- tune_race_anova(
  object       = lgbm_spec,
  preprocessor = yield_recipe,
  resamples    = cv_folds,
  grid         = lgbm_grid,
  metrics      = metric_set(rmse, rsq, mae),
  control      = control_race(save_pred = TRUE, verbose = TRUE)
)

stopImplicitCluster()

write_csv(collect_metrics(lgbm_res), file.path(out_dir, "05_lgbm_tune_metrics_all.csv"))

# Step 12: Select Best 
log_msg("Step 14: selecting best hyperparameters")

xgb_best <- select_best(xgb_res, metric = "rsq")
lgbm_best <- select_best(lgbm_res, metric = "rsq")

write_csv(xgb_best, file.path(out_dir, "07_xgb_final_best.csv"))
write_csv(lgbm_best, file.path(out_dir, "07_lgbm_final_best.csv"))

xgb_final_spec <- finalize_model(xgb_spec, xgb_best)
lgbm_final_spec <- finalize_model(lgbm_spec, lgbm_best)

# Step 13: Final Validation Split 
log_msg("Step 15: final validation split")

set.seed(931735)
yield_split <- initial_split(train_raw, prop = 0.8, strata = yield_mg_ha)
yield_train <- training(yield_split)
yield_val   <- testing(yield_split)

p_split <- ggplot() +
  geom_density(data = yield_train, aes(x = yield_mg_ha), color = "red") +
  geom_density(data = yield_val, aes(x = yield_mg_ha), color = "blue") +
  labs(title = "Yield Distribution — Train (red) vs Validation (blue)",
       x = "Yield (Mg/ha)") +
  theme_bw()

ggsave(file.path(out_dir, "08_train_val_distribution.png"), p_split, width = 8, height = 6)

# Step 14: last_fit Validation
log_msg("Step 16: running final validation")

set.seed(10)

xgb_final_fit <- last_fit(
  xgb_final_spec,
  preprocessor = yield_recipe,
  split = yield_split
)

lgbm_final_fit <- last_fit(
  lgbm_final_spec,
  preprocessor = yield_recipe,
  split = yield_split
)

xgb_val_metrics <- collect_metrics(xgb_final_fit) %>% mutate(model = "XGBoost")
lgbm_val_metrics <- collect_metrics(lgbm_final_fit) %>% mutate(model = "LightGBM")

write_csv(xgb_val_metrics, file.path(out_dir, "09_xgb_validation_metrics.csv"))
write_csv(lgbm_val_metrics, file.path(out_dir, "09_lgbm_validation_metrics.csv"))

xgb_val_preds <- collect_predictions(xgb_final_fit)
lgbm_val_preds <- collect_predictions(lgbm_final_fit)

# Step 15: Predicted vs Observed
log_msg("Step 17: writing predicted vs observed plots")

p_xgb <- ggplot(xgb_val_preds, aes(x = yield_mg_ha, y = .pred)) +
  geom_point(alpha = 0.3, color = "#2c7bb6", size = 0.8) +
  geom_abline() +
  geom_smooth(method = "lm") +
  labs(title = "XGBoost — Predicted vs Observed",
       x = "Observed Yield", y = "Predicted Yield") +
  theme_bw()

p_lgbm <- ggplot(lgbm_val_preds, aes(x = yield_mg_ha, y = .pred)) +
  geom_point(alpha = 0.3, color = "#1a9641", size = 0.8) +
  geom_abline() +
  geom_smooth(method = "lm") +
  labs(title = "LightGBM — Predicted vs Observed",
       x = "Observed Yield", y = "Predicted Yield") +
  theme_bw()

ggsave(file.path(out_dir, "12_xgb_pred_vs_obs.png"), p_xgb, width = 8, height = 6)
ggsave(file.path(out_dir, "12_lgbm_pred_vs_obs.png"), p_lgbm, width = 8, height = 6)

write_csv(xgb_val_preds, file.path(out_dir, "12_xgb_val_predictions.csv"))
write_csv(lgbm_val_preds, file.path(out_dir, "12_lgbm_val_predictions.csv"))

# Step 16: Fit Final Models on All Training Data
log_msg("Step 18: fitting final models on all training data")

xgb_full_fit <- fit(
  workflow() %>%
    add_recipe(yield_recipe) %>%
    add_model(xgb_final_spec),
  data = train_raw
)

lgbm_full_fit <- fit(
  workflow() %>%
    add_recipe(yield_recipe) %>%
    add_model(lgbm_final_spec),
  data = train_raw
)

# Step 17: Test Predictions
log_msg("Step 19: predicting test set")

xgb_test_preds <- predict(xgb_full_fit, new_data = test_raw) %>%
  rename(xgb_pred = .pred)

lgbm_test_preds <- predict(lgbm_full_fit, new_data = test_raw) %>%
  rename(lgbm_pred = .pred)

test_predictions <- test_raw %>%
  select(year, site, hybrid) %>%
  bind_cols(xgb_test_preds, lgbm_test_preds)

write_csv(test_predictions, file.path(out_dir, "14_test_predictions.csv"))

site_pred_summary <- test_predictions %>%
  group_by(site) %>%
  summarise(
    n_rows = n(),
    n_unique_xgb = n_distinct(xgb_pred),
    n_unique_lgbm = n_distinct(lgbm_pred),
    .groups = "drop"
  ) %>%
  arrange(n_unique_xgb)

write_csv(site_pred_summary, file.path(out_dir, "14_site_prediction_summary.csv"))

# Step 18: Variable Importance
log_msg("Step 20: variable importance")

p_vip_xgb <- xgb_full_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20, geom = "col",
      aesthetics = list(fill = "#2c7bb6", color = "white")) +
  labs(title = "XGBoost — Top 20 Features") +
  theme_bw()

p_vip_lgbm <- lgbm_full_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20, geom = "col",
      aesthetics = list(fill = "#1a9641", color = "white")) +
  labs(title = "LightGBM — Top 20 Features") +
  theme_bw()

ggsave(file.path(out_dir, "13_xgb_vip.png"), p_vip_xgb, width = 8, height = 6)
ggsave(file.path(out_dir, "13_lgbm_vip.png"), p_vip_lgbm, width = 8, height = 6)

# Step 19: Save Fitted Models
log_msg("Step 21: saving fitted models")

saveRDS(xgb_full_fit, file.path(out_dir, "xgb_monthly_gs_parent_fit.rds"))
saveRDS(lgbm_full_fit, file.path(out_dir, "lgbm_monthly_gs_parent_fit.rds"))

# Step 20: Final Summary
log_msg("SUCCESS: completed full run")
sessionInfo()