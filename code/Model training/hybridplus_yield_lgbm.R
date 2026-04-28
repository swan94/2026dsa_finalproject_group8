# Corn Yield Prediction — Monthly (Growing Season) Parent Model + Hybrid Model — LightGBM Only

# Step 1: Load Libraries
.libPaths(c("~/Rlibs", .libPaths()))

library(tidyverse)
library(tidymodels)
library(vip)
library(bonsai)
library(lightgbm)
library(doParallel)
library(stringr)

tidymodels_prefer()
set.seed(2026)

# Step 2: File Paths
train_path <- "data/training_model_monthly_gs_final.csv"
test_path  <- "data/testing_model_monthly_gs_final.csv"

out_dir <- "output/monthly_gs_hybridplus_lgbm"
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

# Step 5: Hybrid Feature Engineering
log_msg("Step 5: adding hybrid and parent features")

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
      parent_pair = paste(parent_a, parent_b, sep = "_x_"),
      hybrid_has_slash = if_else(str_detect(hybrid, "/"), "yes", "no")
    )
}

train_raw <- add_hybrid_parent_features(train_raw)
test_raw  <- add_hybrid_parent_features(test_raw)

write_csv(
  train_raw %>%
    select(hybrid, parent_a, parent_b, parent_pair, hybrid_has_slash) %>%
    head(20),
  file.path(out_dir, "hybrid_parent_feature_preview.csv")
)

# Step 6: EDA Outputs
log_msg("Step 6: writing EDA outputs")

p_yield <- ggplot(train_raw, aes(x = yield_mg_ha)) +
  geom_histogram(bins = 60, fill = "#1a9641", color = "white", alpha = 0.85) +
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
log_msg("Step 7: building recipe")

yield_recipe <- recipe(yield_mg_ha ~ ., data = train_raw) %>%
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

# Step 9: Cross-Validation
log_msg("Step 9: building CV folds")

set.seed(235)
cv_folds <- vfold_cv(
  train_raw,
  v = 10,
  strata = yield_mg_ha
)

# Step 10: LightGBM Spec
log_msg("Step 10: defining LightGBM model")

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

# Step 11: LightGBM Grid
log_msg("Step 11: creating LightGBM tuning grid")

set.seed(345)

lgbm_grid <- grid_space_filling(
  trees(range = c(300L, 2000L)),
  tree_depth(range = c(3L, 10L)),
  min_n(range = c(5L, 40L)),
  loss_reduction(range = c(-5, 2)),
  sample_size = sample_prop(range = c(0.5, 1.0)),
  mtry(range = c(3L, as.integer(ceiling(n_preds * 0.8)))),
  learn_rate(range = c(-3, -1)),
  size = 50,
  type = "latin_hypercube"
)

write_csv(lgbm_grid, file.path(out_dir, "04_lgbm_hyperparam_grid.csv"))

# Step 12: Tune LightGBM
log_msg("Step 12: tuning LightGBM")
registerDoParallel(cores = max(1, parallel::detectCores() - 1))

set.seed(76545)
lgbm_res <- tune_grid(
  object       = lgbm_spec,
  preprocessor = yield_recipe,
  resamples    = cv_folds,
  grid         = lgbm_grid,
  metrics      = metric_set(rmse, rsq, mae),
  control      = control_grid(save_pred = TRUE, verbose = TRUE)
)

stopImplicitCluster()

write_csv(collect_metrics(lgbm_res), file.path(out_dir, "05_lgbm_tune_metrics_all.csv"))

# Step 13: Select Best
log_msg("Step 13: selecting best hyperparameters")

lgbm_best <- select_best(lgbm_res, metric = "rsq")
write_csv(lgbm_best, file.path(out_dir, "07_lgbm_final_best.csv"))

lgbm_final_spec <- finalize_model(lgbm_spec, lgbm_best)

# Step 14: Final Validation Split
log_msg("Step 14: final validation split")

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

# Step 15: Final Validation
log_msg("Step 15: running final validation")

set.seed(10)

lgbm_final_fit <- last_fit(
  lgbm_final_spec,
  preprocessor = yield_recipe,
  split = yield_split
)

lgbm_val_metrics <- collect_metrics(lgbm_final_fit) %>% mutate(model = "LightGBM")
write_csv(lgbm_val_metrics, file.path(out_dir, "09_lgbm_validation_metrics.csv"))

lgbm_val_preds <- collect_predictions(lgbm_final_fit)

# Step 16: Predicted vs Observed
log_msg("Step 16: writing predicted vs observed plot")

p_lgbm <- ggplot(lgbm_val_preds, aes(x = yield_mg_ha, y = .pred)) +
  geom_point(alpha = 0.3, color = "#1a9641", size = 0.8) +
  geom_abline() +
  geom_smooth(method = "lm") +
  labs(title = "LightGBM — Predicted vs Observed",
       x = "Observed Yield", y = "Predicted Yield") +
  theme_bw()

ggsave(file.path(out_dir, "12_lgbm_pred_vs_obs.png"), p_lgbm, width = 8, height = 6)
write_csv(lgbm_val_preds, file.path(out_dir, "12_lgbm_val_predictions.csv"))

# Step 17: Fit Final Model on All Training Data
log_msg("Step 17: fitting final LightGBM model on all training data")

lgbm_full_fit <- fit(
  workflow() %>%
    add_recipe(yield_recipe) %>%
    add_model(lgbm_final_spec),
  data = train_raw
)

# Step 18: Test Predictions
log_msg("Step 18: predicting test set")

lgbm_test_preds <- predict(lgbm_full_fit, new_data = test_raw) %>%
  rename(lgbm_pred = .pred)

test_predictions <- test_raw %>%
  select(year, site, hybrid) %>%
  bind_cols(lgbm_test_preds)

write_csv(test_predictions, file.path(out_dir, "14_test_predictions.csv"))

site_pred_summary <- test_predictions %>%
  group_by(site) %>%
  summarise(
    n_rows = n(),
    n_unique_lgbm = n_distinct(lgbm_pred),
    .groups = "drop"
  ) %>%
  arrange(n_unique_lgbm)

write_csv(site_pred_summary, file.path(out_dir, "14_site_prediction_summary.csv"))

# Step 19: Variable Importance
log_msg("Step 19: variable importance")

p_vip_lgbm <- lgbm_full_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20, geom = "col",
      aesthetics = list(fill = "#1a9641", color = "white")) +
  labs(title = "LightGBM — Top 20 Features") +
  theme_bw()

ggsave(file.path(out_dir, "13_lgbm_vip.png"), p_vip_lgbm, width = 8, height = 6)

# Step 20: Save Model
log_msg("Step 20: saving fitted LightGBM model")
saveRDS(lgbm_full_fit, file.path(out_dir, "lgbm_monthly_gs_hybridplus_fit.rds"))

log_msg("SUCCESS: completed LightGBM hybrid-plus run")
sessionInfo()