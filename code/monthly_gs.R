# Build Revised Monthly Weather Files Focused on Growing Season
# Keep only biologically relevant months for corn: Apr, May, Jun, Jul, Aug, Sep, Oct


library(tidyverse)
library(janitor)

# File paths
train_monthly_path <- "output/final_model_files/training_model_monthly_weather_final.csv"
test_monthly_path  <- "output/final_model_files/testing_model_monthly_weather_final.csv"

out_dir <- "output/final_model_files"

# Read data
train_monthly <- read_csv(train_monthly_path, show_col_types = FALSE) %>%
  clean_names()

test_monthly <- read_csv(test_monthly_path, show_col_types = FALSE) %>%
  clean_names()

# Inspect columns
names(train_monthly)

# Define months to keep
grow_months <- c("apr", "may", "jun", "jul", "aug", "sep", "oct")

# Define non-weather columns to always keep
train_id_cols <- c(
  "yield_mg_ha",
  "year",
  "site",
  "hybrid",
  "previous_crop",
  "longitude",
  "latitude",
  "soilp_h",
  "om_pct",
  "soilk_ppm",
  "soilp_ppm"
)

test_id_cols <- setdiff(train_id_cols, "yield_mg_ha")

# Select monthly weather columns only for growing-season months
train_weather_keep <- names(train_monthly) %>%
  setdiff(train_id_cols) %>%
  keep(~ str_detect(.x, paste0("_(", paste(grow_months, collapse = "|"), ")$")))

test_weather_keep <- names(test_monthly) %>%
  setdiff(test_id_cols) %>%
  keep(~ str_detect(.x, paste0("_(", paste(grow_months, collapse = "|"), ")$")))

# Build revised datasets
training_model_monthly_gs_final <- train_monthly %>%
  select(all_of(train_id_cols), all_of(train_weather_keep))

testing_model_monthly_gs_final <- test_monthly %>%
  select(all_of(test_id_cols), all_of(test_weather_keep))

# Check dimensions
dim(training_model_monthly_gs_final)
dim(testing_model_monthly_gs_final)

# Check predictor alignment
train_predictors <- training_model_monthly_gs_final %>%
  select(-yield_mg_ha) %>%
  names()

test_predictors <- names(testing_model_monthly_gs_final)

identical(train_predictors, test_predictors)

# Save files
write_csv(
  training_model_monthly_gs_final,
  file.path(out_dir, "training_model_monthly_gs_final.csv")
)

write_csv(
  testing_model_monthly_gs_final,
  file.path(out_dir, "testing_model_monthly_gs_final.csv")
)

cat("Saved revised growing-season monthly files to:", out_dir, "\n")