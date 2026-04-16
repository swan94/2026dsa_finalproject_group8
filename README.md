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