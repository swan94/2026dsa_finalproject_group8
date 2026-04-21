#!/bin/bash

#SBATCH --job-name=monthly_yield_xgb_lgbm         # Job name
#SBATCH --partition=batch                # Partition
#SBATCH --nodes=1                        # Request 1 node (Better for R)
#SBATCH --ntasks=1                       # Run 1 instance of R
#SBATCH --cpus-per-task=32               # Give that 1 instance all 32 cores
#SBATCH --mem=32G                        # 32GB RAM total
#SBATCH --time=15:00:00                  # Time limit (hrs:min:sec)
#SBATCH --mail-user=hm64666@uga.edu      # Your UGA email
#SBATCH --mail-type=BEGIN,END,FAIL       # Notifications
#SBATCH --output=%x_%j.out               # Standard output log
#SBATCH --error=%x_%j.err                # Standard error log

# 1. Clean the environment and load modules
module purge
module load R/4.4.2-gfbf-2024a
#module load CMake/3.27.6-GCCcore-13.2.0

# 2. Run the R script
Rscript 	yield_xgb_lgbm_monthly_gs.R
