
set.seed(42)

# Set working directory
dir = "C:/Users/Jose Alvarez/Documents/Projects/CounterfactualSituationTesting/"
setwd(dir)

# Set folder paths
path_data = paste(dir, "data/", sep = "")
path_code = paste(dir, "src/", sep = "")
path_rslt = paste(dir, "results/", sep = "")

# Packages
library(dplyr)
library(caret)
library(data.table)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Setup -------------------------------------------------------------------
