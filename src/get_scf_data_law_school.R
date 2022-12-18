# Seed
set.seed(42)

# Packages
library(dplyr)
library(caret)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(data.table)

# Setup -------------------------------------------------------------------

# Set working directory
dir = "C:/Users/Jose Alvarez/Documents/Projects/CounterfactualSituationTesting/"
setwd(dir)

# Set folder paths
path_data = paste(dir, "data/", sep = "")
path_mdls = paste(dir, "src/stan_models/", sep = "")
path_rslt = paste(dir, "results/", sep = "")

# original data
org_df <- read.csv(file = paste(path_data, "clean_LawData.csv", sep = ""), sep = '|')

# initial vars
use_race = "race_nonwhite"
vars <- c("LSAT", "UGPA", "sex")
vars <- append(vars, use_race)
# modified vars for scf generation
vars_m <- c("LSAT", "UGPA")

# modeling data
df <- org_df %>% select(vars)

# var transformation
df$LSAT <- round(df$LSAT)
df$female <- as.numeric(df$sex == "Female")
df$male <- as.numeric(df$sex == "Male")
sense_cols <- c("female", "male")

if (use_race == "race_nonwhite"){
  df$white <- as.numeric(df$race_nonwhite == "White")
  sense_cols <- append(sense_cols, "white")
  df$nonwhite <- as.numeric(df$race_nonwhite == "NonWhite")
  sense_cols <- append(sense_cols, "nonwhite")
}

if (use_race == "race_simpler"){
  df$white <- as.numeric(df$race_simpler == "White")
  sense_cols <- append(sense_cols, "white")
  df$black <- as.numeric(df$race_simpler == "Black")
  sense_cols <- append(sense_cols, "black")
  df$Latino <- as.numeric(df$race_simpler == "Latino")
  sense_cols <- append(sense_cols, "latino")
  df$asian <- as.numeric(df$race_simpler == "Asian")
  sense_cols <- append(sense_cols, "asian")
  df$other <- as.numeric(df$race_simpler == "Other")
  sense_cols <- append(sense_cols, "Other")
}

print(sense_cols)
vars_m <- append(vars_m, sense_cols)

# The Protected Attributes
table(df$sex)
table(df$race_nonwhite)

# #-- test on a sample
# trainIndex <- createDataPartition(df$sex, p = .5, list = FALSE, times = 1)
# df <- df[trainIndex, ]

table(df$sex)
table(df$race_nonwhite)

# Abduction step via MCMC -------------------------------------------------

# Prepare data for Stan
law_school_train <- list(N = nrow(df), K = length(sense_cols),
                         a = data.matrix(df[ , sense_cols]),
                         ugpa = df[ , c("UGPA")],
                         lsat = df[ , c("LSAT")])

# Run the MCMC
fit_law_school_train <- stan(file = paste(path_mdls, 'law_school_train.stan', sep=""),
                             data = law_school_train,iter = 2000,
                             chains = 1,
                             verbose = TRUE)

# Extract the information
la_law_school_train <- extract(fit_law_school_train, permuted=TRUE)

# Get (hyper)parameters
U          <- colMeans(la_law_school_train$u)
ugpa0      <- mean(la_law_school_train$ugpa0)
eta_u_ugpa <- mean(la_law_school_train$eta_u_ugpa)
eta_a_ugpa <- colMeans(la_law_school_train$eta_a_ugpa)
lsat0      <- mean(la_law_school_train$lsat0)
eta_u_lsat <- mean(la_law_school_train$eta_u_lsat)
eta_a_lsat <- colMeans(la_law_school_train$eta_a_lsat)
SIGMA_G    <- mean(la_law_school_train$sigma_g)

# save all
# get scf here (no need for a python script...)

# write results to counterfactual folfer in data!!!

# Abduction step via residuals --------------------------------------------


# use same specification as above but estimate residuals via LM
# in level 3, u seems to be a common var (include it in the regression)




#
# EOF
#

