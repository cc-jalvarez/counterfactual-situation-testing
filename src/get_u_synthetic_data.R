
set.seed(42)

# Set working directory
dir = "C:/Users/Jose Alvarez/Documents/Projects/CounterfactualSituationTesting/"
setwd(dir)

# Set folder paths
path_data = paste(dir, "data/", sep = "")
path_mdls = paste(dir, "src/stan_models/", sep = "")
path_rslt = paste(dir, "results/", sep = "")

# Packages
library(dplyr)
library(caret)
library(data.table)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Setup -------------------------------------------------------------------

org_df <- read.csv(file = paste(path_data, "karimi2020_v1.csv", sep = ""), sep = '|')

df <- org_df %>% select("LoanApproval", "AnnualSalary", "AccountBalance")
df$LoanApproval <- as.numeric(df$LoanApproval==1)
# vars_x <- c("AnnualSalary", "AccountBalance")
# vars_y <- c("LoanApproval")
# vars_for_betas <- c("AnnualSalary")

summary(df$AnnualSalary)
summary(df$AccountBalance)

# V1 ----------------------------------------------------------------------

# From Karimi2020: Account Balance uses a lambda of 2500 (so 2500^2 for sigma)

# ---
# Prepare data for Stan
train_v1 <- list(N = nrow(df),
                 asalary = df[ , c("AnnualSalary")],
                 account = df[ , c("AccountBalance")]
                 )

# FIT: run the MCMC
fit_train_v1 <-
  stan(file = paste(path_mdls, 'karimi2020_v1_train.stan', sep=""),
       data = train_v1,
       iter = 2000,
       chains = 1,
       verbose = TRUE)

# LA: extract the information
la_train_v1 <- extract(fit_train_v1, permuted=TRUE)

u_account <- colMeans(la_train_v1$u_account)
hist(u_account)

lambda_account <- mean(la_train_v1$lambda_account)
lambda_account
hist(lambda_account*u_account)
beta_1 <- mean(la_train_v1$beta_1)
beta_1
intercept <- mean(la_train_v1$intercept)
intercept
hist(lambda_account*u_account + intercept)

upd_df <- org_df
upd_df$u2 <- u_account
upd_df$lambda_u2 <- lambda_account*u_account
upd_df$hat_account <- beta_1*df[ , c("AnnualSalary")] + lambda_account*u_account + intercept
upd_df$diff_account <- upd_df$hat_account - upd_df$AccountBalance
hist(upd_df$diff_account)
summary(upd_df$diff_account)
plot(upd_df$diff_account)

mse_account <- round( sqrt( sum( (upd_df$diff_account)^2 ) / nrow(upd_df) ) )

mse_account*100/mean(upd_df$AccountBalance) # the MSE is 8% of the mean.. not terrible, no?



