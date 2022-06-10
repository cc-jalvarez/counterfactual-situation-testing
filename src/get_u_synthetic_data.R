
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


# V1 ----------------------------------------------------------------------

# Under causal sufficiency, there's nothing to discover for the abduction step
# as we know how the error will appear and, under correct specification, we are
# able to retrieve them. It's pointless for this case to go too technical.


# V2 ----------------------------------------------------------------------

org_df <- read.csv(file = paste(path_data, "karimi2020_v2.csv", sep = ""), sep = '|')

# modeling dataset
df <- org_df %>% select("LoanApproval", "AnnualSalary", "AccountBalance")
df$LoanApproval <- as.numeric(df$LoanApproval==1)

# assume linear Gaussian ADM (i.e., a f*cking OLS)
model1 <- lm(AccountBalance ~ AnnualSalary + 1, data = df)
model1
# check u2 which will determine the scf
u2_hat <- df$AccountBalance - predict.lm(model1)
plot(u2_hat)
plot(org_df$u2)
hist(u2_hat)
hist(org_df$u2, add=TRUE)

# ---
# Prepare data for Stan
train_v2 <- list(N = nrow(df),
                 asalary = df[ , c("AnnualSalary")],
                 account = df[ , c("AccountBalance")]
                 )

# FIT: run the MCMC
fit_train_v2 <-
  stan(file = paste(path_mdls, 'karimi2020_v2_train.stan', sep=""),
       data = train_v2,
       iter = 2000,
       chains = 1,
       verbose = TRUE)

# LA: extract the information
la_train_v2 <- extract(fit_train_v2, permuted=TRUE)

z <- colMeans(la_train_v2$z)
hist(org_df$z)
hist(z) #, add=TRUE)

lambda.asalary <- mean(la_train_v2$lambda_asalary)
lambda.account <- mean(la_train_v2$lambda_account)

b0.asalary <- mean(la_train_v2$beta_0_asalary)
print(b0.asalary)

# compare to model1!
b1 <- mean(la_train_v2$beta_1)
print(b1)
b0.account <- mean(la_train_v2$beta_0_account)
print(b0.account)

hist(lambda.account*z) # too small to shift the weights!








