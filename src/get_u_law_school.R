
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
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(data.table)

# Setup -------------------------------------------------------------------

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

print(sense_cols)
vars_m <- append(vars_m, sense_cols)

# The Protected Attributes
table(df$sex)
table(df$race_nonwhite)

# # opt 1: take male and white as benchmark
# sense_cols <- c("female", "nonwhite")
# opt 2: take all
#sense_cols <- c("female", "male", "white", "nonwhite") todo: clean up



#-- temp: test on a sample
trainIndex <- createDataPartition(df$sex, p = .5, list = FALSE, times = 1)
df <- df[trainIndex, ]

table(df$sex)
table(df$race_nonwhite)
#---


# U Pooled ----------------------------------------------------------------

# ---
# Prepare data for Stan
law_school_train_pooled <- list(N = nrow(df),
                                K = length(sense_cols),
                                a = data.matrix(df[ , sense_cols]),
                                ugpa = df[ , c("UGPA")],
                                lsat = df[ , c("LSAT")])

# FIT: run the MCMC
fit_law_school_train_pooled <- 
  stan(file = paste(path_code, 'law_school_train_pooled.stan', sep=""),
       data = law_school_train_pooled,
       iter = 4000,
       chains = 1,
       verbose = TRUE)

# LA: extract the information
la_law_school_train_pooled <- extract(fit_law_school_train_pooled, 
                                      permuted=TRUE)

# ---
# Get (hyper)parameters
U <- colMeans(la_law_school_train_pooled$u)

ugpa0      <- mean(la_law_school_train_pooled$ugpa0)
eta_u_ugpa <- mean(la_law_school_train_pooled$eta_u_ugpa)
eta_a_ugpa <- colMeans(la_law_school_train_pooled$eta_a_ugpa)

lsat0      <- mean(la_law_school_train_pooled$lsat0)
eta_u_lsat <- mean(la_law_school_train_pooled$eta_u_lsat)
eta_a_lsat <- colMeans(la_law_school_train_pooled$eta_a_lsat)

SIGMA_G <- mean(la_law_school_train_pooled$sigma_g)

# ---
hist(U)

# Test 'model' accuracy: UGPA
test_fit_ugpa <- df[, c("UGPA", "sex", use_race)]
test_fit_ugpa$ugpa0 <- ugpa0
test_fit_ugpa$weighted_sense_cols <- data.matrix(df[ , sense_cols])%*%eta_a_ugpa
test_fit_ugpa$weighted_u <- eta_u_ugpa*U
test_fit_ugpa$pred_UGPA <- test_fit_ugpa %>%
  select(ugpa0, weighted_u, weighted_sense_cols) %>% 
  rowSums()
test_fit_ugpa$delta <- test_fit_ugpa$pred_UGPA - test_fit_ugpa$UGPA

plot(test_fit_ugpa$delta)
summary(test_fit_ugpa$delta)
hist(test_fit_ugpa$delta)

# MSE
pu_mse_ugpa <- sqrt( sum( (test_fit_ugpa$delta)^2 ) / nrow(test_fit_ugpa) )
print(pu_mse_ugpa)

# # get MSE
# model_ugp <- lm(UGPA ~ female + male + white + nonwhite + 1, data=df)
# model_ugp
# sqrt( sum( (predict(model_ugp, newdata = df) - test_fit_ugpa$UGPA)^2 ) / nrow(test_fit_ugpa) )

# Test 'model' accuracy: LSAT
test_fit_lsat <- df[, c("LSAT", "sex", use_race)]
test_fit_lsat$lsat0 <- lsat0
test_fit_lsat$weighted_sense_cols <- data.matrix(df[ , sense_cols])%*%eta_a_lsat
test_fit_lsat$weighted_u <- eta_u_lsat*U
test_fit_lsat$pred_LSAT <- test_fit_lsat %>%
  select(lsat0, weighted_u, weighted_sense_cols) %>% 
  rowSums() %>% 
  exp()
test_fit_lsat$delta <- test_fit_lsat$pred_LSAT - test_fit_lsat$LSAT

plot(test_fit_lsat$delta)
summary(test_fit_lsat$delta)
hist(test_fit_lsat$delta)

# MSE
pu_mse_lsat <- sqrt( sum( (test_fit_lsat$delta)^2 ) / nrow(test_fit_lsat) )
print(pu_mse_lsat)

# ---
# keep track of the deltas
write.table(test_fit_ugpa, 
            file = paste(path_rslt,"pU_delta_ugpa.csv", sep = ""), 
            sep = "|")

write.table(test_fit_lsat, 
            file = paste(path_rslt,"pU_delta_lsat.csv", sep = ""), 
            sep = "|")

# Store the data for modeling the scfs
upd_df <- df %>% 
  select(vars_m)
upd_df$U <- U

write.table(upd_df, 
            file = paste(path_rslt,"pU_upd_LawData.csv", sep = ""), 
            sep = "|")

# Store var-specific weights
weights_ugpa <- {}

for (i in 1:length(sense_cols)){
  print(sense_cols[i])
  print(eta_a_ugpa[i])
  weights_ugpa[sense_cols[i]] = eta_a_ugpa[i]
}

weights_ugpa["ugpa0"] <- ugpa0
weights_ugpa["eta_u_ugpa"] <- eta_u_ugpa

temp_weights_ugpa <- as.data.frame(weights_ugpa)
weights_ugpa <- data.table::transpose(temp_weights_ugpa)
colnames(weights_ugpa) <- rownames(temp_weights_ugpa)
rownames(weights_ugpa) <- colnames(temp_weights_ugpa)
remove(temp_weights_ugpa)

write.table(weights_ugpa, 
            file = paste(path_rslt,"pU_wUGPA_LawData.csv", sep = ""), 
            sep = "|", 
            row.names = FALSE)

weights_lsat <- {}

for (i in 1:length(sense_cols)){
  print(sense_cols[i])
  print(eta_a_lsat[i])
  weights_lsat[sense_cols[i]] = eta_a_lsat[i] # exp transform?
}

weights_lsat["lsat0"] <- lsat0 # exp transform?
weights_lsat["eta_u_lsat"] <- eta_u_lsat # exp transform?

temp_weights_lsat <- as.data.frame(weights_lsat)
weights_lsat <- data.table::transpose(temp_weights_lsat)
colnames(weights_lsat) <- rownames(temp_weights_lsat)
rownames(weights_lsat) <- colnames(temp_weights_lsat)
remove(temp_weights_lsat)

write.table(weights_lsat, 
            file = paste(path_rslt,"pU_wLSAT_LawData.csv", sep = ""), 
            sep = "|", 
            row.names = FALSE)

# ---
# clear model output(s) in case of cont. run
remove(test_fit_ugpa, test_fit_lsat)
remove(upd_df, weights_ugpa, weights_lsat)

# U Separated -------------------------------------------------------------

# ---
# Prepare data for Stan
law_school_train_separated <- list(N = nrow(df),
                                   K = length(sense_cols),
                                   a = data.matrix(df[ , sense_cols]),
                                   ugpa = df[ , c("UGPA")],
                                   lsat = df[ , c("LSAT")])

# FIT: run the MCMC
fit_law_school_train_separated <- 
  stan(file = paste(path_code, 'law_school_train_separated.stan', sep=""),
       data = law_school_train_separated,
       iter = 4000,
       chains = 1,
       verbose = TRUE)

# LA: extract the information
la_law_school_train_seperated <- extract(fit_law_school_train_separated, 
                                         permuted=TRUE)

# ---
# Get (hyper)parameters
U_ugpa <- colMeans(la_law_school_train_seperated$u_ugpa)
U_lsat <- colMeans(la_law_school_train_seperated$u_lsat)

ugpa0      <- mean(la_law_school_train_seperated$ugpa0)
eta_u_ugpa <- mean(la_law_school_train_seperated$eta_u_ugpa)
eta_a_ugpa <- colMeans(la_law_school_train_seperated$eta_a_ugpa)

lsat0      <- mean(la_law_school_train_seperated$lsat0)
eta_u_lsat <- mean(la_law_school_train_seperated$eta_u_lsat)
eta_a_lsat <- colMeans(la_law_school_train_seperated$eta_a_lsat)

SIGMA_G <- mean(la_law_school_train_seperated$sigma_g)

# ---
hist(U_ugpa)
hist(U_lsat)

# Test 'model' accuracy: UGPA
test_fit_ugpa <- df[, c("UGPA", "sex", use_race)]
test_fit_ugpa$ugpa0 <- ugpa0
test_fit_ugpa$weighted_sense_cols <- data.matrix(df[ , sense_cols])%*%eta_a_ugpa
test_fit_ugpa$weighted_u <- eta_u_ugpa*U_ugpa
test_fit_ugpa$pred_UGPA <- test_fit_ugpa %>%
  select(ugpa0, weighted_u, weighted_sense_cols) %>% 
  rowSums()

plot(test_fit_ugpa$pred_UGPA - test_fit_ugpa$UGPA)
summary(test_fit_ugpa$pred_UGPA - test_fit_ugpa$UGPA)
hist(test_fit_ugpa$pred_UGPA - test_fit_ugpa$UGPA)

# MSE
su_mse_ugpa <- sqrt( sum( (test_fit_ugpa$pred_UGPA - test_fit_ugpa$UGPA)^2 ) / nrow(test_fit_ugpa) )
print(su_mse_ugpa)

# Test 'model' accuracy: LSAT
test_fit_lsat <- df[, c("LSAT", "sex", use_race)]
test_fit_lsat$lsat0 <- lsat0
test_fit_lsat$weighted_sense_cols <- data.matrix(df[ , sense_cols])%*%eta_a_lsat
test_fit_lsat$weighted_u <- eta_u_lsat*U_lsat
test_fit_lsat$pred_LSAT <- test_fit_lsat %>%
  select(lsat0, weighted_u, weighted_sense_cols) %>% 
  rowSums() %>% 
  exp()

plot(test_fit_lsat$pred_LSAT - test_fit_lsat$LSAT)
summary(test_fit_lsat$pred_LSAT - test_fit_lsat$LSAT)
hist(test_fit_lsat$pred_LSAT - test_fit_lsat$LSAT)

# MSE
su_mse_lsat <- sqrt( sum( (test_fit_lsat$pred_LSAT - test_fit_lsat$LSAT)^2 ) / nrow(test_fit_lsat) )
print(su_mse_lsat)

# ---
# keep track of the deltas
write.table(test_fit_ugpa, 
            file = paste(path_rslt,"sU_delta_ugpa.csv", sep = ""), 
            sep = "|")

write.table(test_fit_lsat, 
            file = paste(path_rslt,"sU_delta_lsat.csv", sep = ""), 
            sep = "|")

# Store the data for modeling the scfs
upd_df <- df %>% 
  select(vars_m)
upd_df$U_UGPA <- U_ugpa
upd_df$U_LSAT <- U_lsat

write.table(upd_df, 
            file = paste(path_rslt,"sU_upd_LawData.csv", sep = ""), 
            sep = "|")

# Store var-specific weights
weights_ugpa <- {}

for (i in 1:length(sense_cols)){
  print(sense_cols[i])
  print(eta_a_ugpa[i])
  weights_ugpa[sense_cols[i]] = eta_a_ugpa[i]
}

weights_ugpa["ugpa0"] <- ugpa0
weights_ugpa["eta_u_ugpa"] <- eta_u_ugpa

temp_weights_ugpa <- as.data.frame(weights_ugpa)
weights_ugpa <- data.table::transpose(temp_weights_ugpa)
colnames(weights_ugpa) <- rownames(temp_weights_ugpa)
rownames(weights_ugpa) <- colnames(temp_weights_ugpa)
remove(temp_weights_ugpa)

write.table(weights_ugpa, 
            file = paste(path_rslt,"sU_wUGPA_LawData.csv", sep = ""), 
            sep = "|", 
            row.names = FALSE)

weights_lsat <- {}

for (i in 1:length(sense_cols)){
  print(sense_cols[i])
  print(eta_a_lsat[i])
  weights_lsat[sense_cols[i]] = eta_a_lsat[i] # exp transform?
}

weights_lsat["lsat0"] <- lsat0 # exp transform?
weights_lsat["eta_u_lsat"] <- eta_u_lsat # exp transform?

temp_weights_lsat <- as.data.frame(weights_lsat)
weights_lsat <- data.table::transpose(temp_weights_lsat)
colnames(weights_lsat) <- rownames(temp_weights_lsat)
rownames(weights_lsat) <- colnames(temp_weights_lsat)
remove(temp_weights_lsat)

write.table(weights_lsat, 
            file = paste(path_rslt,"sU_wLSAT_LawData.csv", sep = ""), 
            sep = "|", 
            row.names = FALSE)

#
# EOF
#
