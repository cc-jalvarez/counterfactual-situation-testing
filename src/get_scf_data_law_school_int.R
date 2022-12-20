# Seed
set.seed(42)

# Packages
library(dplyr)
library(caret)
# library(rstan)
# rstan_options(auto_write = TRUE)
# options(mc.cores = parallel::detectCores())
library(data.table)


# Setup -------------------------------------------------------------------

# working directory
dir = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir)
setwd('..')
wd = getwd()

# folder paths
path_data = paste(wd, "/data/", sep = "")
path_mdls = paste(wd, "/src/stan_models/", sep = "")
path_rslt = paste(wd, "/data/counterfactuals/", sep = "")

# original data
org_df <- read.csv(file = paste(path_data, "clean_LawSchool.csv", sep = ""), sep = '|')

# initial vars
use_race = "race_nonwhite"
vars <- c("LSAT", "UGPA", "sex")
vars <- append(vars, use_race)
# # modified vars for scf generation
# vars_m <- c("LSAT", "UGPA")

# modeling data
df <- org_df %>% select(vars)

# var transformation
df$LSAT <- round(df$LSAT)

sense_cols <- c("female", "male")
df$female <- as.numeric(df$sex == "Female")
df$male <- as.numeric(df$sex == "Male")

if (use_race == "race_nonwhite"){
  df$white <- as.numeric(df$race_nonwhite == "White")
  sense_cols <- append(sense_cols, "white")
  df$nonwhite <- as.numeric(df$race_nonwhite == "NonWhite")
  sense_cols <- append(sense_cols, "nonwhite")
  table(df$race_nonwhite)
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
  table(df$race_simpler)
}

print(sense_cols)
# vars_m <- append(vars_m, sense_cols)


# Level 3: Intersectional Discrimination ----------------------------------

# DAG: Sex -> UGPA; Race -> UGPA; Sex -> LSAT; Race -> LSAT + interaction term
df_lev3 <- df
df_lev3$female_nonwhite <- df_lev3$female*df_lev3$nonwhite

# Option 1: the econometrics form  ----------------------------------------

# Step 1: train model for descendant nodes, and 
model_ugpa_opt1 <- 
  lm(UGPA ~ female + nonwhite + female_nonwhite + 1, data=df_lev3)

model_lsat_opt1 <- 
  lm(LSAT ~ female + nonwhite + female_nonwhite + 1, data=df_lev3)

# Option 2: the machine learning form -------------------------------------

# Step 1: train model for descendant nodes, and 
model_ugpa_opt2 <- 
  lm(UGPA ~ female_nonwhite + 1, data=df_lev3)

model_lsat_opt2 <- 
  lm(LSAT ~ female_nonwhite + 1, data=df_lev3)

summary(model_ugpa_opt1)
summary(model_ugpa_opt2)

summary(model_lsat_opt1)
summary(model_lsat_opt2)


# perform the abduction step: estimate the residuals
df_lev3$resid_UGPA = df_lev3$UGPA - predict.glm(model_ugpa, newdata=df_lev3)
hist(df_lev3$resid_UGPA)

df_lev3$resid_LSAT = df_lev3$LSAT - predict.glm(model_lsat, newdata=df_lev3)
hist(df_lev3$resid_LSAT)

# Step 2: action on race and gender (accordingly: under multiple disc.)
# do(Gender:='Male')
df_lev3_do_male <- data.frame(female=rep(0, nrow(df_lev3)), 
                              nonwhite=df_lev3$nonwhite)

# do(Race:='White')
df_lev3_do_white <- data.frame(female=df_lev3$female, 
                               nonwhite=rep(0, nrow(df_lev3)))
# Step 3: prediction
# do(Gender:='Male')
df_lev3_do_male$Sex <- df_lev3$sex
df_lev3_do_male$Race <- df_lev3$race_nonwhite
df_lev3_do_male$resid_LSAT <- df_lev3$resid_LSAT
df_lev3_do_male$resid_UGPA <- df_lev3$resid_UGPA

df_lev3_do_male$scf_LSAT <- round(predict(model_lsat, newdata=df_lev3_do_male) 
                                  + df_lev3_do_male$resid_LSAT, 3)
df_lev3_do_male$scf_UGPA <- round(predict(model_ugpa, newdata=df_lev3_do_male) 
                                  + df_lev3_do_male$resid_UGPA, 3)

summary(df_lev3_do_male$scf_LSAT) # btw 10 - 48
summary(df_lev3_do_male$scf_UGPA) # btw 0 - 4

df_lev3_do_male$scf_LSAT <- 
  ifelse(df_lev3_do_male$scf_LSAT > 48.00, 48.00, df_lev3_do_male$scf_LSAT)
df_lev3_do_male$scf_LSAT <- 
  ifelse(df_lev3_do_male$scf_LSAT < 10.00, 10.00, df_lev3_do_male$scf_LSAT)
summary(df_lev3_do_male$scf_LSAT)

df_lev3_do_male$scf_UGPA <- 
  ifelse(df_lev3_do_male$scf_UGPA > 4.00, 4.00, df_lev3_do_male$scf_UGPA)
df_lev3_do_male$scf_UGPA <- 
  ifelse(df_lev3_do_male$scf_UGPA < 0.00, 0.00, df_lev3_do_male$scf_UGPA)
summary(df_lev3_do_male$scf_UGPA)

write.table(df_lev3_do_male, 
            file = paste(path_rslt, "cf_LawSchool_lev3_doMale.csv", sep = ""), 
            sep = "|")

# do(Race:='White')
df_lev3_do_white$Sex <- df_lev3$sex
df_lev3_do_white$Race <- df_lev3$race_nonwhite
df_lev3_do_white$resid_LSAT <- df_lev3$resid_LSAT
df_lev3_do_white$resid_UGPA <- df_lev3$resid_UGPA

df_lev3_do_white$scf_LSAT <- round(predict(model_lsat, newdata=df_lev3_do_white) 
                                   + df_lev3_do_white$resid_LSAT, 3)
df_lev3_do_white$scf_UGPA <- round(predict(model_ugpa, newdata=df_lev3_do_white) 
                                   + df_lev3_do_white$resid_UGPA, 3)

summary(df_lev3_do_white$scf_LSAT) # btw 10 - 48
summary(df_lev3_do_white$scf_UGPA) # btw 120 - 180

df_lev3_do_white$scf_LSAT <- 
  ifelse(df_lev3_do_white$scf_LSAT > 48.00, 48.00, df_lev3_do_white$scf_LSAT)
df_lev3_do_white$scf_LSAT <- 
  ifelse(df_lev3_do_white$scf_LSAT < 10.00, 10.00, df_lev3_do_white$scf_LSAT)
summary(df_lev3_do_white$scf_LSAT)

df_lev3_do_white$scf_UGPA <- 
  ifelse(df_lev3_do_white$scf_UGPA > 4.00, 4.00, df_lev3_do_white$scf_UGPA)
df_lev3_do_white$scf_UGPA <- 
  ifelse(df_lev3_do_white$scf_UGPA < 0.00, 0.00, df_lev3_do_white$scf_UGPA)
summary(df_lev3_do_white$scf_UGPA)

write.table(df_lev3_do_white, 
            file = paste(path_rslt, "cf_LawSchool_lev3_doWhite.csv", sep = ""), 
            sep = "|")

#
# EOF
#
