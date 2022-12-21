# This script gets the structural counterfactual (CF) datasets possible under 
# the law school data for the intersectional setting. We do so for Level 3 only

# Seed
set.seed(42)

# Packages
library(dplyr)
library(caret)
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

# Train model for descendant nodes 
model_ugpa_opt1 <- 
  lm(UGPA ~ female + nonwhite + female_nonwhite + 1, data=df_lev3)

model_lsat_opt1 <- 
  lm(LSAT ~ female + nonwhite + female_nonwhite + 1, data=df_lev3)

# Option 2: the machine learning form -------------------------------------

# Train model for descendant nodes 
model_ugpa_opt2 <- 
  lm(UGPA ~ female_nonwhite + 1, data=df_lev3)

model_lsat_opt2 <- 
  lm(LSAT ~ female_nonwhite + 1, data=df_lev3)

summary(model_ugpa_opt1)
summary(model_ugpa_opt2)

hist(
  df_lev3$UGPA - predict.glm(model_ugpa_opt1, newdata=df_lev3)
)
summary(
  df_lev3$UGPA - predict.glm(model_ugpa_opt1, newdata=df_lev3)
)
hist(
  df_lev3$UGPA - predict.glm(model_ugpa_opt2, newdata=df_lev3)
)
summary(
  df_lev3$UGPA - predict.glm(model_ugpa_opt2, newdata=df_lev3)
)

summary(model_lsat_opt1)
summary(model_lsat_opt2)

hist(
  df_lev3$LSAT - predict.glm(model_lsat_opt1, newdata=df_lev3)
)
summary(
  df_lev3$LSAT - predict.glm(model_lsat_opt1, newdata=df_lev3)
)
hist(
  df_lev3$LSAT - predict.glm(model_lsat_opt2, newdata=df_lev3)
)
summary(
  df_lev3$LSAT - predict.glm(model_lsat_opt2, newdata=df_lev3)
)

# Notes: the main difference between these options is that the coefficients are
# broken down in opt 1. Since, to be, say, female-nonwhite the individual must
# also be female and nonwhite (constituent relationship) performing the do
# on (female-nonwhite, female, nonwhite) vs (female-nonwhite) should have the 
# same overall effect on UGPA and LSAT. Therefore, we focus on opt 2 for now.
# Notice that this is further shown by looking at the distribution of the 
# residuals for each attribute-option.

# Step 1: perform the abduction step by estimating the residuals
df_lev3$resid_UGPA = df_lev3$UGPA - 
  predict.glm(model_ugpa_opt2, newdata=df_lev3)
df_lev3$resid_LSAT = df_lev3$LSAT - 
  predict.glm(model_lsat_opt2, newdata=df_lev3)

# Step 2: action on race and gender (accordingly: under multiple disc.)
# do(Gender:='Male-White')
df_lev3_do_male_white <- data.frame(female_nonwhite=rep(0, nrow(df_lev3)))

# Step 3: prediction
df_lev3_do_male_white$GenderRace <- paste(df_lev3$sex, df_lev3$race_nonwhite, sep='-')
df_lev3_do_male_white$resid_LSAT <- df_lev3$resid_LSAT
df_lev3_do_male_white$resid_UGPA <- df_lev3$resid_UGPA

df_lev3_do_male_white$scf_LSAT <- 
  round(
    predict.glm(model_lsat_opt2, newdata=df_lev3_do_male_white) + 
      df_lev3_do_male_white$resid_LSAT, 3
    )

df_lev3_do_male_white$scf_UGPA <- 
  round(
    predict.glm(model_ugpa_opt2, newdata=df_lev3_do_male_white) + 
      df_lev3_do_male_white$resid_UGPA, 3
    )

summary(df_lev3_do_male_white$scf_LSAT) # btw 10 - 48
summary(df_lev3_do_male_white$scf_UGPA) # btw 0 - 4

df_lev3_do_male_white$scf_LSAT <- 
  ifelse(df_lev3_do_male_white$scf_LSAT > 48.00, 48.00, df_lev3_do_male_white$scf_LSAT)
df_lev3_do_male_white$scf_LSAT <- 
  ifelse(df_lev3_do_male_white$scf_LSAT < 10.00, 10.00, df_lev3_do_male_white$scf_LSAT)
summary(df_lev3_do_male_white$scf_LSAT)

df_lev3_do_male_white$scf_UGPA <- 
  ifelse(df_lev3_do_male_white$scf_UGPA > 4.00, 4.00, df_lev3_do_male_white$scf_UGPA)
df_lev3_do_male_white$scf_UGPA <- 
  ifelse(df_lev3_do_male_white$scf_UGPA < 0.00, 0.00, df_lev3_do_male_white$scf_UGPA)
summary(df_lev3_do_male_white$scf_UGPA)

write.table(df_lev3_do_male_white, 
            file = paste(path_rslt, "cf_LawSchool_lev3_doMaleWhite.csv", sep = ""), 
            sep = "|")

#
# EOF
#
