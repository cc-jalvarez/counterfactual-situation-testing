// Stan model for Karimi2020 (v2)
data {
  int<lower = 0> N;
  vector<lower = 0>[N] asalary; // annual salary
  vector<lower = 0>[N] account; // account balance
}

parameters {
  vector[N] z;
  real lambda_account;
  real lambda_asalary;
  
  real beta_0_asalary;
  real beta_0_account;
  real beta_1;
  
  real<lower=0> sigma_g_Sq_asalary;
  real<lower=0> sigma_g_Sq_account;
}

transformed parameters  {
  
 // Population standard deviation (a positive real number)
 real<lower=0> sigma_g_asalary;
 real<lower=0> sigma_g_account;
 
 // Standard deviation (derived from variance)
 sigma_g_asalary = sqrt(sigma_g_Sq_asalary);
 sigma_g_account = sqrt(sigma_g_Sq_account);
 
}

model {
  z ~ normal(0, 1);
  lambda_asalary ~ normal(0, 1);
  lambda_account ~ normal(0, 1);
  
  asalary ~ normal(beta_0_asalary + lambda_asalary*z, sigma_g_asalary);
  
  account ~ normal(beta_0_account + beta_1*asalary + lambda_account*z, sigma_g_account);
  
}
