// Stan model for Karimi2020 (v1)
data {
  int<lower = 0> N;
  vector<lower = 0>[N] asalary; // annual salary
  vector<lower = 0>[N] account; // account balance
  real<lower = 0> mu_account;
  real<lower = 0> si_account;
}

parameters {
  vector[N] u_account;
  real lambda_account;
  real<lower=0> sigma_account;
  real beta_1;
  real intercept;
}

model {
  u_account ~ normal(0, 1);
/*  u_account ~ normal(mu_account, si_account);*/
  
  lambda_account ~ normal(0, 1);
  account ~ normal(beta_1*asalary + lambda_account*u_account + intercept, sigma_account);
  
/*  account ~ normal(beta_1*asalary + u_account + intercept, sigma_account);*/

}
