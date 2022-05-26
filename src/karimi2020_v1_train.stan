// Stan model for Karimi2020 (v1)
data {
  int<lower = 0> N; // number of individuals
  vector<lower = 0>[N] asalary; // annual salary
  vector<lower = 0>[N] account; // account balance
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
  
  lambda_account ~ normal(0, 1);
  
  account ~ normal(beta_1*asalary + lambda_account*u_account + intercept, sigma_account);

}
