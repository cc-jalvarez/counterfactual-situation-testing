# Counterfactual Situation Testing

This, as in the *master* branch, is the ongoing repository for the counterfactual situation testing framework. 

See *eeamo2023* branch for the code of ACM EAAMO 2023 paper, [Counterfactual Situation Testing: Uncovering Discrimination under Fairness given the Difference](https://dl.acm.org/doi/10.1145/3617694.3623222).

All datasets are contained within the data/ folder. The scripts get_data_< > prepare each dataset for the counterfactual situation testing pipeline. The rest of the scripts are within the src/ folder. 

The scripts get_cf_data_< > generate the counterfactual dataset for the < > case. In the case for the law school data, a second get_cf_data_ script is used for the intersectional discrimination case (denoted by _int). For this dataset in particular, we generate the counterfactual dataset in R to be able to compare the more 'frequentist' and 'Bayesian' approaches for the abduction step. The latter approach uses RStan to run a MCMC.

The scripts run_exp_< > run the experiments -- standard situation testing, counterfactual situation testing with/without centers, and counterfactual fairness -- for the < > case. 

Finally, the scripts analysis_< > contain the additional analysis (e.g., relevant figures) performed on the < > case. Results fall within the results/ folder. Different runs (in terms of parameter runs) are stored in specific folders. 
