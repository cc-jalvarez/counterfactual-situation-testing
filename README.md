# counterfactual-situation-testing

Code for ``Counterfactual Situation Testing: Uncovering Discrimination under Fairness given the Difference." *ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO)*, pages 1-11, 2023 [[paper]](https://dl.acm.org/doi/10.1145/3617694.3623222). 

See *eeamo2023* branch for specific code to the conference paper; new experiments for the journal version will start soon.

All datasets are contained within the data/ folder. The scripts get_data_< > (also within the data/ folder) prepare each dataset for the counterfactual situation testing pipeline. The rest of the scripts are within the src/ folder. 

The scripts get_cf_data_< > generate the counterfactual dataset for the < > case. In the case for the law school (real) data, a second get_cf_data_ script is used for the intersectional discrimination case (denoted by _int). For this dataset in particular, we generate the counterfactual dataset in R to be able to compare the more 'frequentist' and 'Bayesian' approaches for the abduction step. The latter approach uses RStan to run a MCMC.

The scripts run_exp_< > run the experiments -- standard situation testing, counterfactual situation testing with/without centers, and counterfactual fairness -- for the < > case. 

Finally, the scripts analysis_< > contain the additional analysis (e.g., relevant figures) performed on the < > case. Results fall within the results/ folder. Different runs (in terms of parameter runs) are stored in specific folders. 

The folder scm_models contains an abstract class for structural causal models as well as an example class for the loan application scenario. Ignore this folder as it is not used in the current implementation.
