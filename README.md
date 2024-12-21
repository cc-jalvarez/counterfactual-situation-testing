# Counterfactual Situation Testing

The *master* branch is the ongoing repository for CST. For the code of the [EAAMO 2023 paper](https://dl.acm.org/doi/10.1145/3617694.3623222), see the *eeamo2023* branch.

The datasets are in the data/ folder. The scripts get_data_< > prepare each < > dataset. All other scripts are in the src/ folder. The scripts get_cf_data_< > generate the counterfactual dataset. In the case for the law school data, a second get_cf_data_ script is used for the intersectional discrimination case. For this dataset, in particular, we generate the counterfactual dataset in R also to be able to compare the 'frequentist' and 'Bayesian' approaches of the abduction step. The latter approach uses RStan. The scripts run_exp_< > run the experiments. Finally, the scripts analysis_< > contain additional analysis. The results, including relevant figures, are in the results/ folder.
