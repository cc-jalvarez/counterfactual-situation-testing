# counterfactual-situation-testing

Code for the working paper on Uncovering Discrimination under Fairness given the Difference [[arXiv]](https://arxiv.org/abs/2302.11944). We stress that CST, above an implementation, presents a pipeline for testing individual discrimination claims based on ``fairness given the difference'', meaning the elements presented here (e.g., linear models, kNN search, etc.) are exchangeable. What matters is that the individual test group is derived using the (structural) counterfactual of the (observed) factual. 

Datasets are contained within the data/ folder. The scripts get_data_< > (also within the data/ folder) prepare each dataset for the counterfactual situation testing pipeline. The rest of the scripts are within the src/ folder. 

The scripts get_cf_data_< > generate the counterfactual dataset for the < > case. In the case for the law school (real) data, a second get_cf_data_ script is used for the intersectional discrimination case (denoted by _int). For this dataset in particular, we generate the counterfactual dataset in R to be able to compare the more 'frequentist' and 'Bayesian' approaches for the abduction step. The latter approach uses RStan to run a MCMC.

The scripts run_exp_< > run the experiments -- standard situation testing, counterfactual situation testing with/without centers, and counterfactual fairness -- for the < > case. 

Finally, the scripts analysis_< > contain the additional analysis (e.g., relevant figures) performed on the < > case. Results fall within the results/ folder. Different runs (in terms of parameter runs) are stored in specific folders. 
