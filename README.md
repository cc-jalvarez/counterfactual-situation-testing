# Counterfactual Situation Testing

This is the repository for [*Counterfactual Situation Testing: Uncovering Discrimination under Fairness given the Difference* (EAAMO'23)](https://dl.acm.org/doi/10.1145/3617694.3623222). 

For the code of the latest, journal version of the paper see the *master* branch.

The datasets are in the data/ folder. The scripts get_data_< > prepare each dataset for the counterfactual situation testing pipeline. The rest of the scripts are within the src/ folder. 
The scripts get_cf_data_< > generate the counterfactual dataset for the < > case. In law school data case, a second get_cf_data_ script is used for the intersectional discrimination case (denoted by _int). 
For this dataset, we generate the counterfactual dataset in R to compare the 'frequentist' and 'Bayesian' approaches for the abduction step.
The latter approach uses RStan to run a MCMC. The scripts run_exp_< > run the experiments -- standard situation testing, counterfactual situation testing with/without centers, and counterfactual fairness -- for the < > case. 
Finally, the scripts analysis_< > contain the additional analysis (e.g., relevant figures) performed on the < > case. Results fall within the results/ folder. Different runs, parameter-wise, are stored in specific folders. 

## References

*Counterfactual Situation Testing: Uncovering Discrimination under Fairness given the Difference*. Jose M. Alvarez, and Salvatore Ruggieri. ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO), 2023.

If you make use of this code, the CST algorithm, or the generated synthetic data in your work, please cite the following paper:

<pre><code>
@inproceedings{DBLP:conf/eaamo/RuggieriA23,
  author       = {Jos{\'{e}} M. {\'{A}}lvarez and
                  Salvatore Ruggieri},
  title        = {Counterfactual Situation Testing: Uncovering Discrimination under
                  Fairness given the Difference},
  booktitle    = {{EAAMO}},
  pages        = {2:1--2:11},
  publisher    = {{ACM}},
  year         = {2023}
}
</code></pre>
