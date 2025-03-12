# Counterfactual Situation Testing

This is the repository for [*Counterfactual Situation Testing: From Single to Multidimensional Discrimination*](https://arxiv.org/abs/2502.01267), forthcoming in [The Journal of Artificial Intelligence](https://www.jair.org/index.php/jair/SpecialTrack-FBAI). The datasets are located in the data/ folder. The scripts get_data_< > prepare each < > dataset. All other scripts are in the src/ folder. The scripts get_cf_data_< > generate the counterfactual dataset. In the case for the law school data, a second get_cf_data_ script is used for the intersectional discrimination case. For this dataset, in particular, we generate the counterfactual dataset in R also to be able to compare the 'frequentist' and 'Bayesian' approaches of the abduction step. The latter approach uses RStan. The scripts run_exp_< > run the experiments. Finally, the scripts analysis_< > contain additional analysis. The results, including relevant figures, are in the results/ folder.

## The EAAMO'23 version

An earlier version of this work appeared, [*Counterfactual Situation Testing: Uncovering Discrimination under Fairness given the Difference*](https://dl.acm.org/doi/10.1145/3617694.3623222), was published at EAAMO 2023. See the *eeamo2023* branch for the code used in that version of the paper. It has changed for the journal version: please use the *master* branch for the latest.

## References

*Counterfactual Situation Testing: Uncovering Discrimination under Fairness given the Difference*. Jose M. Alvarez, and Salvatore Ruggieri. ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO), 2023.

*Counterfactual Situation Testing: From Single to Multidimensional Discrimination*. Jose M. Alvarez, and Salvatore Ruggieri. Forthcoming in Journal of Artificial Intelligence Research (JAIR), 2025.

If you make use of this code, the CST algorithm, or the generated synthetic data in your work, please cite the following paper:

<pre><code>
@article{DBLP:journals/corr/abs-2502-01267,
  author       = {Jos{\'{e}} M. {\'{A}}lvarez and
                  Salvatore Ruggieri},
  title        = {Counterfactual Situation Testing: From Single to Multidimensional
                  Discrimination},
  journal      = {CoRR},
  volume       = {abs/2502.01267},
  year         = {2025}
}
</code></pre>
