---
title: '`rjaf`: Regularized Joint Assignment Forest with Treatment Arm Clustering'
tags:
- machine learning
- causal inference
- "multi-arm randomized controlled trial"
- heterogeneous treatment effects
- personalized treatment rules
- optimal assignment
- R
- C++
date: "November 27, 2024"
output:
  pdf_document: default
  html_document:
    df_print: paged
authors:
- name: Wenbo Wu
  orcid: "0000-0002-7642-9773"
  corresponding: true
  affiliation: 1
- name: Xinyi Zhang
  orcid: "0009-0007-7306-491X"
  affiliation: 1
- name: Jann Spiess
  orcid: "0000-0002-4120-8241"
  affiliation: 2
- name: Rahul Ladhania
  orcid: "0000-0002-7902-7681"
  affiliation: 3
bibliography: paper.bib
affiliations:
- name: Departments of Population Health and Medicine, NYU Grossman School of Medicine, USA
  index: 1
- name: Graduate School of Business and Department of Economics, Stanford University, USA
  index: 2
- name: Departments of Health Management and Policy and Biostatistics, University of Michigan School of Public Health, USA
  index: 3
---

# Summary

Learning optimal assignment of treatments is an important problem in economics and public health, specifically when there are many treatment strategies to choose from. It arises, for example, in settings where a variety of behavioral science informed interventions are tested in a randomized controlled trial (RCT) to identify ones that are most effective at behavior change [@milkman2021megastudiesnature]. This has been studied in domains as diverse as encouraging gym visits to vaccine uptake against the coronavirus disease 2019 (COVID-19) or influenza [@milkman2021megastudiesnature; @milkman2021megastudy; @dai2021behavioural; @milkman2022680]. Most studies focus on identifying interventions which have (on average) the best performance, but ignoring effect heterogeneity can be a missed opportunity or, worse, perpetuate disparities [@bryan2021behavioural]. Subject-specific covariates/features containing information regarding sociodemographics, past behavior, clinical characteristics, and comorbidities, if available, can be harnessed to identify which interventions works best for different sub-groups in the data. The `rjaf` package provides a user-friendly implementation of the regularized joint assignment forest (RJAF) [@ladhania2023personalized], a regularized forest-type assignment algorithm based on greedy recursive partitioning [@athey2019generalized] that shrinks effect estimates across treatment arms. The algorithm is augmented by outcome residualization (to reduce baseline variation) and a clustering scheme [@hartigan1979algorithm] that combines treatment arms with consistently similar outcomes. Personalized treatment learning is achieved through optimizing a regularized empirical analogue of the expected outcome. The integration of `R` [@R2024] and `C++` [@stroustrup2013c] substantially boosts the computational efficiency in tree partitioning and aggregating. This package is especially suitable in RCT settings where a large number of treatment arms are present, with potential overlap among the arms, and constrained sample sizes.

# Statement of Need

There is an ever-growing literature at the intersection of machine learning and causal inference attempting to address the problem of optimal treatment assignment through heterogeneous treatment effect estimation [@athey2016recursive; @wager2018estimation; @hitsch2018heterogeneous; @athey2019generalized; @sverdrup2020policytree; @athey2021policy]. Other methods focus on maximizing the benefit (empirical welfare) from treatment assignment [e.g., @kitagawa2018should], or the chance of assigning an individual to an optimal treatment arm [e.g., @JMLR:v6:murphy05a; @zhou2018sequential]. Most of these methods perform well with a limited number of treatment and control groups. A large number of arms renders the identification of best arm increasingly difficult, and assignments based on separate arm-wise estimation are inefficient. Commonly used implementations such as the multi-arm causal forest [@tibshirani2020grf] and random forest [@JSSv077i01] might lead to suboptimal assignment, particularly in settings with high noise. By contrast, the RJAF yields elevated empirical outcome estimates closer to the optimal level from the oracle assignment than the multi-arm causal forest approach in high noise settings, and performs at the same level in low-noise ones. Despite the methodological advantage over existing approaches, the incorporation of machine learning and causal inference techniques such as recursive tree partitioning, bootstrap aggregating, and treatment arm clustering makes it challenging to implement the RJAF from scratch even for well-trained data scientists. The `rjaf` is an open-source software package in `R` and `C++` that efficiently implements the RJAF, offering data scientists a user-friendly analytic toolbox for learning personalized treatment rules in real-world settings.

# Workflow

\autoref{fig:pkg} outlines the workflow for using the `rjaf` package to perform personalized treatment assignment and honest outcome estimation. The process begins with partitioning the input data--consisting of outcomes, treatment arms, covariates, individual identifiers, and optional probabilities of treatment assignment--into two parts, one for model training and estimation, and the other is the heldout set on which personalized assignment rules are obtained. The `rjaf` function first checks whether outcome residualization for reducing baseline variation should be performed via the `residualize` function, using the `resid` argument. If `resid` is set to `TRUE` (the default), a new column of residualized outcomes is added to the input data and used for tree growing on the training set. Next, the `rjaf` function evaluates whether treatment clustering should be performed on the training-estimation set during tree growing using the `clus.tree.growing` argument. If `clus.tree.growing` is `TRUE`, the `rjaf_cpp` function is employed to estimate cross-validated counterfactual outcomes for the $K+1$ treatment arms, after which k-means clustering is used to learn $M+1$ treatment arm clusters. The optimal number of treatment clusters is determined using the elbow method. After clustering, the `rjaf_cpp` function is reapplied to the preprocessed data, with assignment forest fitted on $M+1$ treatment clusters and counterfactual outcomes estimated for the original $K+1$ arms. If `clus.tree.growing` is `FALSE`, the `rjaf_cpp` function is employed to estimate counterfactual outcomes for the $K+1$ arms. Lastly, `rjaf_cpp` function is used to obtain optimal treatment arms and predicted counterfactual outcomes under all treatment arms for individuals in the heldout set.

![A sketch of the `rjaf` package. \label{fig:pkg}](pkg_sketch.pdf){height=95%}

\autoref{fig:rjaf_cpp} provides a detailed description of the `rjaf_cpp` function, which grows a large number of trees through the `growTree` function. `growTree` begins by taking the training and estimation set as input, splitting it into two separate training and estimation subsets proportionally by treatment arms or clusters. Initially, utility is set at the root node, where optional inverse probability weighting (IPW) can be applied. A tree is then grown via recursive partitioning of the training subset based on covariate splits. Each potential split is generated by the `splitting` function, where regularization tuned by the `lambda1` parameter can be performed along with IPW to calculate average outcomes across treatment arms or clusters. A potential split is retained if it meets three criteria: (1) each child node contains at least the minimum number of units specified by the `nodesize` argument, (2) the utility gain is at least `eps` times the empirical standard deviation of outcomes in the entire input data, and (3) the child nodes have different optimal treatment arm or cluster assignments. Recursive partitioning ends when no potential split meets these criteria. Once terminal nodes are determined on the training subset, the same splitting rules applied to the estimation subset to obtain its terminal nodes. Outcomes from the estimation subset are then used to calculate treatment-specific average outcomes in each terminal node, with optional regularization tuned by `lambda2` and imputation controlled by `impute`. The validation set undergoes the same splitting, with treatment-specific outcomes from the estimation subset assigned to corresponding terminal nodes to achieve honest outcome estimates, thus concluding the `growTree` function. The final step in `rjaf_cpp` is bootstrap aggregation of a large number of trees grown via `growTree`, where the total number of trees is set by the `ntree` parameter of the `rjaf_cpp` function.

As in the implementations of other forest-based methods, built-in hyper-parameter tuning (e.g., `eps`, `lambda1`, and `lambda2`) is not provided in the `rjaf` package. Interested users are referred to the `caret` package [@kuhn2008building] for details.

![A description of the `rjaf_cpp` function. \label{fig:rjaf_cpp}](rjaf_cpp.pdf){width=100%}

# Quick Start

The `rjaf` package is publicly available on [GitHub](https://github.com/wustat/rjaf), where the use of the `rjaf` package, including installation instructions and an example, has been documented in the `README.md` file. The package is also available on the [Comprehensive R Archive Network](https://CRAN.R-project.org/package=rjaf). This section provides an introduction to the basics of `rjaf`. One can install and load the `rjaf` package by executing the following `R` commands:

```r
install.packages('rjaf')
library(rjaf)
```

# Example

Next we present an example that illustates the usage of the package. A function `sim.data` used for simulating a synthetic data set with three covariates is provided below, where `n` indicates the sample size, `K`+1 represents the number of treatment arms, `gamma` denotes the strength of treatment effects, `sigma` is the noise level, and `probability` is a vector of sampling probabilities of treatment arms. This function depends on two widely used `R` packages `MASS` and `dplyr`. The output of the `sim.data` function is a data frame, containing a column (named `id`) of individual IDs, a column (named `Y`) of outcomes, three columns of covariates `X1`, `X2`, and `X3`, a column (named `trt`) of treatment arms, and a column (named `prob`) of probabilities of treatment assignment. Readers are referred to @ladhania2023personalized [Section 4.1] for more details about the simulation setup.

```r
sim.data <- function(n, K, gamma, sigma, probability = rep(1,K+1)/(K+1)) {
  options(stringsAsFactors=FALSE)
  data <- left_join(data.frame(id=1:n,
                               trt=sample(0:K, n, replace=TRUE, probability),
                               mvrnorm(n, rep(0,3), diag(3))),
                    data.frame(trt=0:K, prob=probability), by="trt")
  data <- mutate(data, tmp1=10+20*(X1>0)-20*(X2>0)-40*(X1>0&X2>0),
                 tmp2=gamma*(2*(X3>0)-1)/(K-1),
                 tmp3=-10*X1^2,
                 Y=tmp1+tmp2*(trt>0)*(2*trt-K-1)+tmp3*(trt==0)+rnorm(n,0,sigma))
  
  Y.cf <- data.frame(sapply(0:K, function(t)
    mutate(data, Y=tmp1+tmp2*(t>0)*(2*t-K-1)+tmp3*(t==0))$Y))
  names(Y.cf) <- paste0("Y",0:K)
  return(mutate(bind_cols(dplyr::select(data, -c(tmp1,tmp2,tmp3)), Y.cf),
                across(c(id, trt), as.character)))
}
```

Using the `sim.data` function, we generate two data sets: one for training and estimation, and the other held out for validation, both of which have a sample size of 5000. The number of treatment arms is set to 30 (`K`=29), with a treatment effect strength of `gamma`=10 and a noise level of `sigma`=20. Treatment arms are uniformly assigned to all individuals across both data sets. Calling the `rjaf` function returns a tidyverse tibble [@muller2023] containing individual IDs, optimal treatment arms, counterfactual outcomes, predicted outcomes, and treatment arm clusters.

```r
library(MASS)
library(dplyr)
K <- 29; gamma <- 10; sigma <- 20; probability <- rep(1,K+1)/(K+1)
n.heldout <- n.trainest <- 5000
data.trainest <- sim.data(n.trainest, K, gamma, sigma, probability)
data.heldout <- sim.data(n.heldout, K, gamma, sigma, probability)
fit <- rjaf(data.trainest, data.heldout, y = "Y", id = "id", trt = "trt", 
            vars = paste0("X", 1:3), prob = "prob", ntrt = K+1, nvar = 3,
            lambda1 = 0, lambda2 = 0.5, nodesize = 3, eps = 0.5, reg = TRUE,
            impute = FALSE, clus.tree.growing = TRUE, clus.max = 5)
head(fit)
# A tibble: 6 × 5
  id    trt.rjaf  Y.cf Y.rjaf clus.rjaf
  <chr> <chr>    <dbl>  <dbl>     <int>
1 1     29        20     11.6         1
2 2     29        40     11.5         1
3 3     3         18.6   10.3         3
4 4     3         38.6   10.4         3
5 5     3         18.6   10.4         3
6 6     3        -21.4   10.3         3
```

To demonstrate the advantage of treatment arm clustering, we conducted a series of simulated data experiments following the above setup. In each experiment, 500 simulated Training-Estimation sets were generated, while the same heldout data set was used for validation across all Training-Estimation sets. The number of treatment arms was set to 10, 30, 50, and 100, respectively. \autoref{fig:rjaf_clustering} displays boxplots of 500 simulations comparing the average outcome of the heldout set from unclustered and clustered RJAF. "Oracle Optimal Assignment" denotes the assignment strategy derived from the ground truth in simulations, ensuring the best possible outcomes. "Random Assignment" involves randomly distributing units across treatment arms in each simulation. "Global Best Assignment" refers to assigning all units to the treatment in a simulation that demonstrates the highest average performance. Across all settings, the clustered RJAF is associated with a higher average outcome than the unclustered RJAF; the RJAF (both clustered and unclustered) consistently outperforms "Random Assignment" and "Global Best Assignment," while closely approximating the performance of the "Oracle Optimal Assignment."

![Boxplots of 500 simulations comparing the average outcome of the heldout set from unclustered and clustered RJAF. "Oracle Optimal Assignment" denotes the assignment strategy derived from the ground truth in simulations, ensuring the best possible outcomes. "Random Assignment" involves randomly distributing units across treatment arms in each simulation. "Global Best Assignment" refers to assigning all units to the treatment in a simulation that demonstrates the highest average performance. \label{fig:rjaf_clustering}](rjaf_trt_clustering.png){width=100%}

# Acknowledgments

Wenbo Wu and Xinyi Zhang contributed equally to the project. Wenbo Wu and Rahul Ladhania were partially supported by a research grant from the Robert Wood Johnson Foundation titled *Informing Strategies to Increase Use of COVID-19 and Flu Vaccines by Different Racial and Ethnic Groups to Improve Health Equity during Health Crises* (award number 78416).

# References

