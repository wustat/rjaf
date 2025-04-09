---
title: 'rjaf: Regularized Joint Assignment Forest with Treatment Arm Clustering'
tags:
- machine learning
- causal inference
- multi-arm randomized controlled trial
- heterogeneous treatment effects
- personalized treatment rules
- optimal assignment
- R
- C++
date: "March 22, 2025"
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
- name: Department of Population Health, NYU Grossman School of Medicine, USA
  index: 1
- name: Graduate School of Business, Stanford University, USA
  index: 2
- name: Departments of Health Management and Policy and Biostatistics, University of Michigan School of Public Health, USA
  index: 3
---

# Summary

Learning optimal assignment of treatments is an important problem in economics, public health, and related fields, particularly when faced with a variety of treatment strategies. The problem arises, for example, in settings where randomized controlled trials (RCT) are conducted to evaluate various behavioral science-informed interventions aimed at fostering behavior change [@milkman2021megastudiesnature]. Such interventions have been studied across diverse domains, including encouraging gym attendance and increasing vaccine uptake for influenza or COVID-19 [@milkman2021megastudiesnature; @milkman2021megastudy; @dai2021behavioural; @milkman2022680]. While most studies focus on identifying interventions that perform best on average, this approach often overlooks effect heterogeneity. Ignoring heterogeneity can be a missed opportunity to tailor interventions for maximum effectiveness and may even exacerbate disparities [@bryan2021behavioural]. Subject-specific covariates, such as sociodemographics can be harnessed to identify which interventions work best for different segments of the population, allowing for more impactful intervention assignments. The `rjaf` package provides a user-friendly implementation of the regularized joint assignment forest (RJAF) [@ladhania2023personalized], a regularized forest-type assignment algorithm based on greedy recursive partitioning [@athey2019generalized] that shrinks effect estimates across treatment arms. The algorithm is augmented by outcome residualization to reduce baseline variation, and employs a clustering scheme [@hartigan1979algorithm] that combines treatment arms with consistently similar outcomes. Personalized treatment learning is achieved by optimizing a regularized empirical analogue of the expected outcome. The integration of `R` [@R2024] and `C++` [@stroustrup2013c] substantially boosts computational efficiency in tree partitioning and aggregating. It is especially suitable in RCT settings with numerous treatment arms and constrained sample sizes, making it a powerful tool for learning personalized intervention strategies.

# Statement of Need

There is an ever-growing literature at the intersection of machine learning and causal inference attempting to address the problem of optimal treatment assignment through heterogeneous treatment effect estimation [@zhao2012estimating; @athey2016recursive; @zhou2017residual; @wager2018estimation; @hitsch2018heterogeneous; @athey2019generalized; @sverdrup2020policytree; @athey2021policy]. Among them, the `policytree` approach [@sverdrup2020policytree] pursues doubly robust estimation that relies on the `grf` package, while the outcome (or residual) weighted learning approach [@zhao2012estimating; @zhou2017residual] is based on support vector machines. Other methods focus on maximizing the benefit (empirical welfare) from treatment assignment [e.g., @kitagawa2018should], or the chance of assigning an individual to an optimal treatment arm [e.g., @JMLR:v6:murphy05a; @zhou2018sequential]. Most of these approaches perform well with a limited number of treatment and control groups. A large number of arms renders the identification of best arm increasingly difficult, and assignments based on separate arm-wise estimation are inefficient. Commonly used implementations such as the multi-arm causal forest (implemented through the `grf` package) [@tibshirani2020grf] and random forest [@JSSv077i01] might lead to suboptimal assignment, particularly in settings with high noise. By contrast, the RJAF [@ladhania2023personalized] yields elevated empirical outcome estimates closer to the optimal level from the oracle assignment than the multi-arm causal forest approach in high noise settings, and performs at the same level in low-noise ones. Despite the methodological advantage over existing approaches, the incorporation of machine learning and causal inference techniques such as recursive tree partitioning, bootstrap aggregating, and treatment arm clustering makes it challenging to implement the RJAF from scratch even for well-trained data scientists. The `rjaf` is an open-source software package in `R` and `C++` that efficiently implements the RJAF, offering data scientists a user-friendly analytic toolbox for learning personalized treatment rules in real-world settings.

# Workflow

\autoref{fig:pkg} outlines the workflow of the `rjaf` package. The process begins with partitioning the input data---consisting of outcomes, treatment arms, covariates, individual identifiers, and optional probabilities of treatment assignment---into two parts, one for model training and estimation, and the other is the held-out set on which personalized assignment rules are obtained. The `rjaf` function first checks whether outcome residualization for reducing baseline variation should be performed via the `residualize` function, using the `resid` argument. If `resid` is set to `TRUE` (the default), a new column of residualized outcomes is added to the input data and used for tree growing on the training set. Next, the `rjaf` function evaluates whether treatment clustering should be performed on the training-estimation set during tree growing using the `clus.tree.growing` argument. If `clus.tree.growing` is `TRUE`, an `Rcpp` function is employed to estimate cross-validated counterfactual outcomes for the $K+1$ treatment arms, after which k-means clustering is used to learn $M+1$ treatment arm clusters. The optimal number of treatment clusters is determined using the elbow method. After clustering, the `Rcpp` function is reapplied to the preprocessed data, with assignment forest fitted on $M+1$ treatment clusters and counterfactual outcomes estimated for the original $K+1$ arms. If `clus.tree.growing` is `FALSE`, the `Rcpp` function is employed to estimate counterfactual outcomes for the $K+1$ arms. Lastly, the `Rcpp` function is used to obtain optimal treatment arms and predicted counterfactual outcomes under all treatment arms for individuals in the held-out set.

![A sketch of the `rjaf` package. \label{fig:pkg}](pkg_sketch.pdf){height=95%}

\autoref{fig:rjaf} describes the RJAF algorithm. Tree growing begins by taking the training-estimation data set as input, randomly splitting it into separate training and estimation subsets proportionally by treatment arms (or clusters). Initially, utility is set at the root node, where optional Horvitz--Thompson estimator-based inverse probability weighting (IPW) can be applied. A tree is then grown via recursive partitioning of the training subset based on covariate splits. Each potential split is generated by an internal `Rcpp` function, where regularization specified by the `lambda1` parameter can be performed along with IPW to calculate weighted average outcomes by treatment arms or clusters. A potential split is retained if it meets three criteria: (1) each child node contains at least the minimum number of units specified by the `nodesize` argument, (2) the utility gain is at least `eps` times the empirical standard deviation of outcomes in the entire input data, and (3) the child nodes have different optimal treatment arm (or cluster) assignments from the parent node. Recursive partitioning ends when no further splits meet these criteria. Once terminal nodes are determined in the training subset, the learned splitting rules are applied to the estimation subset to assign its units to the terminal nodes. Outcomes from units in the estimation subset are used to calculate treatment-arm-specific average outcomes for each terminal node, with optional regularization specified by the `lambda2` parameter and imputation controlled by `impute`. On the held-out data set, treatment-arm-specific outcomes from the estimation subset are assigned to corresponding terminal nodes to achieve honest outcome estimates, thus concluding the tree growing process. The final step is bootstrap aggregation of a large number of trees, where the total number of trees is set by the `ntree` parameter.

![A description of the RJAF algorithm. \label{fig:rjaf}](rjaf.pdf){width=100%}

# Acknowledgments

Wenbo Wu and Xinyi Zhang contributed equally to the project. Wenbo Wu and Rahul Ladhania were partially supported by a research grant from the Robert Wood Johnson Foundation titled *Informing Strategies to Increase Use of COVID-19 and Flu Vaccines by Different Racial and Ethnic Groups to Improve Health Equity during Health Crises* (award number 78416).

# References
