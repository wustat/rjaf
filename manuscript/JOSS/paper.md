---
title: '`rjaf`: Regularized Joint Assignment Forest with Treatment Arm Clustering'
tags:
- machine learning
- causal inference
- "multi-armed randomized controlled trial"
- heterogeneous treatment effects
- personalized treatment rules
- optimal assignment
- R
- C++
date: "October 27, 2024"
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
  orcid: null
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

Learning the assignment of treatments is an omnipresent problem in economics and public health. It arises, for example, from randomized controlled trials where a variety of behavioral nudges (treatments) are developed to enhance vaccination uptake against coronavirus disease 2019 (COVID-19) or influenza, especially among racially or ethnically underrepresented and socioeconomically disadvantaged populations [@milkman2021megastudy; @dai2021behavioural; @milkman2022680]. Subject-specific covariates containing information regarding sociodemographics, clinical characteristics, and comorbid conditions, if available, can be harnessed to identify personalized treatment assignment schemes. The `rjaf` package provides a user-friendly implementation of the regularized joint assignment forest (RJAF) [@ladhania2023personalized], a forest-based treatment assignment algorithm featuring greedy recursive partitioning [@athey2019generalized], treatment and covariate resampling in bootstrap tree aggregating [@breiman1996bagging], outcome residualization and regularization, and k-means treatment arm clustering [@hartigan1979algorithm]. Personalized treatment learning is achieved through optimizing a regularized empirical analogue of the expected outcome. The integration of `R` [@R2024] and `C++` [@stroustrup2013c] substantially boosts the computational efficiency in tree partitioning and aggregating. This package is especially suitable in randomized controlled trial settings where a large number of treatment arms are present.

# Statement of Need

There is an ever-growing literature in the intersection of machine learning and causal inference attempting to address the problem of optimal treatment assignment through heterogeneous treatment effect estimation [@athey2016recursive; @wager2018estimation; @hitsch2018heterogeneous; @athey2019generalized; @sverdrup2020policytree; @athey2021policy]. Other methods focus on maximizing the benefit (empirical welfare) from treatment assignment [e.g., @kitagawa2018should], or the chance of assigning an individual to an optimal treatment arm [e.g., @JMLR:v6:murphy05a; @zhou2018sequential]. Most of these methods perform well with a limited number of treatment and control groups. As more arms are present, the estimation of arm-specific empirical welfare and the identification of individual-specific optimal arms become increasingly difficult. Commonly used implementations such as the multi-arm causal forest [@tibshirani2020grf] and random forest [@JSSv077i01] lead to significantly suboptimal assignment with insufficient levels of empirical welfare. By contrast, the RJAF yields elevated empirical welfare closer to the optimal level from the oracle assignment than the multi-arm causal forest and random forest. Despite the methodological advantage over existing approaches, the incorporation of machine learning and causal inference techniques such as recursive tree partitioning, bootstrap aggregating, and treatment arm clustering makes it challenging to implement the RJAF from scratch even for well-trained data scientists. The `rjaf` is an open-source software package in `R` and `C++` that efficiently implements the RJAF, offering data scientists a user-friendly analytic toolbox for learning personalized treatment rules in real-world settings.

# Workflow

\autoref{fig:pkg} outlines the workflow for using the `rjaf` package to perform personalized treatment assignment and honest outcome estimation. The process begins with partitioning the input data--consisting of outcomes, treatment arms, covariates, individual identifiers, and optional probabilities of treatment assignment--into two parts, one for training and estimation, and the other for validation. The `rjaf` function first checks whether residualization should be performed via the `residualize` function, using the `resid` argument. If `resid` is set to `TRUE` (the default), a new column of residualized outcomes is added to the input data and used for tree growing on the training subset to reduce baseline variation. Next, the `rjaf` function evaluates whether treatment clustering should be performed on the training and estimation set during tree growing using the `clus.tree.growing` argument. If `clus.tree.growing` is `TRUE`, the `rjaf_cpp` function is employed to estimate cross-validated counterfactual outcomes, which are then subjected to k-means clustering. The optimal number of treatment clusters is determined using the elbow method. Lastly, the `rjaf_cpp` function is applied to the pre-processed data to obtain optimal treatment arms or clusters and predicted outcomes for all individuals in the validation set.

![A sketch of the `rjaf` package. \label{fig:pkg}](pkg_sketch.pdf){height=95%}

\autoref{fig:rjaf_cpp} provides a detailed description of the `rjaf_cpp` function, which grows a large number of trees through the `growTree` function. `growTree` begins by taking the training and estimation set as input, splitting it into two separate training and estimation subsets proportionally by treatment arms or clusters. Initially, utility is set at the root node, where optional inverse probability weighting (IPW) can be applied. A tree is then grown via recursive partitioning of the training subset based on covariate splits. Each potential split is generated by the `splitting` function, where regularization tuned by the `lambda1` parameter can be performed along with IPW to calculate average outcomes across treatment arms or clusters. A potential split is retained if it meets three criteria: (1) each child node contains at least the minimum number of units specified by the `nodesize` argument, (2) the utility gain is at least `epi` times the empirical standard deviation of outcomes in the entire input data, and (3) the child nodes have different optimal treatment arm or cluster assignments. Recursive partitioning ends when no potential split meets these criteria. Once terminal nodes are determined on the training subset, the same splitting rules applied to the estimation subset to obtain its terminal nodes. Outcomes from the estimation subset are then used to calculate treatment-specific average outcomes in each terminal node, with optional regularization tuned by `lambda2` and imputation controlled by `impute`. The validation set undergoes the same splitting, with treatment-specific outcomes from the estimation subset assigned to corresponding terminal nodes to achieve honest outcome estimates, thus concluding the `growTree` function. The final step in `rjaf_cpp` is bootstrap aggregation of a large number of trees grown via `growTree`, where the total number of trees is set by the `ntree` parameter of the `rjaf_cpp` function.

As in the implementations of other forest-based methods, built-in hyper-parameter tuning (e.g., `epi`, `lambda1`, and `lambda2`) is not provided in the `rjaf` package. Interested users are referred to the `caret` package [@kuhn2008building] for details.

![A description of the `rjaf_cpp` function. \label{fig:rjaf_cpp}](rjaf_cpp.pdf){width=100%}

# Availability

The `rjaf` package is publicly available on [GitHub](https://github.com/wustat/rjaf), where the use of the `rjaf` package, including installation instructions and an example, has been documented in the `README.md` file. The package is also available on the Comprehensive R Archive Network.

# Acknowledgments

Wenbo Wu and Rahul Ladhania were partially supported by a research grant from the Robert Wood Johnson Foundation titled *Informing Strategies to Increase Use of COVID-19 and Flu Vaccines by Different Racial and Ethnic Groups to Improve Health Equity during Health Crises* (award number 78416).

# References
