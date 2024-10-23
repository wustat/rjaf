---
title: '`rjaf`: Regularized Joint Assignment Forest with Treatment Arm Clustering'
tags:
- machine learning
- causal inference
- multi-armed randomized controlled trial
- heterogeneous treatment effects
- personalized treatment rules
- optimal assignment
- R
- C++
date: "May 23, 2024"
output: pdf_document
authors:
- name: Wenbo Wu
  orcid: 0000-0002-7642-9773
  corresponding: yes
  affiliation: 1
- name: Xinyi Zhang
  orcid: 
  affiliation: 1
- name: Jann Spiess
  orcid: 0000-0002-4120-8241
  affiliation: 2
- name: Rahul Ladhania
  orcid: 0000-0002-7902-7681
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

Figure 1 outlines the workflow for using the `rjaf` package to perform personalized treatment assignment and honest outcome estimation. The process begins with partitioning the input data--consisting of outcomes, treatment arms, covariates, individual identifiers, and optional probabilities of treatment assignment--into two parts, one for training and estimation, and the other for validation. The `rjaf` function first checks whether residualization should be performed via the `residualize` function, using the `resid` argument. If `resid` is set to `TRUE` (the default), a new column of residualized outcomes is added to the input data and used for tree growing on the training subset to reduce baseline variation. Next, the `rjaf` function evaluates whether treatment clustering should be performed on the training and estimation set during tree growing using the `clus.tree.growing` argument. If `clus.tree.growing` is `TRUE`, the `rjaf_cpp` function is employed to estimate cross-validated counterfactual outcomes, which are then subjected to k-means clustering. The optimal number of treatment clusters is determined using the elbow method. Lastly, the `rjaf_cpp` function is applied to the pre-processed data to obtain optimal treatment arms or clusters and predicted outcomes for all individuals in the validation set.

Figure 2 provides a detailed description of the `rjaf_cpp` function. The function can be used to grow a large number of trees using the `growTree` function. The training and estimation set is taken as input by the `growTree` function, which is then split into two separate training and estimation sets by treatment arms or clusters. The first step of growing a tree is to initialize the utility at the root node, where the optional inverse probability weighting (IPW) can be applied. A whole tree is then grown via recursive partitioning of the training subset at covariate splits. Each potential split is generated by the `splitting` function, where regularization (tuned via the parameter `lambda1`) can be performed along with IPW to calculate average outcomes across treatment arms or clusters. A potential split is kept only when the following three criteria are met: (1) there are at least a certain number of units (controlled via the `nodesize` argument) in each child node; (2) the increase in utility is at least `epi` times the empirical standard deviation of outcomes in the entire input data; and (3) the two child nodes have different optimal treatment arm or cluster assignment. The recursive partitioning is terminated when no potential split meets the above three criteria. After recursive partitioning, all terminal nodes are determined on the training subset. The same set of splits is then applied to the estimation subset and validation set. Next, outcomes from estimation subset are used to calculate treatment-specific average outcomes for all terminal nodes, with optional regularization tuned via the parameter `lambda2`.



 (simulation results available in \autoref{fig:compare})

![Direct optimization forest, random forest, and multi-arm causal forest with an increasing number of treatment arms. \label{fig:compare}](Figure1JOSS.pdf){width=85%}

parameter tuning is not provided in the package.

# Mathematics

Single dollars (\$) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX{} for equations \begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation} and refer to \autoref{eq:fourier} from text.

# Acknowledgments

Wenbo Wu and Rahul Ladhania were supported by a research grant from the Robert Wood Johnson Foundation titled *Informing Strategies to Increase Use of COVID-19 and Flu Vaccines by Different Racial and Ethnic Groups to Improve Health Equity during Health Crises* (award number 78416).

# References
