
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rjaf

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/rjaf)](https://CRAN.R-project.org/package=rjaf)
![](http://cranlogs.r-pkg.org/badges/grand-total/rjaf) [![License: GPL
v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
<!-- badges: end -->

> Regularized Joint Assignment Forests with Treatment Arm Clustering

Wenbo Wu, Xinyi Zhang, Jann Spiess, Rahul Ladhania

------------------------------------------------------------------------

## Introduction

`rjaf` is an `R` package that implements a regularized and clustered
joint assignment forest which targets joint assignment to one of many
treatment arms as described in Ladhania, Spiess, Ungar, and Wu (2023).
It utilizes a regularized forest-based greedy recursive algorithm to
shrink effect estimates across arms and a clustering approach to combine
treatment arm with similar outcomes. The optimal treatment assignment is
estimated by pooling information across treatment arms. In this tutorial
we introduce the use of `rjaf` through an example data set.

## Installation

`rjaf` can be installed from CRAN with

    install.packages("rjaf")

The stable, development version can be installed from GitHub with

    require("devtools")
    require("remotes")
    devtools::install_github("wustat/rjaf", subdir = "r-package/rjaf")

## What is regularized and clustered joint assignment forest (rjaf)?

The algorithm aims to train a joint forest model to estimate the optimal
treatment assignment by pooling information across treatment arms.

It first obtains an assignment forest by bagging trees as described in
Kallus (2017), with covariate and treatment arm randomization for each
tree. Then, it generates “honest” and regularized estimates of the
treatment-specific counterfactual outcomes on the training sample
following Wager and Athey (2018).

Like Bonhomme and Manresa (2015), it uses a clustering of treatment arms
when constructing the assignment trees. It employs a k-means algorithm
for clustering the K treatment arms into M treatment groups based on the
K predictions for each of the n units in the training sample. After
clustering, it then repeats the assignment-forest algorithm on the full
training data with M+1 (including control) “arms” (where data from the
original arms are combined by groups) to obtain an ensemble of trees.

The following scripts demonstrate the function `rjaf()`, which
constructs a joint forest model to estimate the optimal treatment
assignment by pooling information across treatment arms using a
clustering scheme. By inputting training, estimation, and heldout data,
we can obtain final regularized predictions and assignments in
`forest.reg`, where the algorithm estimates regularized averages
separately by the original treatment arms $k \in \{0,\ldots,K\}$ and
obtains the corresponding assignment.

## Example

``` r
library(rjaf)
#> Warning: package 'rjaf' was built under R version 4.3.3
```

We use a data set simulated by `sim.data()` under the example section of
`rjaf.R`. This dataset contains a total of 100 items and 5 treatment
arms, with a total of 12 covariates as documented in `data.R`. After
preparing the `Example_data` into training, estimation, and heldout, we
can obtain regularized averages by 5 treatment arms and acquire the
optimal assignment.

Our algorithm returns a tibble named `forest.reg`, with individual IDs,
optimal treatment arms identified (`trt.rjaf`), predicted optimal
outcomes (`Y.rjaf`), and treatment arm clusters (`clus.rjaf`). As
counterfactual outcomes present, they are also included in the tibble
(`Y.cf`).

``` r
library(magrittr)
library(dplyr)

# prepare training, estimation, and heldout data
data("Example_data")

# training and estimation
data.trainest <- Example_data %>% 
                  slice_sample (n = floor(0.5 * nrow(Example_data)))
# heldout
data.heldout <- Example_data %>% 
                  filter (!id %in% data.trainest$id)

# specify variables needed
id <- "id"; y <- "Y"; trt <- "trt";  
vars <- paste0("X", 1:3); prob <- "prob";

# calling the ``rjaf`` function and implement clustering scheme
forest.reg <- rjaf(data.trainest, data.heldout, y, id, trt, vars, 
                   prob, clus.max = 3, 
                   clus.tree.growing = TRUE, setseed = TRUE)

head(forest.reg)
#> # A tibble: 6 × 5
#>   id    trt.rjaf  Y.cf Y.rjaf clus.rjaf
#>   <chr> <chr>    <dbl>  <dbl>     <int>
#> 1 2     4           40  16.5          2
#> 2 4     1           20  15.9          2
#> 3 5     4          -20   6.57         2
#> 4 11    4          -20  19.7          2
#> 5 12    4           20  22.3          2
#> 6 13    4          -20   6.55         2
```

## References

Bonhomme, Stéphane and Elena Manresa (2015). Grouped Patterns of
Heterogeneity in Panel Data. *Econometrica*, 83: 1147-1184.

Kallus, Nathan (2017). Recursive Partitioning for Personalization using
Observational Data. In Precup, Doina and Yee Whye Teh, editors,
Proceedings of the 34th International Conference on Machine Learning,
*Proceedings of the 34th International Conference on Machine Learning*,
PMLR 70:1789-1798.

Ladhania Rahul, Spiess Jann, Ungar Lyle, Wu Wenbo (2023). Personalized
Assignment to One of Many Treatment Arms via Regularized and Clustered
Joint Assignment Forests. <https://doi.org/10.48550/arXiv.2311.00577>.

Wager, Stefan and Susan Athey (2018). Estimation and inference of
heterogeneous treatment effects using random forests. *Journal of the
American Statistical Association*, 113(523):1228–1242.
