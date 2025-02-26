
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rjaf

<!-- badges: start -->

[![CRANstatus](https://www.r-pkg.org/badges/version/rjaf)](https://cran.r-project.org/package=rjaf)
[![](https://cranlogs.r-pkg.org/badges/grand-total/rjaf)](https://cran.r-project.org/package=rjaf)
[![License: GPL
v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![status](https://joss.theoj.org/papers/ff8fa725cc40d0247158bd244f1117be/status.svg)](https://joss.theoj.org/papers/ff8fa725cc40d0247158bd244f1117be)
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
```

We use a data set simulated by `sim.data()` under the example section of
`rjaf.R`. This dataset contains a total of 100 items and 5 treatment
arms, with a total of 12 covariates as documented in `data.R`. After
preparing the `Example_data` into training, estimation, and heldout, we
can obtain regularized averages by 5 treatment arms and acquire the
optimal assignment.

Our algorithm returns a list named `forest.reg`, which includes two
tibbles named `fitted` and `counterfactuals`. `fitted` contains
individual IDs, optimal treatment arms identified (`trt.rjaf`),
predicted optimal outcomes (`Y.rjaf`), and treatment arm clusters
(`clus.rjaf`). As counterfactual outcomes present, they are also
included in `fitted` as `Y.cf`. `counterfactuals` contains estimated
counterfactual outcomes from every treatment arm. If performing
clustering, `xwalk` is included, which contains cluster number of
treatment assigned by k-means algorithm.

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
```

``` r
head(forest.reg$fitted)
#> # A tibble: 6 × 5
#>   id    trt.rjaf  Y.cf Y.rjaf clus.rjaf
#>   <chr> <chr>    <dbl>  <dbl>     <int>
#> 1 3     4            0   16.6         2
#> 2 4     4            0   15.0         2
#> 3 5     4          -20   10.6         2
#> 4 6     4           40   23.8         2
#> 5 8     4           20   17.8         2
#> 6 9     4          -20   22.8         2
head(forest.reg$counterfactuals)
#> # A tibble: 6 × 5
#>   Y_0.rjaf Y_1.rjaf Y_2.rjaf Y_3.rjaf Y_4.rjaf
#>      <dbl>    <dbl>    <dbl>    <dbl>    <dbl>
#> 1    -1.84   0.0886    1.35     0.141     16.6
#> 2    -5.35  -1.07     -0.464   -4.02      15.0
#> 3    -6.77  -7.86     -4.25    -8.76      10.6
#> 4    -4.37  -6.92     10.2     10.2       23.8
#> 5     4.69   7.93      6.87    11.1       17.8
#> 6   -11.2  -14.1       6.02     2.69      22.8
head(forest.reg$xwalk)
#>   cluster trt
#> 1       3   0
#> 2       1   1
#> 3       1   2
#> 4       1   3
#> 5       2   4
```

## References

Bonhomme, Stéphane and Elena Manresa (2015). Grouped Patterns of
Heterogeneity in Panel Data. *Econometrica*, 83: 1147-1184.

Kallus, Nathan (2017). Recursive Partitioning for Personalization using
Observational Data. In Precup, Doina and Yee Whye Teh, editors,
Proceedings of the 34th International Conference on Machine Learning,
*Proceedings of the 34th International Conference on Machine Learning*,
PMLR 70:1789-1798.

Ladhania, Rahul, Jann Spiess, Lyle Ungar, and Wenbo Wu (2023).
Personalized Assignment to One of Many Treatment Arms via Regularized
and Clustered Joint Assignment Forests.
<https://doi.org/10.48550/arXiv.2311.00577>.

Wager, Stefan and Susan Athey (2018). Estimation and inference of
heterogeneous treatment effects using random forests. *Journal of the
American Statistical Association*, 113(523):1228–1242.
