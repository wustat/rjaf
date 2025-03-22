
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
clustering scheme. By inputting training, estimation, and held-out data,
we can obtain final regularized predictions and assignments in
`forest.reg`, where the algorithm estimates regularized averages
separately by the original treatment arms $k \in \{0,\ldots,K\}$ and
obtains the corresponding assignment.

## Example

``` r
library(rjaf)
```

We use a dataset simulated by `sim.data()` under the example section of
`rjaf.R`. This dataset contains a total of 100 rows and 5 treatment
arms, with a total of 12 covariates as documented in `data.R`. After
dividing the `Example_data` into training-estimation and held-out sets,
we can obtain regularized averages by 5 treatment arms and optimal
treatment assignments.

Our algorithm returns a list named `forest.reg`, which includes two
tibbles named `fitted` and `counterfactuals`. The `fitted` contains
individual IDs from the held-out set, optimal treatment arms identified
(`trt.rjaf`), predicted optimal outcomes (`Y.rjaf`), and treatment arm
clusters (`clus.rjaf`). In this example, since we know the
counterfactual outcomes, we include those under optimal treatment arms
`trt.rjaf` identified by the algorithm in `fitted` as `Y.cf`. The tibble
`counterfactuals` contains estimated counterfactual outcomes for every
treatment arm. If performing clustering, the tibble `xwalk` is also
returned by the algorithm. `xwalk` has the treatments and their assigned
cluster memberships (based on the k-means algorithm).

``` r
library(magrittr)
library(dplyr)

# prepare training, estimation, and heldout data
data("Example_data")
set.seed(1)
# training and estimation
data.trainest <- Example_data %>% 
                  slice_sample(n=floor(0.5*nrow(Example_data)))
# held-out
data.heldout <- Example_data %>% 
                  filter(!id %in% data.trainest$id)

# specify variables needed
id <- "id"; y <- "Y"; trt <- "trt";  
vars <- paste0("X", 1:3); prob <- "prob";

# calling the ``rjaf`` function and implement clustering scheme
forest.reg <- rjaf(data.trainest, data.heldout, y, id, trt, vars, 
                   prob, clus.max=3, 
                   clus.tree.growing=TRUE, setseed=TRUE)
```

``` r
head(forest.reg$fitted)
#> # A tibble: 6 × 5
#>   id    trt.rjaf   Y.cf Y.rjaf clus.rjaf
#>   <chr> <chr>     <dbl>  <dbl>     <int>
#> 1 3     2         13.3   11.6          3
#> 2 4     4          0      8.56         1
#> 3 5     2         -6.67   7.38         3
#> 4 8     3         13.3   18.9          2
#> 5 9     3        -26.7   20.7          2
#> 6 10    3        -26.7   16.7          2
head(forest.reg$counterfactuals)
#> # A tibble: 6 × 6
#>   Y_0.rjaf Y_1.rjaf Y_2.rjaf Y_3.rjaf Y_4.rjaf id   
#>      <dbl>    <dbl>    <dbl>    <dbl>    <dbl> <chr>
#> 1     1.55     6.72    11.6      5.77     9.18 3    
#> 2    -2.09     6.92     6.83    -4.10     8.56 4    
#> 3    -1.72     1.19     7.38     3.09     4.46 5    
#> 4     5.99    10.8     15.3     18.9     10.1  8    
#> 5     9.59     3.15    16.7     20.7     13.3  9    
#> 6     7.01     1.86    13.4     16.7      9.94 10
head(forest.reg$xwalk)
#>   cluster trt
#> 1       1   0
#> 2       1   1
#> 3       3   2
#> 4       2   3
#> 5       1   4
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
