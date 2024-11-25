#' Regularized Joint Assignment Forest with Treatment Arm Clustering
#' 
#' This algorithm trains a joint forest model to estimate the optimal treatment assignment
#' by pooling information across treatment arms.
#'
#' It first obtains an assignment forest by bagging trees as in Kallus (2017) with 
#' covariate and treatment arm randomization for each tree
#' and estimating "honest" and regularized estimates of the treatment-specific counterfactual outcomes
#' on the training sample following Wager and Athey (2018).
#'
#' Like Bonhomme and Manresa (2015), it uses a clustering of treatment arms when 
#' constructing the assignment trees. It employs a k-means algorithm for
#' clustering the K treatment arms into M treatment groups 
#' based on the K predictions for each of the n units in the training sample.
#'
#' After clustering, it then repeats the assignment-forest algorithm on the full training data 
#' with M+1 (including control) "arms" (where data from the original arms are combined by groups) 
#' to obtain an ensemble of trees.
#'
#' It obtains final regularized predictions and assignments, where it estimates 
#' regularized averages separately by the original treatment arms \eqn{k \in \{0,\ldots,K\}}
#' and obtain the corresponding assignment.
#'
#' @param data.trainest input data used for training and estimation, where each
#' row corresponds to an individual and columns contain information on treatments,
#' covariates, probabilities of treatment assignment, and observed outcomes.
#' @param data.heldout input data used for validation with the same row and
#' column information as in `data.trainest`.
#' @param y a character string denoting the column name of outcomes.
#' @param id a character string denoting the column name of individual IDs.
#' @param trt a character string denoting the column name of treatments.
#' @param vars a vector of character strings denoting the column names of covariates. 
#' @param prob a character string denoting the column name of probabilities of
#' treatment assignment. If missing, a column named "prob" will be added to `data.trainest` and
#' `data.heldout` indicating simple random treatment assignment.
#' @param ntrt number of treatments randomly sampled at each split. It should be
#' at most equal to the number of unique treatments available. The default value is 5.
#' @param nvar number of covariates randomly sampled at each split. It should be
#' at most equal to the number of unique covariates available. The default value is 3.
#' @param lambda1 regularization parameter for shrinking arm-wise within-leaf average
#' outcomes towards the overall within-leaf average outcome during recursive
#' partitioning. The default value is 0.5.
#' @param lambda2 regularization parameter for shrinking arm-wise within-leaf average
#' outcomes towards the overall within-leaf average outcome during outcome estimation.
#' It is only valid when `reg` is `TRUE`. The default value is 0.5.
#' @param ipw a logical indicator of inverse probability weighting when calculating
#' leaf-wise weighted averages based on Wu and Gagnon-Bartsch (2018). The default value is `TRUE`.
#' @param nodesize minimum number of observations in a terminal node. The default value is 5.
#' @param ntree number of trees to grow in the forest. This should not be set to
#' too small a number. The default value is 1000.
#' @param prop.train proportion of data used for training in `data.trainest`.
#' The default value is 0.5.
#' @param eps threshold for minimal welfare gain in terms of the empirical standard
#' deviation of the overall outcome `y`. The default value is 0.1.
#' @param resid a logical indicator of arbitrary residualization. If `TRUE`,
#' residualization is implemented to reduce the variance of the outcome.
#' The default value is `TRUE`.
#' @param clus.tree.growing a logical indicator of clustering for tree growing.
#' The default value is `FALSE`.
#' @param clus.outcome.avg a logical indicator of clustering for tree bagging.
#' If `TRUE`, the average outcome is calculated across treatment clusters
#' determined by the k-means. The default value is `FALSE`. This option is deprecated.
#' @param clus.max the maximum number of clusters for k-means. It should be
#' greater than 1 and at most equal to the number of unique treatments.
#' The default value is 10.
#' @param reg a logical indicator of regularization when calculating the arm-wise
#' within-leaf average outcome.
#' @param impute a logical indicator of imputation. If `TRUE`, the within-leaf
#' average outcome is used to impute the arm-wise within-leaf average outcome
#' when the arm has no observation. If `FALSE`, the within-leaf average outcome
#' is set to zero when the arm has no observation. The default value is `TRUE`.
#' @param setseed a logical indicator. If `TRUE`, a seed is set through the
#' argument `seed` below and passed to the function `rjaf_cpp`.
#' The default value is `FALSE`.
#' @param seed an integer used as a random seed if `setseed=TRUE`.
#' The default value is 1.
#' @param nfold the number of folds used for cross-validation in outcome
#' residualization and k-means clustering. The default value is 5.
#' 
#' 
#' @return If `clus.tree.growing` and `clus.outcome.avg` are `TRUE`, `rjaf`
#' returns a list of two objects: a tibble named as `res` consisting of individual
#' IDs, cluster identifiers, and predicted outcomes, and a data frame named as
#' `clustering` consisting of cluster identifiers, probabilities of being assigned
#' to the clusters, and treatment arms. Otherwise, `rjaf` simply returns a tibble
#' of individual IDs (`id`), optimal treatment arms identified by the algorithm (`trt.rjaf`), treatment
#' clusters (`clus.rjaf`) if `clus.tree.growing` is `TRUE`, and predicted optimal outcomes (`Y.rjaf`). 
#' If counterfactual outcomes are also present, they will be included
#' in the tibble along with the column of predicted outcomes (`Y.cf`).
#' @export
#'
#' @examples
#' library(dplyr)
#' library(MASS)
#' sim.data <- function(n, K, gamma, sigma, prob=rep(1,K+1)/(K+1)) {
#'    # K: number of treatment arms
#'   options(stringsAsFactors=FALSE)
#'   data <- left_join(data.frame(id=1:n,
#'                                trt=sample(0:K, n, replace=TRUE, prob),
#'                                mvrnorm(n, rep(0,3), diag(3))),
#'                     data.frame(trt=0:K, prob), by="trt")
#'   data <- mutate(data, tmp1=10+20*(X1>0)-20*(X2>0)-40*(X1>0&X2>0),
#'                  tmp2=gamma*(2*(X3>0)-1)/(K-1),
#'                  tmp3=-10*X1^2,
#'                  Y=tmp1+tmp2*(trt>0)*(2*trt-K-1)+tmp3*(trt==0)+rnorm(n,0,sigma))
#'   # Y: observed outcomes
#'   Y.cf <- data.frame(sapply(0:K, function(t) # counterfactual outcomes
#'     mutate(data, Y=tmp1+tmp2*(t>0)*(2*t-K-1)+tmp3*(t==0))$Y))
#'   names(Y.cf) <- paste0("Y",0:K)
#'   return(mutate(bind_cols(dplyr::select(data, -c(tmp1,tmp2,tmp3)), Y.cf),
#'                 across(c(id, trt), as.character)))
#' }
#' 
#' n <- 200; K <- 3; gamma <- 10; sigma <- 10
#' Example_data <- sim.data(n, K, gamma, sigma)
#' Example_trainest <- Example_data %>% slice_sample(n = floor(0.5 * nrow(Example_data)))
#' Example_heldout <- Example_data %>% filter(!id %in% Example_trainest$id)
#' id <- "id"; y <- "Y"; trt <- "trt"
#' vars <- paste0("X", 1:3)
#' forest.reg <- rjaf(Example_trainest, Example_heldout, y, id, trt, vars, ntrt = 4, ntree = 100,
#'                    clus.tree.growing = FALSE)
#'
#' @useDynLib rjaf, .registration=TRUE
#' @importFrom Rcpp evalCpp 
#' @importFrom stats kmeans as.formula predict
#' @importFrom rlang := 
#' @importFrom MASS mvrnorm
#' @import dplyr forcats magrittr readr tibble stringr
#'
#' @references 
#' Bonhomme, Stéphane and Elena Manresa (2015). Grouped Patterns of Heterogeneity in Panel Data. Econometrica, 83: 1147-1184.
#' \cr
#' 
#' Kallus, Nathan (2017). Recursive Partitioning for Personalization using Observational Data. In Precup, Doina and Yee Whye Teh, editors, 
#' Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1789–1798. PMLR.
#' \cr
#' 
#' Wager, Stefan and Susan Athey (2018). Estimation and inference of heterogeneous treatment effects
#' using random forests. Journal of the American Statistical Association, 113(523):1228–1242.
#' \cr
#' 
#' Wu, Edward and Johann A Gagnon-Bartsch (2018). The LOOP Estimator: Adjusting
#' for Covariates in Randomized Experiments. Evaluation Review, 42(4):458–488.
#' \cr
#' 

rjaf <- function(data.trainest, data.heldout, y, id, trt, vars, prob,
                 ntrt=5, nvar=3, lambda1=0.5, lambda2=0.5, ipw=TRUE,
                 nodesize=5, ntree=1000, prop.train=0.5, eps=0.1,
                 resid=TRUE, clus.tree.growing=FALSE, clus.outcome.avg=FALSE,
                 clus.max=10, reg=TRUE, impute=TRUE,
                 setseed=FALSE, seed=1, nfold=5) {
  trts <- unique(pull(data.trainest, trt))
  if (ntrt>length(trts)) stop("Invalid ntrt!")
  if (nvar>length(vars)) stop("Invalid nvar!")
  if (missing(prob)) { # default to simple random treatment assignment
    prob <- "prob"
    proportions <- (table(data.trainest$trt) + table(data.heldout$trt)) / (nrow(data.trainest) + nrow(data.heldout))
    data.trainest <- data.trainest %>% mutate(!!(prob):= proportions[as.character(trt)])
    data.heldout <- data.heldout %>% mutate(!!(prob):= proportions[as.character(trt)])
  }
  data.trainest <- mutate(data.trainest, across(c(id, trt), as.character))
  data.heldout <- mutate(data.heldout, across(c(id, trt), as.character))
  if (resid) {
    data.trainest <- residualize(data.trainest, y, vars, nfold)
  } else { # if resid is FALSE, the two columns of outcomes are identical.
    data.trainest <- data.trainest %>% mutate(!!(paste0(y, ".resid")):=!!sym(y))
  }
  if (clus.tree.growing) {
    if (clus.max>length(trts) | clus.max<2) stop("Invalid clus.max!")
    fold <- sample(1:nfold, NROW(data.trainest), TRUE, rep(1, nfold))
    data.trainest <- data.trainest %>%
      mutate(fold=fold)
    ls.kmeans <- lapply(2:clus.max, function(i)
      stats::kmeans(
        t(do.call(rbind, lapply(1:nfold, function(k) {
          data.onefold <- data.trainest %>% filter(fold==k)
          data.rest <- data.trainest %>% filter(fold!=k)
          rjaf_cpp(pull(data.rest, y), pull(data.rest, paste0(y, ".resid")),
                   as.matrix(dplyr::select(data.rest, all_of(vars))),
                   as.integer(factor(pull(data.rest, trt),
                                     as.character(trts))),
                   pull(data.rest, prob),
                   as.integer(factor(pull(data.rest, trt),
                                     as.character(trts))),
                   as.matrix(dplyr::select(data.onefold, all_of(vars))),
                   ntrt, nvar, lambda1, lambda2, ipw, nodesize, ntree,
                   prop.train, eps, reg, impute, setseed, seed)$Y.cf
        }))), i, nstart=5))
    vec.prop <- sapply(ls.kmeans, function(list) list$betweenss/list$totss)
    cluster <- ls.kmeans[[which.max(diff(vec.prop))+1]]$cluster
    df <- data.frame(cluster)
    df[trt] <- as.character(trts)
    xwalk <- df
    df <- summarise(group_by(data.trainest, !!sym(trt)),
                    !!(prob):=mean(!!sym(prob)), .groups="drop") %>%
      inner_join(df, by=trt) %>% group_by(cluster) %>%
      summarise(prob_cluster=sum(!!sym(prob)), .groups="drop") %>%
      inner_join(df, by="cluster") %>% as.data.frame
    data.trainest <- inner_join(data.trainest, df, by=trt) %>%
      mutate(cluster=as.character(cluster))
    clus <- unique(pull(data.trainest, cluster))
    str.tree.growing <- as.integer(factor(pull(data.trainest, cluster),
                                          as.character(clus)))
    prob.tree.growing <- data.trainest %>% pull(.data$prob_cluster)
    nstr <- length(unique(cluster))
    if (clus.outcome.avg) {
      str.outcome.avg <- as.integer(factor(pull(data.trainest, cluster),
                                           as.character(clus)))
    } else {
      str.outcome.avg <- as.integer(factor(pull(data.trainest, trt),
                                           as.character(trts)))
    }
  } else {
    str.tree.growing <- as.integer(factor(pull(data.trainest, trt),
                                          as.character(trts)))
    prob.tree.growing <- pull(data.trainest, prob)
    nstr <- ntrt
    str.outcome.avg <- as.integer(factor(pull(data.trainest, trt),
                                         as.character(trts)))
  }
  ls.forest <-
    rjaf_cpp(pull(data.trainest, y), pull(data.trainest, paste0(y, ".resid")),
             as.matrix(dplyr::select(data.trainest, all_of(vars))),
             str.tree.growing, prob.tree.growing, str.outcome.avg,
             as.matrix(dplyr::select(data.heldout, all_of(vars))),
             nstr, nvar, lambda1, lambda2, ipw, nodesize, ntree,
             prop.train, eps, reg, impute, setseed, seed)
  if (clus.tree.growing & clus.outcome.avg) {
    res <- tibble(!!(id):=as.character(pull(data.heldout, id)),
                  cluster=as.character(clus[ls.forest$trt.rjaf]),
                  !!(paste0(y, ".rjaf")):=as.numeric(ls.forest$Y.pred))
    return(list(res=res, clustering=df))
  } else {
    res <- tibble(!!(id):=as.character(pull(data.heldout, id)),
                  !!(trt):=as.character(trts[ls.forest$trt.rjaf]),
                  !!(paste0(y, ".rjaf")):=as.numeric(ls.forest$Y.pred))
    if (clus.tree.growing) {
      res <- res %>% left_join(xwalk, by=trt) %>%
        rename(clus.rjaf=cluster)
    }
    if (all(paste0(y, trts) %in% names(data.heldout))) {
      # all counterfactual outcomes are present
      res <- data.heldout %>%
        dplyr::select(all_of(c(id, paste0(y, trts)))) %>%
        tidyr::pivot_longer(cols=paste0(y, trts), names_to=trt, names_prefix=y,
                     values_to=y) %>%
        mutate(across(c(id, trt), as.character)) %>%
        inner_join(res, by=c(id, trt)) %>%
        rename_with(~str_c(.,".rjaf"), trt) %>%
        rename_with(~str_c(.,".cf"), y)
    }
    return(res)
  }
}
