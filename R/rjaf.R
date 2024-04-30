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
#' regularized averages separately by the original treatment arms $k \\in \\{0,\ldots,K\\}$ 
#' and obtain the corresponding assignment.
#'
#' @param data.trainest input data used for training and estimation, where each
#' row corresponds to an individual and columns contain information on treatments,
#' covariates, probabilities of treatment assignment, and observed outcomes.
#' @param data.validation input data used for validation with the same row and
#' column information as in `data.trainest`.
#' @param y a character string denoting the column name of outcomes.
#' @param id a character string denoting the column name of individual IDs.
#' @param trt a character string denoting the column name of treatments.
#' @param vars a vector of character strings denoting the column names of covariates. 
#' @param prob a character string denoting the column name of probabilities of
#' treatment assignment.
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
#' @param epi threshold for minimal welfare gain in terms of the empirical standard
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
#' of individual IDs, treatment arms identified by the algorithm, and predicted
#' outcomes. If counterfactual outcomes are also present, they will be included
#' in the tibble along with the column of predicted outcomes (ending with `.rjaf`).

#' @export
#'
#' @examples
#' 
#' 
#'\dontrun{
#' sim.data <- function(n, K, gamma, sigma, count=rep(1,K+1)) {
#'   # K: number of clusters
#'   options(stringsAsFactors=F)
#'   data <- left_join(data.frame(id=1:n,
#'                                cl=sample(0:K, n, T, count),
#'                                cid=0,
#'                                MASS::mvrnorm(n, rep(0,3), diag(3))),
#'                     data.frame(cl=0:K, prob=1/sum(count)), by="cl")
#'   invisible(sapply(0:K, function(t) data[data$cl==t, "cid"] <<-
#'                      sample(count[t+1], sum(data$cl==t), T, rep(1, count[t+1]))))
#'   data <- data %>%
#'     mutate(tmp1=10+20*(X1>0)-20*(X2>0)-40*(X1>0&X2>0),
#'            tmp2=gamma*(2*(X3>0)-1)/(K-1),
#'            tmp3=-10*X1^2,
#'            Y=tmp1+tmp2*(cl>0)*(2*cl-K-1)+tmp3*(cl==0)+rnorm(n,0,sigma),
#'            trt=str_c("c", cl, "t", cid))
#'   mapping <- data %>% distinct(trt, .keep_all=T) %>%
#'     dplyr::select(c(cl, cid, trt)) %>% arrange(trt)
#'   # Y: observed outcomes
#'   Y.cf.trt <- data.frame(sapply(mapping %>% pull(trt), function(t) {
#'     # counterfactural outcomes
#'     clus <- mapping %>% filter(trt==t) %>% pull(cl)
#'     mutate(data, Y=tmp1+tmp2*(clus>0)*(2*clus-K-1)+tmp3*(clus==0))$Y
#'   }))
#'   names(Y.cf.trt) <- paste0("Y", mapping %>% pull(trt))
#'   Y.cf.cl <- data.frame(sapply(0:K, function(t) {
#'     # counterfactural outcomes
#'     mutate(data, Y=tmp1+tmp2*(t>0)*(2*t-K-1)+tmp3*(t==0))$Y
#'   }))
#'   names(Y.cf.cl) <- paste0("Y",0:K)
#'   return(mutate(bind_cols(dplyr::select(data, -c(tmp1,tmp2,tmp3)), Y.cf.trt, Y.cf.cl),
#'                 across(c(id, cl), as.character)))
#' }
#' 
#' n <- 100; K <- 5; gamma <- 10; sigma <- 10
#' count <- rep(1, K+1)
#' Example_data <- sim.data(n, K, gamma, sigma, count)
#' Example_trainest <- Example_data %>% slice_sample(n = floor(0.3 * nrow(Example_data)))
#' Example_valid <- Example_data %>% filter(!id %in% Example_trainest$id)
#' id <- "id"; trts <- as.character(0:K); y <- "Y"; trt <- "trt";  
#' vars <- paste0("X", 1:3); prob <- "prob";
#' forest.reg <- rjaf(Example_trainest, Example_valid, y, id, trt, vars, prob, clus.max = 3)
#'}
#'
#' @useDynLib rjaf, .registration=TRUE
#' @importFrom Rcpp evalCpp 
#' @importFrom stats kmeans as.formula predict
#' @importFrom rlang :=
#' @import dplyr forcats magrittr readr tibble
#'
#' @references 
#' Bonhomme, Stéphane and Elena Manresa (2015). Grouped Patterns of Heterogeneity in Panel Data. Econometrica, 83: 1147-1184.
#' \cr
#' Kallus, Nathan (2017). Recursive Partitioning for Personalization using Observational Data. In Precup, Doina and Yee Whye Teh, editors, 
#' Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1789–1798. PMLR.
#' \cr
#' Wager, Stefan and Susan Athey (2018). Estimation and inference of heterogeneous treatment effects
#' using random forests. Journal of the American Statistical Association, 113(523):1228–1242.
#' \cr
#' Wu, Edward and Johann A Gagnon-Bartsch (2018). The LOOP Estimator: Adjusting
#' for Covariates in Randomized Experiments. Evaluation Review, 42(4):458–488.
#' \cr
#' 

rjaf <- function(data.trainest, data.validation, y, id, trt, vars, prob,
                 ntrt=5, nvar=3, lambda1=0.5, lambda2=0.5, ipw=TRUE,
                 nodesize=5, ntree=1000, prop.train=0.5, epi=0.1,
                 resid=TRUE, clus.tree.growing=FALSE, clus.outcome.avg=FALSE,
                 clus.max=10, reg=TRUE, impute=TRUE,
                 setseed=FALSE, seed=1, nfold=5) {
  trts <- unique(pull(data.trainest, trt))
  if (ntrt>length(trts)) stop("Invalid ntrt!")
  if (nvar>length(vars)) stop("Invalid nvar!")
  data.trainest <- mutate(data.trainest, across(c(id, trt), as.character))
  data.validation <- mutate(data.validation, across(c(id, trt), as.character))
  if (resid) data.trainest <- residualize(data.trainest, y, vars, nfold)
  if (clus.tree.growing) {
    if (clus.max>length(trts) | clus.max<2) stop("Invalid clus.max!")
    data.trainest$fold <- sample(1:nfold, NROW(data.trainest), T, rep(1, nfold))
    ls.kmeans <- lapply(2:clus.max, function(i)
      stats::kmeans(
        t(do.call(rbind, lapply(1:nfold, function(k) {
          data.onefold <- data.trainest %>% filter(fold==k)
          data.rest <- data.trainest %>% filter(fold!=k)
          rjaf_cpp(pull(data.rest, y),
                   as.matrix(select(data.rest, all_of(vars))),
                   as.integer(factor(pull(data.rest, trt),
                                     as.character(trts))),
                   pull(data.rest, prob),
                   as.integer(factor(pull(data.rest, trt),
                                     as.character(trts))),
                   as.matrix(select(data.onefold, all_of(vars))),
                   ntrt, nvar, lambda1, lambda2, ipw, nodesize, ntree,
                   prop.train, epi, reg, impute, setseed, seed)$Y.cf
        }))), i, nstart=5))
    vec.prop <- sapply(ls.kmeans, function(list) list$betweenss/list$totss)
    cluster <- ls.kmeans[[which.max(diff(vec.prop))+1]]$cluster
    df <- data.frame(cluster)
    df[trt] <- as.character(trts)
    df <- summarise(group_by(data.trainest, trt),
                    !!(prob):=mean(!!sym(prob)), .groups="drop") %>%
      inner_join(df, by=trt) %>% group_by(cluster) %>%
      summarise(prob_cluster=sum(!!sym(prob)), .groups="drop") %>%
      inner_join(df, by="cluster") %>% as.data.frame
    data.trainest <- inner_join(data.trainest, df, by=trt) %>%
      mutate(cluster=as.character(cluster))
    clus <- unique(pull(data.trainest, cluster))
    str.tree.growing <- as.integer(factor(pull(data.trainest, cluster),
                                          as.character(clus)))
    prob.tree.growing <- data.trainest %>% pull(prob_cluster)
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
    rjaf_cpp(pull(data.trainest, y),
             as.matrix(select(data.trainest, all_of(vars))),
             str.tree.growing, prob.tree.growing, str.outcome.avg,
             as.matrix(select(data.validation, all_of(vars))),
             nstr, nvar, lambda1, lambda2, ipw, nodesize, ntree,
             prop.train, epi, reg, impute, setseed, seed)
  if (clus.tree.growing & clus.outcome.avg) {
    res <- tibble(!!(id):=as.character(pull(data.validation, id)),
                  cluster=as.character(clus[ls.forest$trt.rjaf]),
                  !!(paste0(y, ".pred")):=as.numeric(ls.forest$Y.pred))
    return(list(res=res, clustering=df))
  } else {
    res <- tibble(!!(id):=as.character(pull(data.validation, id)),
                  !!(trt):=as.character(trts[ls.forest$trt.rjaf]),
                  !!(paste0(y, ".pred")):=as.numeric(ls.forest$Y.pred))
    if (all(paste0(y, trts) %in% names(data.validation))) {
      # all counterfactual outcomes are present
      res <- data.validation %>%
        dplyr::select(all_of(c(id, paste0(y, trts)))) %>%
        tidyr::pivot_longer(cols=paste0(y, trts), names_to=trt, names_prefix=y,
                     values_to=y) %>%
        mutate(across(c(id, trt), as.character)) %>%
        inner_join(res, by=c(id, trt)) %>%
        rename_with(~str_c(.,".rjaf"), all_of(c(y, trt)))
    }
    return(res)
  }
}
