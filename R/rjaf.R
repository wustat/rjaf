#' Regularized joint assignment forest with treatment clustering
#' 
#' 
#' @param data.trainest input data used for training and estimation, where each
#' row corresponds to an individual and columns contain information on treatments,
#' covariates, probabilities of treatment assignment, and observed outcomes.
#' @param data.validation input data used for validation with the same row and
#' column information as in `data.trainest`.
#' @param y a character string denoting the column name of outcomes.
#' @param id a character string denoting the column name of IDs.
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
#' leaf-wise weighted averages based on Wu and Gartsch (2018). The default value is `TRUE`.
#' @param nodesize minimum number of observations in a terminal node. The default value is 5.
#' @param ntree number of trees to grow in the forest. This should not be set to
#' too small a number. The default value is 1000.
#' @param prop.train proportion of data used for training in `data.trainest`.
#' The default value is 0.5.
#' @param epi threshold for minimal welfare gain in terms of the empirical standard
#' deviation of the overall outcome `y`. The default value is 0.1.
#' @param resid if `TRUE`, we implement the arbitrary residualization so the algorithm can choose baseline function that reduce the variance of the outcome. The default value is `TRUE`.
#' @param clus.tree.growing if `TRUE`, the algorithm should perform tree growing based on clustering. The default value is `FALSE`.
#' @param clus.outcome.avg if `TRUE`, the algorithm should calculate the average
#' outcome within each cluster. This option is deprecated. The default value is `FALSE`.
#' @param clus.max control the maximum number of clusters when performing k-means clustering.
#' It should be greater than 1 and less than or equal to the number of unique treatments `length(trts)`.
#' The default value is 10.
#' @param reg if `TRUE`, we grow the regularized version of the joint assignment forest. 
#' This parameter is passed to the function within `rjaf_cpp` function in the `rjaf.cpp` file.
#' @param impute if `TRUE`, imputation is used to grow tree under regularization. This parameter is passed to the `rjaf_cpp` function in the `rjaf.cpp` file. 
#' @param setseed if `TRUE`,  value of `seed` is passed to `rjaf.cpp` file and function `set_seed()` that sets the random seed in `R`.
#' @param seed value used to set seed in `rjaf.cpp` `set_seed()` function is setseed is `TRUE`. The default value is 1.
#' @param nfold number of folds in cross-validation to choose the combination of these parameters for each arm-“noise” setting
#' 
#' 
#' @return If both `clus.tree.growing` and `clus.outcome.avg` are true, return list containing a tibble (named as "res") with ID, cluster number, 
#' and predicted outcome, and a data frame (named as "clustering") with cluster number, probability of being assigned to the cluster, and treatment.
#' If not, return a tibble with ID, treatment predicted by the regularized joint assignment forest, predicted outcome, and corresponding treatment outcome from the validation data.

#' @export 
#' 
#' @seealso \code{\link{???}}, \code{\link{???}}, and the \code{\link{???}} function. DO WE NEED THIS?
#'
#' @examples 
#' data(Example.trainest)
#' data(Example.valid)
#' id <- "id"; trts <- as.character(0:K); y <- "Y"; trt <- "trt";  vars <- paste0("X", 1:3); prob <- "prob";
#' forest.reg1 <- rjaf(Example.trainest, Example.valid, y, id, trt, vars, prob, reg=T, clus.max = 3, clus.tree.growing = TRUE, clus.outcome.avg = TRUE)
#' forest.reg2 <- rjaf(Example.trainest, Example.valid, y, id, trt, vars, prob, reg=T, clus.max = 3, clus.tree.growing = TRUE, clus.outcome.avg = FALSE)
#' 
#' 
#' @references 
#' Wu, Edward and Johann A Gagnon-Bartsch (2018). The LOOP Estimator: Adjusting for Covariates in Randomized Experiments. Evaluation Review, 42(4):458–488.
#' 
#' \emph{Cambridge University Press}.
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
          data.onefold <- filter(data.trainest, fold==k)
          data.rest <- filter(data.trainest, fold!=k)
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
    prob.tree.growing <- pull(data.trainest, prob_cluster)
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
                  cluster=as.character(clus[ls.forest$trt.dof]),
                  !!(paste0(y, ".pred")):=as.numeric(ls.forest$Y.pred))
    return(list(res=res, clustering=df))
  } else {
    res <- tibble(!!(id):=as.character(pull(data.validation, id)),
                  !!(trt):=as.character(trts[ls.forest$trt.dof]),
                  !!(paste0(y, ".pred")):=as.numeric(ls.forest$Y.pred))
    if (all(paste0(y, trts) %in% names(data.validation))) {
      res <- data.validation %>%
        dplyr::select(all_of(c(id, paste0(y, trts)))) %>%
        pivot_longer(cols=paste0(y, trts), names_to=trt, names_prefix=y, values_to=y) %>%
        mutate(across(c(id, trt), as.character)) %>%
        inner_join(res, by=c(id, trt)) %>%
        rename_with(~str_c(.,".dof"), all_of(c(y, trt)))
    }
    return(res)
  }
}