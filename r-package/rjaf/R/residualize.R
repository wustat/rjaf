#' Arbitrary residualization of outcomes
#'
#'
#' This function employs random forests and cross-validation to residualize
#' outcomes following Wu and Gagnon-Bartsch (2018).
#' That is, predicted outcomes resulting from random forests are
#' subtracted from the original outcomes. Doing so helps in adjusting for small imbalanaces
#' in baseline covariates and removing part of the variation in
#' outcomes common across treatment arms
#' 
#' @param data input data used for training and estimation, where each
#' row corresponds to an individual and columns contain information on treatments,
#' covariates, probabilities of treatment assignment, and observed outcomes.
#' @param y a character string denoting the column name of outcomes.
#' @param vars a vector of character strings denoting the column names of covariates.
#' @param nfold number of folds in cross-validation. The default value is 5.
#' @param fun.rf a character string specifying which random forest package to use.
#' Two options are `ranger` and `randomForest`, with the default being `ranger`.
#' 
#' @return data for training and estimation with residualized outcomes.
#' @export 
#'
#' @examples 
#' data(Example_data)
#' library(dplyr)
#' library(magrittr)
#' Example_trainest <- Example_data %>% slice_sample(n = floor(0.5 * nrow(Example_data)))
#' y <- "Y"
#' vars <- paste0("X", 1:3)
#' Example_resid <- residualize(Example_trainest, y, vars, nfold = 5, fun.rf = "ranger")
#' 
#' @references 
#' Wu, Edward and Johann A Gagnon-Bartsch (2018). The LOOP Estimator: Adjusting
#' for Covariates in Randomized Experiments. Evaluation Review, 42(4):458–488.
#' \cr
#' 

residualize <- function(data, y, vars, nfold=5, fun.rf="ranger") {
  fold <- sample(1:nfold, NROW(data), TRUE, rep(1, nfold))
  data <- data %>% mutate(fold=fold)
  if (fun.rf=="randomForest") {
    lapply(1:nfold, function(i) {
      data.i <- data %>% filter(fold==i)
      data.i[, paste0(y, ".resid")] <- data.i[,y] -
        predict(randomForest::randomForest(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          data %>% filter(fold!=i)), data.i)
      data.i}) %>%
      bind_rows %>%
      dplyr::select(-fold)
  } else if (fun.rf=="ranger") {
    lapply(1:nfold, function(i) {
      data.i <- data %>% filter(fold==i)
      data.i[, paste0(y, ".resid")] <- data.i[,y] -
        predict(ranger::ranger(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          data %>% filter(fold!=i)), data.i)[["predictions"]]
      data.i}) %>%
      bind_rows %>%
      dplyr::select(-fold)
  }
}
