#' Arbitrary residualization of outcomes
#'
#'
#' This function employs random forests and cross-validation to residualize
#' outcomes. That is, predicted outcomes resulting from random forests are
#' subtracted from the original outcomes. Doing so, part of the variation in
#' outcomes common across treatment arms can be removed.
#' 
#' @param data input data used for training and estimation, where each
#' row corresponds to an individual and columns contain information on treatments,
#' covariates, probabilities of treatment assignment, and observed outcomes.
#' @param y a character string indicating the column name of outcomes.
#' @param vars a vector of character strings indicating the column names of covariates.
#' @param nfold number of folds in cross-validation. The default value is 5.
#' @param fun.rf a character string specifying which random forest package to use.
#' Two options are `ranger` and `randomForest`, with the default being `ranger`.
#' 
#' @return data for training and estimation with residualized outcomes.
#' @export 
#'
#' @examples 
#' \dontrun{
#' data(Example_data)
#' Example_trainest <- Example_data %>% slice_sample(n = floor(0.3 * nrow(Example_data)))
#' y <- "Y"
#' vars <- paste0("X", 1:3)
#' Example_resid <- residualize(Example_trainest, y, vars, nfold = 5, fun.rf = "ranger")
#' }
#'
#' 

residualize <- function(data, y, vars, nfold=5, fun.rf="ranger") {
  data$fold <- sample(1:nfold, NROW(data), T, rep(1, nfold))
  if (fun.rf=="randomForest") {
    dplyr::select(bind_rows(lapply(1:nfold, function(i) {
      data.i <- filter(data, data$fold==i)
      data.i[,y] <- data.i[,y] -
        predict(randomForest::randomForest(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          filter(data, data$fold!=i)), data.i)
      data.i})), -.data$fold)
  } else if (fun.rf=="ranger") {
    dplyr::select(bind_rows(lapply(1:nfold, function(i) {
      data.i <- filter(data, data$fold==i)
      data.i[,y] <- data.i[,y] -
        predict(ranger::ranger(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          filter(data, data$fold!=i)), data.i)[["predictions"]]
      data.i})), -.data$fold)
  }
}
