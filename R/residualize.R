#' Implementation of arbitrary residualization to replace the outcome with residuals
#'
#'
#' Some outcome variation is shared among treatment arms. If we can predict this shared variation, we can reduce the variability in evaluating various treatment arms by subtracting it. 
#' We include a pre-processing step in our algorithm to create residualized outcomes. To implement, we fit the baseline prediction function using cross-validation to avoid overfitting. We calculate residuals by subtracting the predicted values from the observed outcomes. 
#' 
#' @param data input training and estimation data from parameter `data.trainest` in `rjaf.R`.
#' @param y a character string stands for the name of outcomes, as usually used to pull out the outcome column in training, estimation, and validation data.
#' @param vars a vector of strings stands for names of covariates, as usually used to pull out covariates column in training estimation, and validation data. 
#' @param nfold number of folds in cross-validation to conduct prediction through the random forest. The default value is 5.
#' @param fun.rf specifies which random forest package to be used (`ranger` or  `randomForest`). The default value is `ranger`.
#' 
#' @return a modified training and estimation dataset with the outcome replaced by residuals from subtracting the predicted outcome from the original outcome.
#' @export 
#'
#' @examples 
#' data(Example.trainest) ##TO DO: need to regenerate this to add est data (name). 
#' y <- "Y"
#' vars <- paste0("X", 1:3)
#' Example_resid <- residualize(Example.trainest, y, vars, nfold = 5, fun.rf = "ranger")
#' 
#' 
#' 
#' \emph{Cambridge University Press}.
#' \cr
#' 

residualize <- function(data, y, vars, nfold=5, fun.rf="ranger") {
  data$fold <- sample(1:nfold, NROW(data), T, rep(1, nfold))
  if (fun.rf=="randomForest") {
    dplyr::select(bind_rows(lapply(1:nfold, function(i) {
      data.i <- filter(data, fold==i)
      data.i[,y] <- data.i[,y] -
        predict(randomForest::randomForest(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          filter(data, fold!=i)), data.i)
      data.i})), -fold)
  } else if (fun.rf=="ranger") {
    dplyr::select(bind_rows(lapply(1:nfold, function(i) {
      data.i <- filter(data, fold==i)
      data.i[,y] <- data.i[,y] -
        predict(ranger::ranger(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          filter(data, fold!=i)), data.i)[["predictions"]]
      data.i})), -fold)
  }
}
