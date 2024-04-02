#' Simulated randomized experiment data with clustering I
#' 
#' A data set where each row corresponds to an individual and columns contain information on treatments,
#' covariates, probabilities of treatment assignment, and observed outcomes. 
#' This simulation give more weights to X1 when determining the outcome and it is the approach used in simulations performed in manuscript.
#'
#' @format A data frame with 100 rows and 21 columns 
#' \describe{
#'   \item{id}{Bbservation ID}
#'   \item{cl}{Assigned cluster number}
#'   \item{cid}{Cluster-specific identifier for each observation within its cluster. It is assigned based on the cluster assignment ``cl`` and the ``count`` parameter.}
#'   \item{X1...X3}{Covariates with each observation}
#'   \item{prob}{Probability of treatment assignment}
#'   \item{Y}{Observed outcomes}
#'   \item{trt}{Treatment assignment for each observation}
#'   \item{Yc0t1...Yc5t1}{Counterfactual outcome for the scenario where an observation belongs to cluster ``c`` and is assigned to treatment ``t``.}
#'   \item{Y0...Y5}{Counterfactual outcome for the scenario where an observation is assigned to a specific treatment}
#' }
"Example_data"



#' Simulated randomized experiment data with clustering II
#' 
#' A data set where each row corresponds to an individual and columns contain information on treatments,
#' covariates, probabilities of treatment assignment, and observed outcomes
#'
#' @format A data frame with 100 rows and 21 columns 
#' \describe{
#'   \item{id}{Bbservation ID}
#'   \item{cl}{Assigned cluster number}
#'   \item{cid}{Cluster-specific identifier for each observation within its cluster. It is assigned based on the cluster assignment ``cl`` and the ``count`` parameter.}
#'   \item{X1...X3}{Covariates with each observation}
#'   \item{prob}{Probability of treatment assignment}
#'   \item{Y}{Observed outcomes}
#'   \item{trt}{Treatment assignment for each observation}
#'   \item{Y0...Y5}{Counterfactual outcome for the scenario where an observation is assigned to a specific treatment}
#' }
"Example_data_clus"


#' Simulated randomized experiment data without clustering
#' 
#' A data set where each row corresponds to an individual and columns contain information on treatments,
#' covariates, probabilities of treatment assignment, and observed outcomes
#'
#' @format A data frame with 100 rows and 13 columns 
#' \describe{
#'   \item{id}{Bbservation ID}
#'   \item{cl}{Assigned cluster number}
#'   \item{cid}{Cluster-specific identifier for each observation within its cluster. It is assigned based on the cluster assignment ``cl`` and the ``count`` parameter.}
#'   \item{X1...X3}{Covariates with each observation}
#'   \item{prob}{Probability of treatment assignment}
#'   \item{Y}{Observed outcomes}
#'   \item{trt}{Treatment assignment for each observation}
#'   \item{Y0...Y5}{Counterfactual outcome for the scenario where an observation is assigned to a specific treatment}
#' }
"Example_data_non_clus"



