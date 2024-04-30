#' Simulated randomized experiment data with clustering I
#' 
#' A data set where each row corresponds to an individual and columns contain
#' information on treatments, covariates, probabilities of treatment assignment,
#' counterfactual outcomes, and observed outcomes. This simulation gives more weights to X1 when determining
#' the outcome and it is the approach used in simulations performed in manuscript.
#'
#' @format A data frame with 100 rows and 21 columns 
#' \describe{
#'   \item{id}{Observation ID}
#'   \item{cl}{Assigned cluster number}
#'   \item{cid}{Cluster-specific identifier for each observation within its cluster. It is assigned based on the cluster assignment ``cl`` and the ``count`` parameter.}
#'   \item{X1}{Covariate X1}
#'   \item{X2}{Covariate X2}
#'   \item{X3}{Covariate X3}
#'   \item{prob}{Probability of treatment assignment}
#'   \item{Y}{Observed outcomes}
#'   \item{trt}{Treatment assignment for each observation}
#'   \item{Yc0t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 0 and is assigned to treatment 1}
#'   \item{Yc1t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 1 and is assigned to treatment 1}
#'   \item{Yc2t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 2 and is assigned to treatment 1}
#'   \item{Yc3t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 3 and is assigned to treatment 1}
#'   \item{Yc4t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 4 and is assigned to treatment 1}
#'   \item{Yc5t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 5 and is assigned to treatment 1}
#'   \item{Y0}{Counterfactual outcome for the scenario where an observation is assigned to treatment 0}
#'   \item{Y1}{Counterfactual outcome for the scenario where an observation is assigned to treatment 1}
#'   \item{Y2}{Counterfactual outcome for the scenario where an observation is assigned to treatment 2}
#'   \item{Y3}{Counterfactual outcome for the scenario where an observation is assigned to treatment 3}
#'   \item{Y4}{Counterfactual outcome for the scenario where an observation is assigned to treatment 4}
#'   \item{Y5}{Counterfactual outcome for the scenario where an observation is assigned to treatment 5}
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
#'   \item{X1}{Covariate X1}
#'   \item{X2}{Covariate X2}
#'   \item{X3}{Covariate X3}
#'   \item{prob}{Probability of treatment assignment}
#'   \item{Y}{Observed outcomes}
#'   \item{trt}{Treatment assignment for each observation}
#'   \item{Yc0t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 0 and is assigned to treatment 1}
#'   \item{Yc1t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 1 and is assigned to treatment 1}
#'   \item{Yc2t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 2 and is assigned to treatment 1}
#'   \item{Yc3t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 3 and is assigned to treatment 1}
#'   \item{Yc4t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 4 and is assigned to treatment 1}
#'   \item{Yc5t1}{Counterfactual outcome for the scenario where an observation belongs to cluster 5 and is assigned to treatment 1}
#'   \item{Y0}{Counterfactual outcome for the scenario where an observation is assigned to treatment 0}
#'   \item{Y1}{Counterfactual outcome for the scenario where an observation is assigned to treatment 1}
#'   \item{Y2}{Counterfactual outcome for the scenario where an observation is assigned to treatment 2}
#'   \item{Y3}{Counterfactual outcome for the scenario where an observation is assigned to treatment 3}
#'   \item{Y4}{Counterfactual outcome for the scenario where an observation is assigned to treatment 4}
#'   \item{Y5}{Counterfactual outcome for the scenario where an observation is assigned to treatment 5}
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
#'   \item{trt}{Treatment assigned for each observation}
#'   \item{X1}{Covariate X1}
#'   \item{X2}{Covariate X2}
#'   \item{X3}{Covariate X3}
#'   \item{prob}{Probability of treatment assignment}
#'   \item{Y}{Observed outcomes}
#'   \item{Y0}{Counterfactual outcome for the scenario where an observation is assigned to treatment 0}
#'   \item{Y1}{Counterfactual outcome for the scenario where an observation is assigned to treatment 1}
#'   \item{Y2}{Counterfactual outcome for the scenario where an observation is assigned to treatment 2}
#'   \item{Y3}{Counterfactual outcome for the scenario where an observation is assigned to treatment 3}
#'   \item{Y4}{Counterfactual outcome for the scenario where an observation is assigned to treatment 4}
#'   \item{Y5}{Counterfactual outcome for the scenario where an observation is assigned to treatment 5}
#' }
"Example_data_non_clus"