library(dplyr)
library(tibble)
library(stringr)
library(magrittr)
library(readr)
library(tidyr)
library(randomForest)
library(ranger)

name_ssh_key <- "id_rsa"
creds <- git2r::cred_ssh_key(
  git2r::ssh_path(paste0(name_ssh_key, ".pub")),
  git2r::ssh_path(name_ssh_key))
url.remote <- "git@github.com:wustat/dof.git"
relpath2local <- "your/path/to/dof"
abspath2local <- file.path(Sys.getenv("HOME"), relpath2local)
if (dir.exists(abspath2local)) {
  git2r::pull(abspath2local, creds)
} else {
  git2r::clone(url.remote, abspath2local, credentials=creds)
}
source(file.path(abspath2local, "library.R"))
Rcpp::sourceCpp(file.path(abspath2local,"library.cpp"))

id <- "id"; trts <- as.character(0:9); y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 9; gamma <- 10; sigma <- 10
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma)
cases <- expand_grid(n.trainest=1000, ntree=c(1000, 2000),
                     lambda=c(0, 0.5, 1, 2), epi=c(0.1, 0.5),
                     ntrt=c(3, 5, 10), nvar=3, nodesize=c(3, 5))
n.sim <- 1000 # number of simulated trainest datasets
noexport <- c("growTree_cpp", "growForest_cpp", "splitting_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 100 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
res <- 
  apply(cases, 1,
        function(case, data.validation, K, gamma, sigma, n.sim,
                 noexport, pkgs, id, trts, y, vars, trt, prob) {
          names.case <- names(case)
          for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
          raw <- foreach::foreach(
            i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
            .noexport=noexport) %dopar% {
              relpath2local <- "your/path/to/dof"
              abspath2local <- file.path(Sys.getenv("HOME"), relpath2local)
              source(file.path(abspath2local, "library.R"))
              Rcpp::sourceCpp(file.path(abspath2local,"library.cpp"))
              data.trainest <- sim.data(n.trainest, K, gamma, sigma)
              growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                         vars=vars, prob=prob,
                         ntrt=ntrt, nvar=nvar, lambda=lambda, nodesize=nodesize,
                         ntree=ntree, epi=epi) %>%
                inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                dplyr::summarise(
                  rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                        !!sym(paste0(trt,".dof"))),
                  fitted.dof=mean(!!sym(paste0(y,".pred"))),
                  true.oracle=mean(!!sym(paste0(y,".oracle"))),
                  true.dof=mean(!!sym(paste0(y,".dof"))),
                  true.random=mean(!!sym(y)))
            }
          bind_cols(as_tibble_row(case),
                    raw %>% dplyr::summarise(across(everything(), mean)))
        }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
        n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id, trts=trts,
        y=y, vars=vars, trt=trt, prob=prob) %>% bind_rows
parallel::stopCluster(cl)