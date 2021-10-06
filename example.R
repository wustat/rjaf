library(dplyr)
library(tibble)
library(stringr)
library(magrittr)
library(readr)
library(tidyr)
library(randomForest)
library(ranger)
library(git2r)
name_ssh_key <- "id_rsa"
creds <- cred_ssh_key(ssh_path(paste0(name_ssh_key, ".pub")),
                      ssh_path(name_ssh_key))
url.remote <- "git@github.com:wustat/dof.git"
relpath2local <- "your/path/to/dof"
abspath2local <- file.path(Sys.getenv("HOME"), relpath2local)
if (dir.exists(path2local)) {
  git2r::pull(path2local, creds)
} else {
  clone(url.remote, path2local, credentials=creds)
}
source(file.path(path2local, "library.R"))
Rcpp::sourceCpp(file.path(path2local,"library.cpp"))
n <- 10000; K <- 9; gamma <- 10; sigma <- 10
id <- "id"; trts <- as.character(0:9); y <- "Y"; trt <- "trt"
vars <- paste0("X", 1:3)
lambda <- 0.5; prob <- "prob"; ipw <- T
ntree <- 1000; ntrt <- 5; nvar <- 3
set.seed(1)
data.validation <- sim.data(n, K, gamma, sigma)
data.trainest <- sim.data(n=500, K, gamma, sigma)
# check default arguments of growForest for details
# on iMac (3.6GHz 8-Core Intel Core i9, 16GB, 2019)
system.time(forest.reg <-
              growForest(data.trainest, data.validation, y, id, trt,
                         vars, prob, ntree=ntree, ntrt=ntrt, nvar=nvar, reg=T))
# user  system elapsed 
# 4.933   0.129   4.403
system.time(forest.noreg <-
              growForest(data.trainest, data.validation, y, id, trt,
                         vars, prob, ntree=ntree, ntrt=ntrt, nvar=nvar, reg=F))
# user  system elapsed 
# 12.764   2.448  14.491