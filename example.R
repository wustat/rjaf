library(dplyr)
library(tibble)
library(stringr)
library(magrittr)
library(readr)
library(tidyr)
library(randomForest)
library(ranger)

source("library.R")
Rcpp::sourceCpp("library.cpp")
n <- 10000; K <- 9; gamma <- 10; sigma <- 10
id <- "id"; trts <- as.character(0:9); y <- "Y"; trt <- "trt"
vars <- paste0("X", 1:3)
lambda <- 0.5; prob <- "prob"; ipw <- T
ntree <- 1000; ntrt <- 5; nvar <- 3
set.seed(123)
data.validation <- sim.data(n, K, gamma, sigma)
data.trainest <- sim.data(n=500, K, gamma, sigma)
# check default arguments of growForest for details
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