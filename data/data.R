# PLEASE REMOVE this file when done with proofreading

source("R/sim.data.R")
n.validation <- 50; n.trainest <- 25; K <- 5; gamma <- 10; sigma <- 10
count <- rep(1, K+1)

Example.valid <- sim.data(n.validation, K, gamma, sigma, count)
Example.trainest <- sim.data(n.trainest, K, gamma, sigma, count)

saveRDS(Example.trainest, "data/Example.trainest.rda")
saveRDS(Example.valid, "data/Example.valid.rda")

