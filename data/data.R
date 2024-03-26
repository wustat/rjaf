n <- 100; K <- 5; gamma <- 10; sigma <- 10
count <- rep(1, K+1)

Example_data <- sim.data(n, K, gamma, sigma, count)
saveRDS(Example_data, "data/Example_data.rda")


