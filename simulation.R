library(dplyr)
library(tibble)
library(stringr)
library(magrittr)
library(readr)
library(tidyr)
library(randomForest)
library(ranger)
library(forcats)
# enable tidyverse warnings more than once per session
options(tidyselect_verbosity="verbose")

source("library.R")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 3; gamma <- 10; sigma <- 10
count <- c(1,1,5,3)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)

cases <- expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=c(0,0.5,1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1,0.5), ntrt=10,
                     nvar=3, nodesize=3,
                     clus.tree.growing=c(T,F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 100 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("library.R")
                sourceCpp("library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)))
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
ylim.outcome <- range(res$fitted.dof)
unreg.fitted <- res %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted.dof, fill=clus.tree.growing)) +
  scale_y_continuous(limits=ylim.outcome) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(. ~ epi, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true.dof, fill=clus.tree.growing)) +
  scale_y_continuous(limits=ylim.outcome) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(. ~ epi, labeller=label_both) +
  theme_classic() + mytheme

unreg.rate <- res %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=rate.optassign, fill=clus.tree.growing)) +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(. ~ epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, unreg.rate,
          common.legend=T, ncol=3)
