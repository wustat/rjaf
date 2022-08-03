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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 3; gamma <- 10; sigma <- 10
count <- c(1,11,15,13)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

# n.sim <- 10 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                # df.cf <- lapply(2:length(trts), function(i) {
                #   data.tmp <- data.tre %>% filter(!!sym(trt)%in%c(1, i))
                #   grf::causal_forest(
                #     data.tmp %>% select(all_of(vars)) %>% as.matrix,
                #     data.tmp %>% pull(y),
                #     data.tmp %>% pull(trt), num.trees=ntree,
                #     mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                #     predict(data.val %>% select(all_of(vars)) %>% as.matrix)
                # }) %>% do.call(cbind, .) %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                    data.tmp, num.trees=ntree,
                    mtry=nvar, min.node.size=nodesize) %>%
                    predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)

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
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_40trt_1_11_15_13_",Sys.Date(),".RData"))
parallel::stopCluster(cl)
# clacc <- function(df) {
#   mean(as.integer(as_factor(df$cluster)) ==
#     as.integer(forcats::as_factor(gsub("(c\\d*)(t\\d*)", "\\1", df$trt))))
# }
# clacc(tmp$clustering)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(clus.tree.growing ~ epi, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(clus.tree.growing ~ epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_noclusoutcomeavg_multi_arm_cf.pdf",
       width=10, height=8)
# unreg.true <- res %>%
#   mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
#   ggplot(aes(x=lambda1, y=true.dof, fill=clus.tree.growing)) +
#   scale_y_continuous(limits=ylim.outcome) +
#   geom_hline(yintercept=11.5, linetype="dotted") +
#   ggtitle("unregularized") +
#   geom_boxplot() + facet_grid(. ~ epi, labeller=label_both) +
#   theme_classic() + mytheme

# unreg.rate <- res %>%
#   mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
#   ggplot(aes(x=lambda1, y=rate.optassign, fill=clus.tree.growing)) +
#   ggtitle("unregularized") +
#   geom_boxplot() + facet_grid(. ~ epi, labeller=label_both) +
#   theme_classic() + mytheme
# 
# ggarrange(unreg.fitted, unreg.true, unreg.rate,
#           common.legend=T, ncol=3)

# reg with clus.outcome.avg=F
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 3; gamma <- 10; sigma <- 10
count <- c(1,11,15,13)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                # df.cf <- lapply(2:length(trts), function(i) {
                #   data.tmp <- data.tre %>% filter(!!sym(trt)%in%c(1, i))
                #   grf::causal_forest(
                #     data.tmp %>% select(all_of(vars)) %>% as.matrix,
                #     data.tmp %>% pull(y),
                #     data.tmp %>% pull(trt), num.trees=ntree,
                #     mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                #     predict(data.val %>% select(all_of(vars)) %>% as.matrix)
                # }) %>% do.call(cbind, .) %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
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
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_40trt_1_11_15_13_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ lambda2 + clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ lambda2 + clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

# reg.rate <- res %>%
#   mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
#   ggplot(aes(x=lambda1, y=rate.optassign, fill=clus.tree.growing)) +
#   ggtitle("regularized") +
#   geom_boxplot() + facet_grid(impute ~ lambda2, labeller=label_both) +
#   theme_classic() + mytheme
# user    system   elapsed 
# 64.043   177.470 20102.437

ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_noclusoutcomeavg_multi_arm_cf.pdf",
       width=18, height=12)


# unreg with clus.outcome.avg=T
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 3; gamma <- 10; sigma <- 10
count <- c(11,11,15,13)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)

cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=T, clus.outcome.avg=T)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

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
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg)$res %>%
                  dplyr::summarise(
                    fitted.dof=mean(!!sym(paste0(y,".pred")))
                    )
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
ylim.outcome <- range(res$fitted.dof)
unreg.fitted <- res %>%
  mutate(across(c(lambda1, epi, clus.outcome.avg), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted.dof, fill=clus.outcome.avg)) +
  scale_y_continuous(limits=ylim.outcome) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(. ~ epi, labeller=label_both) +
  theme_classic() + mytheme


save(res, file=paste0("results/simulations_unreg_50trt_11_11_15_13_clus.outcome.avg=T_",Sys.Date(),".RData"))


# reg with clus.outcome.avg=T
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 3; gamma <- 10; sigma <- 10
count <- c(11,11,15,13)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)

cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=T, clus.outcome.avg=T)


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
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg)$res %>%
                  dplyr::summarise(
                    fitted.dof=mean(!!sym(paste0(y,".pred")))
                  )
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))

save(res, file=paste0("results/simulations_reg_50trt_11_11_15_13_clus.outcome.avg=T_",Sys.Date(),".RData"))
ylim.outcome <- range(res$fitted.dof)
reg.fitted <- res %>%
  mutate(across(c(lambda1, lambda2, clus.outcome.avg), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted.dof, fill=lambda2)) +
  scale_y_continuous(limits=ylim.outcome) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() + facet_grid(.~impute, labeller=label_both) +
  theme_classic() + mytheme











# no clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

# n.sim <- 10 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                # df.cf <- lapply(2:length(trts), function(i) {
                #   data.tmp <- data.tre %>% filter(!!sym(trt)%in%c(1, i))
                #   grf::causal_forest(
                #     data.tmp %>% select(all_of(vars)) %>% as.matrix,
                #     data.tmp %>% pull(y),
                #     data.tmp %>% pull(trt), num.trees=ntree,
                #     mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                #     predict(data.val %>% select(all_of(vars)) %>% as.matrix)
                # }) %>% do.call(cbind, .) %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
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
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_30trt_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_30trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)




# reg with no clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

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
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                # df.cf <- lapply(2:length(trts), function(i) {
                #   data.tmp <- data.tre %>% filter(!!sym(trt)%in%c(1, i))
                #   grf::causal_forest(
                #     data.tmp %>% select(all_of(vars)) %>% as.matrix,
                #     data.tmp %>% pull(y),
                #     data.tmp %>% pull(trt), num.trees=ntree,
                #     mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                #     predict(data.val %>% select(all_of(vars)) %>% as.matrix)
                # }) %>% do.call(cbind, .) %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
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
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_30trt_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

# reg.rate <- res %>%
#   mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
#   ggplot(aes(x=lambda1, y=rate.optassign, fill=clus.tree.growing)) +
#   ggtitle("regularized") +
#   geom_boxplot() + facet_grid(impute ~ lambda2, labeller=label_both) +
#   theme_classic() + mytheme
# user    system   elapsed 
# 64.043   177.470 20102.437

ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_notrueclustering_30trt_multi_arm_cf_clusvsnoclus.pdf",
       width=18, height=12)







# trt=100
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=40,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
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
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_100trt_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_100trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)




# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=40,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
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
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_100trt_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_notrueclustering_100trt_multi_arm_cf_clusvsnoclus.pdf",
       width=18, height=12)







# March 11, 2022
##  sd for 30 trt
# no clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=3000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=sd(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=sd(!!sym(y))) %>% pull(true.rf)
                
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
                    fitted.dof=sd(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=sd(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=sd(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=sd(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_30trt_notrueclustering_clusvsnoclus_stddev",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <-
  res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(med=median(fitted),
            lower=quantile(fitted, 0.025),
            upper=quantile(fitted, 0.975), .groups="drop") %>%
  ggplot(aes(x=method, y=med, ymin=lower, ymax=upper, color=method)) +
  scale_y_continuous(name="std dev (fitted)") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~ epi+lambda1, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(med=median(true),
            lower=quantile(true, 0.025),
            upper=quantile(true, 0.975), .groups="drop") %>%
  ggplot(aes(x=method, y=med, ymin=lower, ymax=upper, color=method)) +
  scale_y_continuous(name="std dev (true)") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~ epi+lambda1, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_30trt_multi_arm_cf_clusvsnoclus_stddev.pdf",
       width=15, height=8)




# reg with no clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=3000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=sd(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=sd(!!sym(y))) %>% pull(true.rf)
                
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
                    fitted.dof=sd(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=sd(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=sd(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=sd(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_30trt_notrueclustering_clusvsnoclus_stddev",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, impute, clus.tree.growing, method) %>%
  summarise(med=median(fitted),
            lower=quantile(fitted, 0.025),
            upper=quantile(fitted, 0.975), .groups="drop") %>%
  ggplot(aes(x=method, y=med, ymin=lower, ymax=upper, color=method)) +
  scale_y_continuous(name="std dev (fitted)") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, impute, clus.tree.growing, method) %>%
  summarise(med=median(true),
            lower=quantile(true, 0.025),
            upper=quantile(true, 0.975), .groups="drop") %>%
  ggplot(aes(x=method, y=med, ymin=lower, ymax=upper, color=method)) +
  scale_y_continuous(name="std dev (true)") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_notrueclustering_30trt_multi_arm_cf_clusvsnoclus_stddev.pdf",
       width=18, height=12)




# March 27, 2022
# trt=10
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 9; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=3000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_10trt_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true.CI <- res %>% select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~lambda1+epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_10trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)
res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle), .groups="drop") %>%
  rename(clus=clus.tree.growing)



# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 9; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=3000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_10trt_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_notrueclustering_10trt_multi_arm_cf_clusvsnoclus.pdf",
       width=18, height=12)
res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle), .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)






# March 28, 2022
# trt=30
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=3000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=20,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_30trt_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true.CI <-
  res %>% select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~lambda1+epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_30trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)




# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=3000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=20,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_30trt_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_notrueclustering_30trt_multi_arm_cf_clusvsnoclus.pdf",
       width=18, height=12)







# March 31, 2022
# trt=100
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=3000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_100trt_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true.CI <-
  res %>% select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~lambda1+epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_30trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)




# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=3000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_100trt_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme
  


ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_notrueclustering_100trt_multi_arm_cf_clusvsnoclus.pdf",
       width=18, height=12)






# Apr 18, 2022
# trt=30, ntrt=30
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_30trt_30ntrt_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true.CI <-
  res %>% select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~lambda1+epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_30trt_30ntrt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle), .groups="drop") %>%
  rename(clus=clus.tree.growing)




# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)
# %>%
#   bind_rows(expand_grid(n.sim=100, n.trainest=1000, ntree=1000, lambda1=0.5,
#                         lambda2=c(0.5, 1), reg=T, impute=c(T,F),
#                         epi=0.5, ntrt=10, nvar=3, nodesize=3,
#                         clus.tree.growing=T, clus.outcome.avg=F))

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_30trt_30ntrt_nodesize3_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_notrueclustering_30trt_30ntrt_multi_arm_cf_clusvsnoclus.pdf",
       width=18, height=12)
res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle), .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)


# Apr 20, 2022
# trt=100
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=80,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_100trt_ntrt80_ntrain5000_nodesize3_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true.CI <-
  res %>% select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~lambda1+epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_100trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)




# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=80,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_100trt_ntrt80_ntrain5000_nodesize3_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/regularized_notrueclustering_100trt_multi_arm_cf_clusvsnoclus.pdf",
       width=18, height=12)





# Apr 24, 2022
# trt=100
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=100,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_100trt_ntrt100_ntrain5000_nodesize3_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true.CI <-
  res %>% select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~lambda1+epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_100trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle), .groups="drop") %>%
  rename(clus=clus.tree.growing)




# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=100,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y))) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
            # dplyr::summarise(across(everything(), mean))
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_100trt_ntrt100_ntrain5000_nodesize3_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)

res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle), .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)






# Apr 30, 2022
# trt=50
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 49; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=40,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_50trt_ntrt40_ntrain5000_nodesize3_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true.CI <-
  res %>% select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~lambda1+epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_50trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)




# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 49; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=40,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_50trt_ntrt40_ntrain5000_nodesize3_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle","globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)

res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)








# May 1, 2022
# trt=50
# unreg with or without clustering
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 49; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=50,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_3methods_unreg_multi_arm_cf_50trt_ntrt50_ntrain5000_nodesize3_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true.CI <-
  res %>% select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, epi, clus.tree.growing, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("unregularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing~lambda1+epi, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)
ggsave("~/K/Users/kecc-wenbowu/RWJF/results/unregularized_notrueclustering_50trt_multi_arm_cf_clusvsnoclus.pdf",
       width=10, height=8)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)




# reg with and without clustering
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 49; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=50,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_50trt_ntrt50_ntrain5000_nodesize3_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle","globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)

res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)





# May 16, 2022
# sigma=10, gamma=20
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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 9; gamma <- 20; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

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
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_10trt_ntrt10_ntrain5000_nodesize3_gamma20_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)





id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 20; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

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
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_30trt_ntrt30_ntrain5000_nodesize3_gamma20_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)






id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 49; gamma <- 20; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=50,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

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
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_50trt_ntrt50_ntrain5000_nodesize3_gamma20_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)









id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 20; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=100,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_100trt_ntrt100_ntrain5000_nodesize3_gamma20_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)

id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 49; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=50,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_50trt_ntrt50_ntrain5000_nodesize3_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle","globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)

res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)



# May 25, 2022

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

setwd("~/K/Users/kecc-wenbowu/RWJF/")
source("code/library.R")
# source('archive/Direct_Opt_Forest_Regularize.R')
# Rcpp::sourceCpp("code/library.cpp")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 9; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

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
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_10trt_ntrt10_ntrain5000_nodesize3_gamma10_sigma20_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)


id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 9; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=10,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_10trt_ntrt10_ntrain5000_nodesize3_gamma10_sigma20_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle","globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)

res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)





id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

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
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_30trt_ntrt30_ntrain5000_nodesize3_gamma10_sigma20_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)


id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 29; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=30,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_30trt_ntrt30_ntrain5000_nodesize3_gamma10_sigma20_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle","globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)

res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)









id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 49; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=50,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

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
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_50trt_ntrt50_ntrain5000_nodesize3_gamma10_sigma20_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)


id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 49; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=50,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_50trt_ntrt50_ntrain5000_nodesize3_gamma10_sigma20_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle","globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)

res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)





id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=100,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_100trt_ntrt100_ntrain5000_nodesize3_gamma10_sigma20_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)

id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(0, 0.5, 1),
                     lambda2=c(0.5, 1), reg=T, impute=c(T, F), epi=0.5, ntrt=50,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

n.sim <- 100 # number of simulated trainest datasets
noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg,
                           clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob, trts=trts,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("results/simulations_reg_multi_arm_cf_100trt_ntrt100_ntrain5000_nodesize3_gamma10_sigma20_notrueclustering_clusvsnoclus_",
                             Sys.Date(),".RData"))

reg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme

reg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle","globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  ggtitle("regularized") +
  geom_boxplot() +
  facet_grid(impute ~ clus.tree.growing + lambda2, labeller=label_both) +
  theme_classic() + mytheme


reg.true.CI <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, lambda2, clus.tree.growing), as_factor)) %>%
  group_by(lambda1, lambda2, clus.tree.growing, impute, method) %>%
  summarise(mean=median(true),
            lower=mean-1.96*sd(true),
            upper=pmin(mean+1.96*sd(true), 11.5), .groups="drop") %>%
  ggplot(aes(x=method, y=mean, ymin=lower, ymax=upper, color=method)) +
  geom_hline(yintercept=11.5, linetype="dotted") +
  scale_y_continuous(name="true") +
  ggtitle("regularized") +
  geom_errorbar(width=0.12, size=0.5) +
  facet_grid(clus.tree.growing+impute~lambda1+lambda2, labeller=label_both) +
  theme_classic() + mytheme



ggarrange(reg.fitted, reg.true, common.legend=T)

res %>% group_by(clus.tree.growing, impute, lambda1, lambda2) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing) %>% print(n=24)


# June 3, 2022
# look for best cases for 100 trts

id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(1,5),
                     lambda2=0, reg=F, impute=F, epi=c(0.1, 0.5), ntrt=80,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_100trt_ntrt80_ntrain5000_nodesize3_gamma10_sigma10_lambda1_1_5_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)






id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 99; gamma <- 10; sigma <- 10
count <- rep(1, K+1)
n.sim <- 100
set.seed(123)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
cases <- expand_grid(n.sim=100, n.trainest=5000, ntree=1000, lambda1=c(1,5),
                     lambda2=0, reg=F, impute=F, epi=c(1, 2), ntrt=80,
                     nvar=3, nodesize=3, clus.tree.growing=c(T,F), clus.outcome.avg=F)

noexport <- c("growTree_cpp")
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cores <- 50 # number of cores in parallel
cl <- parallel::makeCluster(cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.sim,
                   noexport, pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.sim, .packages=pkgs, .combine=dplyr::bind_rows,
              .noexport=noexport) %dopar% {
                source("~/K/Users/kecc-wenbowu/RWJF/code/library.R")
                sourceCpp("~/K/Users/kecc-wenbowu/RWJF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize, honesty=F) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                true.cf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y))) %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                true.rf <- 
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), trt, y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y))) %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign=mean(!!sym(paste0(trt,".oracle"))==
                                          !!sym(paste0(trt,".dof"))),
                    fitted.dof=mean(!!sym(paste0(y,".pred"))),
                    true.oracle=mean(!!sym(paste0(y,".oracle"))),
                    true.dof=mean(!!sym(paste0(y,".dof"))),
                    true.random=mean(!!sym(y)),
                    true.globest=data.validation %>%
                      pull(data.validation %>%
                             summarise(across(starts_with(paste0(y,"c")), mean)) %>%
                             which.max %>% names) %>% mean) %>%
                  mutate(fitted.cf=mean(apply(df.cf, 1, max)),
                         true.cf=true.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.sim=n.sim, noexport=noexport, pkgs=pkgs, id=id,
          y=y, vars=vars, trt=trt, prob=prob,
          clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, file=paste0("results/simulations_unreg_multi_arm_cf_100trt_ntrt80_ntrain5000_nodesize3_gamma10_sigma10_lambda1_1_5_notrueclustering_clusvsnoclus",Sys.Date(),".RData"))
parallel::stopCluster(cl)

library(ggplot2)
library(ggpubr)
mytheme <- theme(axis.title=element_text(size=16),
                 axis.text=element_text(size=16),
                 text=element_text(size=14),
                 legend.text=element_text(size=16))
unreg.fitted <- res %>%
  select(-starts_with(c("true","rate"))) %>%
  pivot_longer(cols=starts_with("fitted"),
               names_to="method", values_to="fitted", names_prefix="fitted.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=fitted, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

unreg.true <- res %>%
  select(-starts_with(c("fitted","rate"))) %>%
  select(-ends_with(c("random", "oracle", "globest"))) %>%
  pivot_longer(cols=starts_with("true"),
               names_to="method", values_to="true", names_prefix="true.") %>%
  mutate(across(c(lambda1, epi, clus.tree.growing), as_factor)) %>%
  ggplot(aes(x=lambda1, y=true, fill=method)) +
  geom_hline(yintercept=res %>% pull(true.oracle) %>% mean, linetype="dotted") +
  ggtitle("unregularized") +
  geom_boxplot() + facet_grid(epi ~ clus.tree.growing, labeller=label_both) +
  theme_classic() + mytheme

ggarrange(unreg.fitted, unreg.true, common.legend=T)

res %>% group_by(clus.tree.growing, epi, lambda1) %>%
  summarise(mean.cf=mean(true.cf),
            mean.dof=mean(true.dof),
            mean.rf=mean(true.rf),
            median.cf=median(true.cf),
            median.dof=median(true.dof),
            median.rf=median(true.rf),
            sd.cf=sd(true.cf),
            sd.dof=sd(true.dof),
            sd.rf=sd(true.rf),
            mean.random=mean(true.random),
            mean.oracle=mean(true.oracle),
            mean.globest=mean(true.globest),
            .groups="drop") %>%
  rename(clus=clus.tree.growing)