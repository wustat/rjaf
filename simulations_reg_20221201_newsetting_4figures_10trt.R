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
source("/gpfs/home/ww857/Projects/Ladhania/DOF/code/library.R")
id <- "id"; y <- "Y"
vars <- paste0("X", 1:3); trt <- "trt"; prob <- "prob"
n.validation <- 10000; K <- 9; gamma <- 10; sigma <- 20
count <- rep(1, K+1)
n.cores <- as.integer(Sys.getenv('SLURM_CPUS_PER_TASK')) # number of cores assigned
n.sim <- n.cores * as.integer(Sys.getenv('SLURM_ARRAY_TASK_COUNT'))
set.seed(857)
data.validation <- sim.data(n.validation, K, gamma, sigma, count)
trts <- data.validation %>% pull(trt) %>% table %>% names
pkgs <- c("dplyr", "tibble", "stringr", "magrittr", "readr", "tidyr",
          "randomForest", "ranger", "Rcpp", "RcppArmadillo")
cases <- expand_grid(n.sim=n.sim, n.trainest=5000, ntree=1000, lambda1=c(0),
                     lambda2=c(0.5), reg=T, impute=c(T), epi=0.5, ntrt=K+1,
                     nvar=3, nodesize=3, clus.tree.growing=c(T, F), clus.outcome.avg=F)

cl <- parallel::makeCluster(n.cores)
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(
  res <- 
    apply(cases, 1,
          function(case, data.validation, K, gamma, sigma, count, n.cores,
                   pkgs, id, y, vars, trt, prob,
                   clus.tree.growing, clus.outcome.avg, trts) {
            names.case <- names(case)
            for (j in 1:length(case)) assign(names.case[j], unname(case[j]))
            raw <- foreach::foreach(
              i=1:n.cores, .packages=pkgs, .combine=dplyr::bind_rows) %dopar% {
                source("/gpfs/home/ww857/Projects/Ladhania/DOF/code/library.R")
                Rcpp::sourceCpp("/gpfs/home/ww857/Projects/Ladhania/DOF/code/library.cpp")
                data.trainest <- sim.data(n.trainest, K, gamma, sigma, count)
                data.val <- data.validation %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                data.tre <- data.trainest %>%
                  mutate(!!(trt):=as.integer(factor(!!sym(trt), trts)))
                df.cf <- grf::multi_arm_causal_forest(
                  data.tre %>% select(all_of(vars)) %>% as.matrix,
                  data.tre %>% pull(y),
                  data.tre %>% pull(trt) %>% factor, num.trees=ntree,
                  mtry=nvar, min.node.size=nodesize) %>%
                  predict(data.val %>% select(all_of(vars)) %>% as.matrix) %>%
                  .$predictions %>% .[,,1] %>% cbind(0, .)
                colnames(df.cf) <- trts
                
                tmp.cf <-
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), names_to=trt, names_prefix=y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.cf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), names_to=trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  rename(!!paste0(trt, ".cf"):=!!sym(trt)) %>%
                  inner_join(oracle(data.validation, y, id, trt) %>%
                               select(!!sym(id), !!sym(paste0(trt, ".oracle"))), by=id) %>%
                  dplyr::summarise(true.cf=mean(!!sym(y)),
                                   rate.optassign.cf=mean(!!sym(paste0(trt,".oracle"))==
                                                            !!sym(paste0(trt,".cf"))))
                rate.optassign.cf <- tmp.cf %>% pull(rate.optassign.cf)
                true.cf <- tmp.cf %>% pull(true.cf)
                
                df.rf <- lapply(1:length(trts), function(i) {
                  data.tmp <- data.tre %>% filter(!!sym(trt)==i)
                  (ranger::ranger(as.formula(paste(y,"~",paste0(vars,collapse="+"))),
                                  data.tmp, num.trees=ntree,
                                  mtry=nvar, min.node.size=nodesize) %>%
                      predict(data.val))[["predictions"]]
                }) %>% do.call(cbind, .)
                colnames(df.rf) <- trts
                
                tmp.rf <-
                  dplyr::select(data.val, all_of(c(id, paste0(y, trts)))) %>%
                  pivot_longer(paste0(y, trts), names_to=trt, names_prefix=y, values_to=y) %>%
                  mutate(across(c(id, trt), as.character)) %>%
                  inner_join(
                    df.rf %>% as.data.frame %>%
                      mutate(!!(id):=data.val %>% pull(id)) %>%
                      relocate(id) %>%
                      pivot_longer(all_of(trts), names_to=trt, values_to=y) %>%
                      group_by(id) %>%
                      dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                                       .groups="drop"), by=c(id, trt)) %>%
                  rename(!!paste0(trt, ".rf"):=!!sym(trt)) %>%
                  inner_join(oracle(data.validation, y, id, trt) %>%
                               select(!!sym(id), !!sym(paste0(trt, ".oracle"))), by=id) %>%
                  dplyr::summarise(true.rf=mean(!!sym(y)),
                                   rate.optassign.rf=mean(!!sym(paste0(trt,".oracle"))==
                                                            !!sym(paste0(trt,".rf"))))
                rate.optassign.rf <- tmp.rf %>% pull(rate.optassign.rf)
                true.rf <- tmp.rf %>% pull(true.rf)
                
                growForest(data.trainest, data.validation, y=y, id=id, trt=trt, 
                           vars=vars, prob=prob, ntrt=ntrt, nvar=nvar,
                           lambda1=lambda1, lambda2=lambda2, nodesize=nodesize,
                           ntree=ntree, epi=epi, reg=reg, impute=impute,
                           clus.tree.growing=clus.tree.growing,
                           clus.outcome.avg=clus.outcome.avg, clus.max=5) %>%
                  inner_join(oracle(data.validation, y, id, trt), by=id) %>%
                  dplyr::summarise(
                    rate.optassign.dof=mean(!!sym(paste0(trt,".oracle"))==
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
                         rate.optassign.cf=rate.optassign.cf,
                         fitted.rf=mean(apply(df.rf, 1, max)),
                         true.rf=true.rf,
                         rate.optassign.rf=rate.optassign.rf)
              }
            bind_cols(as_tibble_row(case), raw)
          }, data.validation=data.validation, K=K, gamma=gamma, sigma=sigma,
          count=count, n.cores=n.cores, pkgs=pkgs, id=id, y=y, vars=vars,
          trt=trt, prob=prob, clus.tree.growing=clus.tree.growing,
          clus.outcome.avg=clus.outcome.avg, trts=trts) %>% bind_rows)
res <- res %>% mutate(reg=ifelse(reg==1, T, F),
                      impute=ifelse(impute==1, T, F),
                      clus.tree.growing=ifelse(clus.tree.growing==1, T, F),
                      clus.outcome.avg=ifelse(clus.outcome.avg==1, T, F))
save(res, cases, file=paste0("/gpfs/home/ww857/Projects/Ladhania/DOF/results/reg_multi_arm_cf_", K+1, "trt_ntrt", K+1,
                             "_ntrain5000_nodesize3_gamma", gamma, "_sigma", sigma, "_notrueclustering_clusvsnoclus_",
                             Sys.Date(), "_job_", Sys.getenv('SLURM_ARRAY_JOB_ID'), "_array_", Sys.getenv('SLURM_ARRAY_TASK_ID'),".RData"))
parallel::stopCluster(cl)