sim.data <- function(n, K, gamma, sigma, prob=rep(1,K+1)/(K+1)) {
  # K: number of treatment arms
  options(stringsAsFactors=F)
  data <- left_join(data.frame(id=1:n,
                               trt=sample(0:K, n, replace=T, prob),
                               MASS::mvrnorm(n, rep(0,3), diag(3))),
                    data.frame(trt=0:K, prob), by="trt")
  data <- mutate(data, tmp1=10+20*(X1>0)-20*(X2>0)-40*(X1>0&X2>0)+1,
                 tmp2=gamma*(2*(X3>0)-1)/(K-1),
                 tmp3=-X1^2,
                 Y=tmp1+tmp2*(trt>0)*(2*trt-K-1)+tmp3*(trt==0)+rnorm(n,0,sigma))
  # Y: observed outcomes
  Y.cf <- data.frame(sapply(0:K, function(t) # counterfactural outcomes
    mutate(data, Y=tmp1+tmp2*(t>0)*(2*t-K-1)+tmp3*(t==0))$Y))
  names(Y.cf) <- paste0("Y",0:K)
  return(mutate(bind_cols(dplyr::select(data, -c(tmp1,tmp2,tmp3)), Y.cf),
                across(c(id, trt), as.character)))
}

residualize <- function(data, y, vars, nfold=5, fun.rf="ranger") {
  data$fold <- sample(1:nfold, NROW(data), T, rep(1, nfold))
  if (fun.rf=="randomForest") {
    dplyr::select(bind_rows(lapply(1:nfold, function(i) {
      data.i <- filter(data, fold==i)
      data.i[,y] <- data.i[,y] -
        predict(randomForest::randomForest(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          filter(data, fold!=i)), data.i)
      data.i})), -fold)
  } else if (fun.rf=="ranger") {
    dplyr::select(bind_rows(lapply(1:nfold, function(i) {
      data.i <- filter(data, fold==i)
      data.i[,y] <- data.i[,y] -
        predict(ranger::ranger(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          filter(data, fold!=i)), data.i)[["predictions"]]
      data.i})), -fold)
  }
}

growForest <- function(data.trainest, data.validation, y, id, trt, vars, prob,
                       ntrt=5, nvar=3, lambda1=0.5, lambda2=0.5, ipw=TRUE,
                       nodesize=5, ntree=1000, prop.train=0.5, epi=0.1,
                       resid=TRUE, clus.tree.growing=FALSE, clus.outcome.avg=FALSE,
                       ncluster=3, reg=TRUE, impute=TRUE,
                       setseed=FALSE, seed=1, nfold=5) {
  trts <- unique(pull(data.trainest, trt))
  if (ntrt>length(trts)) stop("Invalid ntrt!")
  data.trainest <- mutate(data.trainest, across(c(id, trt), as.character))
  data.validation <- mutate(data.validation, across(c(id, trt), as.character))
  if (resid) data.trainest <- residualize(data.trainest, y, vars, nfold)
  clustering <- any(clus.tree.growing, clus.outcome.avg)
  if (clustering) {
    if (ncluster>length(trts)) stop("Invalid ncluster!")
    data.trainest$fold <- sample(1:nfold, NROW(data.trainest), T, rep(1, nfold))
    cluster <-
      stats::kmeans(t(do.call(rbind, lapply(1:nfold, function(k) {
        data.onefold <- filter(data.trainest, fold==k)
        data.rest <- filter(data.trainest, fold!=k)
        growForest_cpp(pull(data.rest, y),
                       as.matrix(select(data.rest, all_of(vars))),
                       as.integer(factor(pull(data.rest, trt),
                                         as.character(trts))),
                       pull(data.rest, prob),
                       as.integer(factor(pull(data.rest, trt),
                                         as.character(trts))),
                       as.matrix(select(data.onefold, all_of(vars))),
                       ntrt, nvar, lambda1, lambda2, ipw, nodesize, ntree,
                       prop.train, epi, reg, impute, setseed, seed)$Y.cf
      }))), ncluster, nstart=25)$cluster
    df <- data.frame(cluster)
    df[trt] <- as.character(trts)
    df <- summarise(group_by(data.trainest, trt),
                    !!(prob):=mean(!!sym(prob)), .groups="drop") %>%
      inner_join(df, by=trt) %>% group_by(cluster) %>%
      summarise(prob_cluster=sum(!!sym(prob)), .groups="drop") %>%
      inner_join(df, by="cluster") %>% as.data.frame
    data.trainest <- inner_join(data.trainest, df, by=trt) %>%
      mutate(cluster=as.character(cluster))
    clus <- unique(pull(data.trainest, cluster))
  }
  if (clus.tree.growing) {
    str.tree.growing <- as.integer(factor(pull(data.trainest, cluster),
                                          as.character(clus)))
    prob.tree.growing <- pull(data.trainest, prob_cluster)
    nstr <- ncluster
  } else {
    str.tree.growing <- as.integer(factor(pull(data.trainest, trt),
                                          as.character(trts)))
    prob.tree.growing <- pull(data.trainest, prob)
    nstr <- ntrt
  }
  if (clus.outcome.avg) {
    str.outcome.avg <- as.integer(factor(pull(data.trainest, cluster),
                                         as.character(clus)))
  } else {
    str.outcome.avg <- as.integer(factor(pull(data.trainest, trt),
                                         as.character(trts)))
  }
  ls.forest <-
    growForest_cpp(pull(data.trainest, y),
                   as.matrix(select(data.trainest, all_of(vars))),
                   str.tree.growing, prob.tree.growing, str.outcome.avg,
                   as.matrix(select(data.validation, all_of(vars))),
                   nstr, nvar, lambda1, lambda2, ipw, nodesize, ntree,
                   prop.train, epi, reg, impute, setseed, seed)
  if (clus.outcome.avg) {
    res <- tibble(!!(id):=as.character(pull(data.validation, id)),
                  cluster=as.character(clus[ls.forest$trt.dof]),
                  !!(paste0(y, ".pred")):=as.numeric(ls.forest$Y.pred))
    return(list(res=res, clustering=df))
  } else {
    res <- tibble(!!(id):=as.character(pull(data.validation, id)),
                  !!(trt):=as.character(trts[ls.forest$trt.dof]),
                  !!(paste0(y, ".pred")):=as.numeric(ls.forest$Y.pred))
    if (all(paste0(y, trts) %in% names(data.validation))) {
      res <- rename_with(inner_join(res, mutate(pivot_longer(
        dplyr::select(data.validation, all_of(c(id, paste0(y, trts)))),
        cols=paste0(y, trts), names_to=trt, names_prefix=y, values_to=y),
        across(c(id, trt), as.character)),
        by=c(id, trt)), ~str_c(.,".dof"), all_of(c(y, trt)))
    }
    return(res)
  }
}

oracle <- function(data, y, id, trt) {
  trts <- data %>% pull(trt) %>% unique
  data %>%
    dplyr::select(all_of(c(id, paste0(y, trts)))) %>%
    pivot_longer(cols=paste0(y, trts), names_to=trt, names_prefix=y,
                 values_to=y) %>% group_by(id) %>%
    dplyr::summarise(!!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
                     !!y:=max(!!sym(y)),
                     .groups="drop") %>%
    rename_with(~str_c(.,".oracle"), all_of(c(y, trt))) %>%
    inner_join(data %>% select(all_of(c(id, y))), by=id)
}