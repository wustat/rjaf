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

sim.data <- function(n, K, gamma, sigma, count=rep(1,K+1)) {
  # K: number of clusters
  options(stringsAsFactors=F)
  data <- left_join(data.frame(id=1:n,
                               cl=sample(0:K, n, T, count),
                               cid=0,
                               MASS::mvrnorm(n, rep(0,3), diag(3))),
                    data.frame(cl=0:K, prob=1/sum(count)), by="cl")
  invisible(sapply(0:K, function(t) data[data$cl==t, "cid"] <<-
                     sample(count[t+1], sum(data$cl==t), T, rep(1, count[t+1]))))
  data <- data %>%
    mutate(tmp1=10+20*(X1>0)-20*(X2>0)-40*(X1>0&X2>0)+1,
           tmp2=gamma*(2*(X3>0)-1)/(K-1),
           tmp3=-X1^2,
           Y=tmp1+tmp2*(cl>0)*(2*cl-K-1)+tmp3*(cl==0)+rnorm(n,0,sigma),
           trt=str_c("c", cl, "t", cid))
  mapping <- data %>% distinct(trt, .keep_all=T) %>%
    select(c(cl, cid, trt)) %>% arrange(trt)
  # Y: observed outcomes
  Y.cf.trt <- data.frame(sapply(mapping %>% pull(trt), function(t) {
    # counterfactural outcomes
    clus <- mapping %>% filter(trt==t) %>% pull(cl)
    mutate(data, Y=tmp1+tmp2*(clus>0)*(2*clus-K-1)+tmp3*(clus==0))$Y
  }))
  names(Y.cf.trt) <- paste0("Y", mapping %>% pull(trt))
  Y.cf.cl <- data.frame(sapply(0:K, function(t) {
    # counterfactural outcomes
    mutate(data, Y=tmp1+tmp2*(t>0)*(2*t-K-1)+tmp3*(t==0))$Y
  }))
  names(Y.cf.cl) <- paste0("Y",0:K)
  return(mutate(bind_cols(dplyr::select(data, -c(tmp1,tmp2,tmp3)), Y.cf.trt, Y.cf.cl),
                across(c(id, cl), as.character)))
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
                       clus.max=10, reg=TRUE, impute=TRUE,
                       setseed=FALSE, seed=1, nfold=5) {
  trts <- unique(pull(data.trainest, trt))
  if (ntrt>length(trts)) stop("Invalid ntrt!")
  data.trainest <- mutate(data.trainest, across(c(id, trt), as.character))
  data.validation <- mutate(data.validation, across(c(id, trt), as.character))
  if (resid) data.trainest <- residualize(data.trainest, y, vars, nfold)
  if (clus.tree.growing) {
    if (clus.max>length(trts) | clus.max<2) stop("Invalid clus.max!")
    data.trainest$fold <- sample(1:nfold, NROW(data.trainest), T, rep(1, nfold))
    ls.kmeans <- lapply(2:clus.max, function(i)
      stats::kmeans(
        t(do.call(rbind, lapply(1:nfold, function(k) {
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
        }))), i, nstart=5))
    vec.prop <- sapply(ls.kmeans, function(list) list$betweenss/list$totss)
    cluster <- ls.kmeans[[which.max(diff(vec.prop))+1]]$cluster
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
    str.tree.growing <- as.integer(factor(pull(data.trainest, cluster),
                                          as.character(clus)))
    prob.tree.growing <- pull(data.trainest, prob_cluster)
    nstr <- length(unique(cluster))
    if (clus.outcome.avg) {
      str.outcome.avg <- as.integer(factor(pull(data.trainest, cluster),
                                           as.character(clus)))
    } else {
      str.outcome.avg <- as.integer(factor(pull(data.trainest, trt),
                                           as.character(trts)))
    }
  } else {
    str.tree.growing <- as.integer(factor(pull(data.trainest, trt),
                                          as.character(trts)))
    prob.tree.growing <- pull(data.trainest, prob)
    nstr <- ntrt
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
  if (clus.tree.growing & clus.outcome.avg) {
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
  trts <- data %>% pull(!!sym(trt)) %>% unique
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