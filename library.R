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

residualize <- function(data, y, vars, nfolds=5, fun.rf="ranger") {
  data$fold <- sample(1:nfolds, NROW(data), T, rep(1, nfolds))
  if (fun.rf=="randomForest") {
    dplyr::select(bind_rows(lapply(1:nfolds, function(i) {
      data.i <- filter(data, fold==i)
      data.i[,y] <- data.i[,y] -
        predict(randomForest::randomForest(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          filter(data, fold!=i)), data.i)
      data.i})), -fold)
  } else if (fun.rf=="ranger") {
    dplyr::select(bind_rows(lapply(1:nfolds, function(i) {
      data.i <- filter(data, fold==i)
      data.i[,y] <- data.i[,y] -
        predict(ranger::ranger(
          as.formula(paste(y, "~", paste0(vars, collapse="+"))),
          filter(data, fold!=i)), data.i)[["predictions"]]
      data.i})), -fold)
  }
}

util.root <- function(data, y, trt, prob, lambda=0.5, ipw=TRUE) { # Section 4.1
  # y, trt, prob are strings; lambda is positive; ipw is logical
  df <- mutate(dplyr::summarise(
    group_by(data, !!sym(trt)), N=n(), avg.arm=mean(!!sym(y)),
    prob=mean(!!sym(prob)), .groups="drop"), sum=sum(N))
  avg.root <-pull(
    dplyr::summarise(mutate(df, numer=avg.arm*N/(N+lambda), denom=N/(N+lambda)),
                     avg.root=sum(numer)/sum(denom)), avg.root)
  return(pull(mutate(
    filter(mutate(df, reg.avg.arm=(avg.arm*N+lambda*avg.root)/(N+lambda)),
           reg.avg.arm==max(reg.avg.arm)),
    util=ifelse(ipw, reg.avg.arm*N/prob, reg.avg.arm*sum)), util))
}

splitting.var.cutoff <- function(data, y, trt, var, prob, cutoff,
                                 lambda=0.5, ipw=TRUE, nodesize=5) {
  df <- mutate(dplyr::summarise(
    group_by(mutate(data, rel=ifelse(!!sym(var)<cutoff, "<", ">=")),
             !!sym(trt), rel),
    N=n(), avg.leaf=mean(!!sym(y)),
    prob=mean(!!sym(prob)), .groups="drop"),
    numer=avg.leaf*N/(N+lambda), denom=N/(N+lambda))
  df <- mutate(right_join(
    dplyr::summarise(
      group_by(df, rel), avg.split=sum(numer)/sum(denom), N.split=sum(N),
      .groups="drop"), df, by="rel"),
    reg.avg.leaf=(avg.leaf*N+lambda*avg.split)/(N+lambda))
  splits.R.2 <- (NROW(filter(dplyr::summarise(
    group_by(df, rel), N.trt=n(), N.split=mean(N.split), .groups="drop"),
    N.trt>=2 & N.split >= nodesize))==2)
  if (splits.R.2) {
    if (ipw) {
      res <- dplyr::select(mutate(
        ungroup(filter(group_by(df, rel), reg.avg.leaf==max(reg.avg.leaf))),
        util=reg.avg.leaf*N/prob,
        var=var, cutoff=cutoff), 
        all_of(c("var", "rel", "cutoff", trt, "util")))
    } else {
      res <- dplyr::select(mutate(
      ungroup(filter(group_by(df, rel), reg.avg.leaf==max(reg.avg.leaf))),
      util=reg.avg.leaf*N.split,
      var=var, cutoff=cutoff), all_of(c("var", "rel", "cutoff", trt, "util")))
    }
    trts.R.different <- length(unique(pull(res, trt)))==2
    if (trts.R.different) return(res) else return(NULL)
  } else return(NULL)
}

splitting <- function(data, y, trt, vars, prob, lambda=0.5, ipw=TRUE,
                      nodesize=5) {
  res <- do.call(c,lapply(vars, function(var) {
    cutoffs <- sort(unique(pull(data, var)))
    cutoffs <- if (length(cutoffs)>=10L)
      quantile(cutoffs, seq(0.1,0.9,0.1), type=5) else cutoffs
    res.by.cutoff <- 
      lapply(cutoffs, splitting.var.cutoff, data=data, y=y, trt=trt, var=var, 
             prob=prob, lambda=lambda, ipw=ipw, nodesize=nodesize)
    res.by.cutoff}))
  res <- res[!sapply(res, is.null)]
  if (length(res)) 
    res[[which.max(sapply(res, function(df) sum(df$util)))]] else NULL
}

growTree <- function(data.trainest, data.validation, y, id, trt, vars, prob,
                     ntrt=5, nvar=3, lambda=0.5, ipw=TRUE, nodesize=5,
                     prop.train=0.5, epi=0.1) {
  data.train <- ungroup(slice_sample(
    group_by(data.trainest, !!sym(trt)), prop=prop.train))
  data.est <- filter(data.trainest, !(!!sym(id)) %in% pull(data.train,id))
  mingain <- epi*sd(pull(data.train, y))  # minimum welfare gain
  tree <- tibble(parent.node=0, node=1, filter="TRUE", type="split",
                 util=util.root(data.train, y, trt, prob, lambda, ipw))
  trts <- unique(pull(data.train, trt))
  repeat {
    nodes <- filter(tree, type=="split")$node # nodes to split
    invisible(sapply(nodes, function(i) {
      filter.node <- filter(tree, node==i)$filter
      split <-
        splitting(filter(filter(data.train, eval(parse(text=filter.node))),
                         !!sym(trt)%in%sample(trts, ntrt)),
                  y, trt, sample(vars, nvar), prob, lambda, ipw, nodesize)
      nogain <- try(sum(split$util) < # check whether welfare gain
                      filter(tree, node==i)$util + mingain, T)
      # check whether the node should be a leaf/terminal node
      is.terminal <- is.null(split) | ifelse(class(nogain)!="logical", T, nogain)
      tree <<-
        if (is.terminal) {
          mutate(tree, type=ifelse(node==i, "leaf", type))
        } else {
          # expressions of split
          exp.split <- mutate(split, split=paste(var,rel,cutoff))$split
          # new.split: check whether splitting rules have been invoked
          new.split <- sapply(exp.split, function(exp)
            !any(stringr::str_detect(tree$filter, exp)))
          # create splitting rules for child nodes
          filter.child <- if (filter.node=="TRUE") exp.split else 
            paste(filter.node, exp.split, sep=" & ")
          split.child <- tibble(parent.node=i, 
                                node=max(tree$node) + 1:2,
                                filter=filter.child, type="split", 
                                util=split$util)
          bind_rows(mutate(tree,type=ifelse(
            node==i, ifelse(any(new.split),"parent","leaf"), type)),
            split.child[new.split,])
        }}))
    stop.split <- all(tree$type!="split")
    if (stop.split) break
  }
  # create weight matrix
  ls.mat <- lapply(filter(tree, type=="leaf")$filter, function(cond) {
    id.est <- pull(filter(data.est, eval(parse(text=cond))), id)
    id.validation <- pull(filter(data.validation, eval(parse(text=cond))), id)
    matrix(1/length(id.est), length(id.validation), length(id.est),
           dimnames=list(id.validation, id.est))})
  mat <- cbind(do.call(Matrix::bdiag, ls.mat),
               Matrix::Matrix(0, nrow(data.validation), nrow(data.train)))
  dimnames(mat) <- list(unlist(sapply(ls.mat, rownames)),
                        c(unlist(sapply(ls.mat, colnames)),
                          pull(data.train, id)))
  return(list(mat=mat[order(rownames(mat)), order(colnames(mat))],
              tree=tree))
}

growForest <- function(data.trainest, data.validation, y, id, trt, vars, prob,
                       ntrt=5, nvar=3, lambda1=0.5, lambda2=0.5, ipw=TRUE,
                       nodesize=5, ntree=1000, prop.train=0.5, epi=0.1,
                       resid=TRUE, parallel=FALSE,
                       threads=parallel::detectCores()/2, Rcpp=TRUE, reg=TRUE,
                       impute=TRUE, setseed=FALSE, seed=1) {
  trts <- unique(pull(data.trainest, trt))
  data.trainest <- mutate(data.trainest, across(c(id, trt), as.character))
  data.validation <- mutate(data.validation, across(c(id, trt), as.character))
  if (resid) data.trainest <- residualize(data.trainest, y, vars)
  if (Rcpp) {
    ls.forest <-
      growForest_cpp(pull(data.trainest, y),
                     as.matrix(select(data.trainest, all_of(vars))),
                     as.integer(factor(pull(data.trainest, trt),
                                       as.character(trts))),
                     pull(data.trainest, prob),
                     as.matrix(select(data.validation, all_of(vars))),
                     ntrt, nvar, lambda1, lambda2, ipw, nodesize, ntree,
                     prop.train, epi, reg, impute, setseed, seed)
    res <- tibble(!!(id):=as.character(pull(data.validation, id)),
                  !!(trt):=as.character(trts[ls.forest$trt.dof]),
                  !!(paste0(y, ".pred")):=as.numeric(ls.forest$Y.pred))
  } else {
    Rownames <- sort(as.character(pull(data.validation, id)))
    Colnames <- sort(as.character(pull(data.trainest, id)))
    if (parallel) { # deprecated
      threads <- numbers::GCD(ntree, threads)
      cl <- parallel::makeCluster(threads)
      doParallel::registerDoParallel(cl)
      `%dopar%` <- foreach::`%dopar%`
      mat.weight <- Reduce("+", lapply(1:(ntree/threads), function(i) {
        ls.mat <- foreach::foreach(
          i=1:threads,
          .export=c("growTree", "splitting", "splitting.var.cutoff", "util.root",
                    formalArgs(growTree)),
          .packages=c("tibble","dplyr","stringr","tidyr", "readr")) %dopar% {
            growTree(data.trainest, data.validation, y, id, trt, vars, prob, ntrt,
                     nvar, lambda1, ipw, nodesize, prop.train, epi)$mat
          }
        Reduce("+",ls.mat)}))
      parallel::stopCluster(cl)
      dimnames(mat.weight) <- list(Rownames, Colnames)
    } else {
      mat.weight <- Matrix::Matrix(0, nrow(data.validation), nrow(data.trainest),
                                   dimnames=list(Rownames, Colnames))
      invisible(sapply(1:ntree, function(i) {
        mat.weight <<-
          growTree(data.trainest, data.validation, y, id, trt, vars, prob, ntrt,
                   nvar, lambda1, ipw, nodesize, prop.train, epi)$mat + mat.weight
      }))
    }
    mat.weight <- mat.weight / ntree
    outcome.p <- sapply(trts, function(i) {
      weight.tmp <- mat.weight[,pull(filter(data.trainest, trt==i), id)]
      as.vector((weight.tmp / Matrix::rowSums(weight.tmp)) %*% 
                  pull(filter(data.trainest, trt==i), y))
    })
    dimnames(outcome.p) <- list(rownames(mat.weight), trts)
    res <- dplyr::summarise(
      group_by(pivot_longer(
        mutate(as_tibble(outcome.p),!!id:=rownames(mat.weight)),
        cols=-all_of(id), names_to=trt, values_to=y),!!sym(id)),
      !!trt:=(!!sym(trt))[!!sym(y)==max(!!sym(y))],
      !!(paste0(y, ".pred")):=max(!!sym(y)), .groups="drop")
  }
  if (all(paste0(y, trts) %in% names(data.validation))) {
    res <- rename_with(inner_join(res, mutate(pivot_longer(
      dplyr::select(data.validation, all_of(c(id, paste0(y, trts)))),
      cols=paste0(y, trts), names_to=trt, names_prefix=y, values_to=y),
      across(c(id, trt), as.character)),
      by=c(id, trt)), ~str_c(.,".dof"), all_of(c(y, trt)))
  }
  return(res)
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