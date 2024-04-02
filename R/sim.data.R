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
    mutate(tmp1=10+20*(X1>0)-20*(X2>0)-40*(X1>0&X2>0),
           tmp2=gamma*(2*(X3>0)-1)/(K-1),
           tmp3=-10*X1^2,
           Y=tmp1+tmp2*(cl>0)*(2*cl-K-1)+tmp3*(cl==0)+rnorm(n,0,sigma),
           trt=str_c("c", cl, "t", cid))
  mapping <- data %>% distinct(trt, .keep_all=T) %>%
    dplyr::select(c(cl, cid, trt)) %>% arrange(trt)
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