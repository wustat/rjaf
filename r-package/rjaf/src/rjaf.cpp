#define ARMA_NO_DEBUG
// #define ARMA_DONT_USE_OPENMP
#ifndef STRICT_R_HEADERS // needed on Windows, not macOS
#define STRICT_R_HEADERS 1
#endif 
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h> // for Rcpp::RcppArmadillo::sample
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
// using namespace arma;
// using namespace std;

template <typename T>
arma::uvec index_subset(T v, T sub) {
  arma::uvec ids = find(v==sub(0));
  for (unsigned int i = 1; i < sub.n_elem; ++i) {
    arma::uvec tmp = find(v==sub(i));
    if (tmp.n_elem>0) ids.insert_rows(0, tmp);
  }
  return sort(ids);
}

bool newsplit(std::vector<unsigned int> &vars, std::vector<double> &cutoffs,
              unsigned int &var, double &cutoff) {
  bool news = true; // default to true
  for (unsigned int i = 0; i < vars.size(); ++i) {
    if (vars[i]==var && cutoffs[i]==cutoff) {
      news = false;
      break;
    }
  }
  return news;
}

arma::uvec setdiff(arma::uvec &a, arma::uvec &b) {
  a = sort(a), b = sort(b);
  std::vector<unsigned int> avec = arma::conv_to<std::vector<unsigned int>>::from(a),
    bvec = arma::conv_to< std::vector<unsigned int>>::from(b), cvec;
  
  set_difference(avec.begin(), avec.end(), bvec.begin(), bvec.end(), 
                 std::inserter(cvec, cvec.end()));
  arma::uvec c = arma::conv_to<arma::uvec>::from(cvec);
  return c;
}

List slice_sample(const arma::uvec &ids, const arma::uvec &trt, const double &prop=0.5) {
  arma::uvec trt_uniq = sort(unique(trt));
  unsigned int ntrt = trt_uniq.n_elem;
  arma::uvec ids_train, ids_est;
  for (unsigned int t = 0; t < ntrt; ++t) {
    arma::uvec ids_tmp = ids(find(trt==trt_uniq(t)));
    unsigned int n_tmp = floor(prop*ids_tmp.n_elem);
    arma::uvec ids_tmp_train = Rcpp::RcppArmadillo::sample(ids_tmp, n_tmp, false);
    ids_train.insert_rows(0, ids_tmp_train);
    ids_est.insert_rows(0, setdiff(ids_tmp, ids_tmp_train));
  }
  return List::create(_["ids_train"]=ids_train, _["ids_est"]=ids_est);
}

void set_seed(unsigned int seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);  
}

// identify the optimal split given a subsample of training data
List splitting(const arma::vec &y, const arma::mat &X, const arma::uvec &trt, const arma::vec &prob,
               const double &lambda=0.5, const bool &ipw=true,
               const unsigned int &nodesize=5) {
  Function Rquantile("quantile"); // call R's quantile within Rcpp
  arma::uvec trt_uniq = sort(unique(trt));
  unsigned int ntrt = trt_uniq.n_elem;
  unsigned int nvar = X.n_cols;
  // group ids, y, X, probs by trt
  std::vector<arma::uvec> idx_trt; std::vector<arma::vec> y_trt;
  std::vector<arma::mat> X_trt; arma::rowvec probs(ntrt);
  for (unsigned int t = 0; t < ntrt; ++t) {
    arma::uvec tmp = find(trt==trt_uniq(t));
    idx_trt.push_back(tmp);
    y_trt.push_back(y.elem(tmp));
    probs(t) = mean(prob.elem(tmp));
    X_trt.push_back(X.rows(tmp));
  }
  // for each covariate, record two values of optimal utility, an optimal cutoff,
  // qualification status, and two optimal treatments
  arma::mat utility(nvar, 2, arma::fill::zeros); arma::vec cutoffs_var(nvar, arma::fill::zeros);
  arma::uvec qualified(nvar); arma::umat treatments(nvar, 2);
  if (ipw) {
    for (unsigned int i = 0; i < nvar; ++i) { // go through each covariate
      arma::vec x_uniq = sort(unique(X.col(i))), cutoffs;
      if (x_uniq.n_elem < 10) { // fewer than 10 unique values 
        cutoffs = x_uniq;
        // adjust the least cutoff to guarantee non-empty nodes
        cutoffs(0) = mean(cutoffs.subvec(0,1));
      } else { // at least 10 unique values
        // cutoffs = quantile(x_uniq, regspace(0.1,0.1,0.9)); // Armadillo
        cutoffs = as<arma::vec>(Rquantile(x_uniq, arma::regspace(0.1,0.1,0.9),
                                          _["type"]=7)); // R
      }
      // for each cutoff, record two values of utility, two optimal treatments,
      // and qualification status
      arma::mat util(cutoffs.n_elem, 2); arma::umat trts(cutoffs.n_elem, 2);
      arma::uvec qual(cutoffs.n_elem);
      for (unsigned int j = 0; j < cutoffs.n_elem; ++j) { // go thru each cutoff
        // record numer, denom, and count for each treatment
        arma::mat numer(ntrt,2), denom(ntrt,2); arma::umat count(ntrt,2);
        for (unsigned int t = 0; t < ntrt; ++t) {
          arma::vec ya = y_trt[t](find(X_trt[t].col(i)<cutoffs(j))); 
          arma::vec yb = y_trt[t](find(X_trt[t].col(i)>=cutoffs(j)));
          count(t,0) = ya.n_elem; count(t,1) = yb.n_elem;
          numer(t,0) = arma::accu(ya)/(ya.n_elem+lambda);
          numer(t,1) = arma::accu(yb)/(yb.n_elem+lambda);
          denom(t,0) = ya.n_elem/(ya.n_elem+lambda);
          denom(t,1) = yb.n_elem/(yb.n_elem+lambda);
        }
        arma::mat regavg = 1 - denom;
        regavg.each_row() %= sum(numer)/sum(denom);
        regavg += numer; // regularized avg outcomes
        arma::rowvec regavg_opt = max(regavg); // max regavg for each column
        // record optimal treatments for both branches (may not be unique)
        arma::uvec id0 = find(regavg.col(0)==regavg_opt(0) && count.col(0)!=0);
        arma::uvec id1 = find(regavg.col(1)==regavg_opt(1) && count.col(1)!=0);
        arma::uvec ids(2);
        // pick a single optimal treatment for each branch
        for (unsigned int k = 0; k < id0.n_elem; ++k) {
          ids(0) = id0(k);
          for (unsigned int l = 0; l < id1.n_elem; ++l) {
            ids(1) = id1(l);
            if (ids(0)!=ids(1)) break;
          }
          if (ids(0)!=ids(1)) break;
        }
        trts(j,0) = trt_uniq(ids(0)); trts(j,1) = trt_uniq(ids(1));
        arma::urowvec count_opt = {count(ids(0),0), count(ids(1),1)};
        arma::rowvec prob_opt = {probs(ids(0)), probs(ids(1))};
        util.row(j) = regavg_opt % count_opt / prob_opt; // ipw adjusted
        arma::urowvec count_split = sum(count); // column sum
        // check conditions 1, 2, and 3 for recursive partitioning
        qual(j) = static_cast<unsigned int>(min(count_split)>=nodesize &&
          min(sum(count!=0))>=2 && ids(0)!=ids(1));
      }
      arma::uvec id_notqual = find(qual==0);
      // drop cutoffs with unqualified splits
      util.shed_rows(id_notqual); cutoffs.shed_rows(id_notqual);
      trts.shed_rows(id_notqual);
      if (util.n_rows==0) { // no qualified record for ith covariate
        qualified(i) = 0;
      } else {
        unsigned int id_max = index_max(sum(util, 1)); // find optimal cutoff
        utility.row(i) = util.row(id_max);
        treatments.row(i) = trts.row(id_max);
        cutoffs_var(i) = cutoffs(id_max);
        qualified(i) = 1;
      }
    }
  } else {
    for (unsigned int i = 0; i < nvar; ++i) { // go through each covariate
      arma::vec x_uniq = sort(unique(X.col(i))), cutoffs;
      if (x_uniq.n_elem < 10) {
        cutoffs = x_uniq;
        cutoffs(0) = mean(cutoffs.subvec(0,1));
      } else {
        // cutoffs = quantile(x_uniq, regspace(0.1,0.1,0.9));
        cutoffs = as<arma::vec>(Rquantile(x_uniq, arma::regspace(0.1,0.1,0.9),
                                          _["type"]=7));
      }
      arma::mat util(cutoffs.n_elem, 2);
      arma::umat trts(cutoffs.n_elem, 2);
      arma::uvec qual(cutoffs.n_elem);
      for (unsigned int j = 0; j < cutoffs.n_elem; ++j) {
        arma::mat numer(ntrt,2), denom(ntrt,2);
        arma::umat count(ntrt,2);
        for (unsigned int t = 0; t < ntrt; ++t) {
          arma::vec ya = y_trt[t](find(X_trt[t].col(i)<cutoffs(j))); 
          arma::vec yb = y_trt[t](find(X_trt[t].col(i)>=cutoffs(j)));
          count(t,0) = ya.n_elem; count(t,1) = yb.n_elem;
          numer(t,0) = accu(ya)/(ya.n_elem+lambda);
          numer(t,1) = accu(yb)/(yb.n_elem+lambda);
          denom(t,0) = ya.n_elem/(ya.n_elem+lambda);
          denom(t,1) = yb.n_elem/(yb.n_elem+lambda);
        }
        arma::mat regavg = 1 - denom; regavg.each_row() %= sum(numer)/sum(denom);
        regavg += numer;
        arma::rowvec regavg_opt = max(regavg);
        arma::uvec id0 = find(regavg.col(0)==regavg_opt(0) && count.col(0)!=0);
        arma::uvec id1 = find(regavg.col(1)==regavg_opt(1) && count.col(1)!=0);
        arma::uvec ids(2);
        for (unsigned int k = 0; k < id0.n_elem; ++k) {
          ids(0) = id0(k);
          for (unsigned int l = 0; l < id1.n_elem; ++l) {
            ids(1) = id1(l);
            if (ids(0)!=id0(1)) break;
          }
          if (ids(0)!=id0(1)) break;
        }
        trts(j,0) = trt_uniq(ids(0));
        trts(j,1) = trt_uniq(ids(1));
        arma::urowvec count_opt = {count(ids(0),0), count(ids(1),1)};
        arma::urowvec count_split = sum(count);
        util.row(j) = regavg_opt % count_split;
        qual(j) = static_cast<unsigned int>(min(count_split)>=nodesize &&
          min(sum(count!=0))>=2 && ids(0)!=ids(1));
      }
      arma::uvec id_notqual = find(qual==0);
      util.shed_rows(id_notqual); cutoffs.shed_rows(id_notqual);
      trts.shed_rows(id_notqual);
      if (util.n_rows==0) { // no qualified record for ith var
        qualified(i) = 0;
      } else {
        unsigned int id_max = index_max(sum(util, 1));
        utility.row(i) = util.row(id_max);
        treatments.row(i) = trts.row(id_max);
        cutoffs_var(i) = cutoffs(id_max);
        qualified(i) = 1;
      }
    }
  }
  if (any(qualified)) { // at least one covariate with qualified optimal split
    arma::uvec ids = find(qualified==1);
    arma::mat util_qualified = utility.rows(ids);
    arma::umat trt_qualified = treatments.rows(ids);
    arma::vec cutoffs_qualified = cutoffs_var.rows(ids);
    unsigned int idx_var = index_max(sum(util_qualified, 1));
    return List::create(_["var"]=ids(idx_var),
                        _["cutoff"]=cutoffs_qualified(idx_var),
                        _["trt"]=trt_qualified.row(idx_var),
                        _["util"]=util_qualified.row(idx_var));
  } else {
    return List::create(_["var"]=-1);
  }
}

List growTree(const arma::vec &y_trainest, const arma::vec &y_trainest_resid, 
              const arma::mat &X_trainest,
              const arma::uvec &trt_trainest, const arma::vec &prob_trainest,
              const arma::uvec &cluster_trainest,
              const arma::mat &X_val, const unsigned int &ntrts=5,
              const unsigned int &nvars=3, const double &lambda1=0.5,
              const double &lambda2=0.5, const bool &ipw=true,
              const unsigned int &nodesize=5,
              const double &prop_train=0.5,
              const double &eps=0.1, const bool &reg=true,
              const bool &impute=true,
              const bool &setseed=false, const unsigned int &seed=1) {
  // lambda1 for tree growing on training set; lambda2 for tree growing on estimation and validation sets
  if (setseed) set_seed(seed);
  // analogous to slice_sample in dplyr: sampling within each treatment slice
  List list_ids = slice_sample(arma::regspace<arma::uvec>(0, X_trainest.n_rows-1),
                               trt_trainest, prop_train);
  const arma::uvec ids_train = list_ids["ids_train"], ids_est = list_ids["ids_est"]; // both with respect to X_trainest
  const arma::uvec trt = trt_trainest(ids_train);
  const arma::vec y = y_trainest_resid(ids_train), prob = prob_trainest(ids_train);
  const arma::mat X = X_trainest.rows(ids_train), X_est = X_trainest.rows(ids_est);
  // trt, y, prob, X all for training set
  // utility at root: begin
  arma::uvec trt_uniq = sort(unique(trt));
  unsigned int ntrt = trt_uniq.n_elem;
  if (trt_uniq.n_elem < ntrts) {
    stop("The number of unique treatment arms or clusters in the training set is less than the number of treatment arms or clusters specified!");
  } 
  arma::vec numer(ntrt), denom(ntrt), probs(ntrt);
  arma::uvec count(ntrt);
  for (unsigned int t = 0; t < ntrt; ++t) {
    arma::uvec tmp = find(trt==trt_uniq(t));
    arma::vec y_trt = y(tmp);
    count(t) = y_trt.n_elem;
    probs(t) = mean(prob(tmp));
    numer(t) = accu(y_trt)/(y_trt.n_elem+lambda1);
    denom(t) = y_trt.n_elem/(y_trt.n_elem+lambda1);
  }
  arma::vec regavg = 1 - denom;
  regavg *= accu(numer)/accu(denom);
  regavg += numer;
  unsigned int id = index_max(regavg);
  double util_root;
  if (ipw) {
    util_root = max(regavg)*count(id)/probs(id);
  } else {
    util_root = max(regavg)*accu(count);
  }
  // utility at root: end
  // tree growing begins
  IntegerVector parentnode = {0}, node = {1}; // parentnode id and node id
  std::set<unsigned int> nodes2split; nodes2split.insert(1); // start from root node
  std::vector<arma::uvec> filter; // filter records row indices of X
  filter.push_back(arma::regspace<arma::uvec>(0, X.n_rows-1)); // all rows of training data
  std::vector<arma::uvec> filter_est;
  filter_est.push_back(arma::regspace<arma::uvec>(0, X_est.n_rows-1)); // all rows of est data
  std::vector<arma::uvec> filter_val;
  filter_val.push_back(arma::regspace<arma::uvec>(0, X_val.n_rows-1)); // all rows of val data
  CharacterVector type = {"split"}; // start from root node as split
  // "leaf" indicates terminal node with no further splitting
  // "parent" indicates splits created already 
  NumericVector util = {util_root}; // util at root node
  std::vector<unsigned int> vars;  std::vector<double> cutoffs;
  // push-backs below are for root node only
  vars.push_back(X.n_cols); cutoffs.push_back(0.0);
  // X.n_cols larger than any index; no var to split on
  double mingain = eps*stddev(y);
  do {
    std::set<unsigned int> nodes2split_tmp = nodes2split;
    for (std::set<unsigned int>::iterator i=nodes2split_tmp.begin();
          i!=nodes2split_tmp.end(); ++i) {
      unsigned int node2split = *i; // node to split on
      arma::uvec ids = filter[node2split-1]; // zero-index in C++
      arma::uvec ids_est = filter_est[node2split-1];
      arma::uvec ids_val = filter_val[node2split-1];
      arma::uvec trt_tmp = trt(ids); // trt column of the subsample
      arma::uvec trt_sub;
      trt_sub = Rcpp::RcppArmadillo::sample(trt_uniq, ntrts, false);
      arma::uvec var_sub = Rcpp::RcppArmadillo::sample(
        arma::regspace<arma::uvec>(0, X.n_cols-1), nvars, false);
      // ids only with selected treatment levels
      arma::uvec ids4split = ids(index_subset(trt_tmp, trt_sub));
      List split; int var_id;
      if (ids4split.n_elem==0) { // trt_tmp and trt_sub are disjoint--nothing to split
        var_id = -1; 
      } else { // ids4split is nonempty
        split = splitting(y(ids4split), X.submat(ids4split, var_sub),
                          trt(ids4split), prob(ids4split),
                          lambda1, ipw, nodesize);
        var_id = split["var"];
      }
      if (var_id==-1) { // split above is null
        type[node2split-1] = "leaf";
        nodes2split.erase(node2split); // split unavailable; drop the node
      } else { // split is nonnull
        arma::rowvec utils_split = split["util"];
        if (accu(utils_split) - util[node2split-1] < mingain) { // no welfare gain
          type[node2split-1] = "leaf";
          nodes2split.erase(node2split); // split is not beneficial; drop the node
        } else { // with welfare gain
          unsigned int var = var_sub(static_cast<unsigned int>(var_id)); // var for splitting
          double cutoff = split["cutoff"];
          if (newsplit(vars, cutoffs, var, cutoff)) { // new split
            arma::uvec var_tmp(1); var_tmp.fill(var);
            arma::uvec ids_est_left = ids_est(find(X_est(ids_est,var_tmp).as_col()<cutoff));
            arma::uvec ids_est_right = ids_est(find(X_est(ids_est,var_tmp).as_col()>=cutoff));
            arma::uvec ids_val_left = ids_val(find(X_val(ids_val,var_tmp).as_col()<cutoff));
            arma::uvec ids_val_right = ids_val(find(X_val(ids_val,var_tmp).as_col()>=cutoff));
            if (ids_est_left.n_elem==0 || ids_est_right.n_elem==0 ||
                ids_val_left.n_elem==0 || ids_val_right.n_elem==0) {
              // split leads to empty branch for est or val data
              type[node2split-1] = "leaf";
              nodes2split.erase(node2split); // drop split leading to empty branch
            } else {
              filter_est.push_back(ids_est_left); filter_est.push_back(ids_est_right);
              filter_val.push_back(ids_val_left); filter_val.push_back(ids_val_right);
              filter.push_back(ids(find(X(ids,var_tmp).as_col()<cutoff)));
              filter.push_back(ids(find(X(ids,var_tmp).as_col()>=cutoff)));
              parentnode.push_back(node2split); parentnode.push_back(node2split);
              nodes2split.insert(max(node)+1); node.push_back(max(node)+1);
              nodes2split.insert(max(node)+1); node.push_back(max(node)+1);
              type[node2split-1] = "parent";
              nodes2split.erase(node2split); // split node off the list
              type.push_back("split"); type.push_back("split");
              util.push_back(utils_split(0)); util.push_back(utils_split(1));
              vars.push_back(var);
              cutoffs.push_back(cutoff);
            }
          } else {
            type[node2split-1] = "leaf";
            nodes2split.erase(node2split); // drop old split
          }
        }
      }
    }
  } while (nodes2split.size() > 0); // at least 1 node to split; if not, stop growing
  // tree growing ends
  for (unsigned int i = 0; i < type.size(); ++i) { // keep terminal nodes only
    if (type[i]=="leaf") { // terminal nodes
      filter_est.insert(filter_est.begin()+i, ids_est(filter_est[i]));
      filter_est.erase(filter_est.begin()+i+1);
    } else { // nonterminal nodes
      type.erase(i);
      filter_est.erase(filter_est.begin()+i);
      filter_val.erase(filter_val.begin()+i);
      --i;
    }
  }
  arma::mat mat_res, mat_ct;
  if (reg) {
    arma::uvec clus_uniq = sort(unique(cluster_trainest));
    mat_res.zeros(X_val.n_rows, clus_uniq.n_elem);
    mat_ct.zeros(X_val.n_rows, clus_uniq.n_elem);
    for (unsigned int i = 0; i < type.size(); ++i) { // go thru each terminal node
      arma::uvec clus_tmp = cluster_trainest(filter_est[i]); // subvector of trt column
      arma::vec y_tmp = y_trainest(filter_est[i]); // subvector of outcome column
      arma::urowvec count(clus_uniq.n_elem);
      arma::rowvec numer(clus_uniq.n_elem), denom(clus_uniq.n_elem);
      for (unsigned int t = 0; t < clus_uniq.n_elem; ++t) {
        arma::uvec id_tmp = find(clus_tmp==clus_uniq(t));
        if (id_tmp.n_elem==0) { // one trt level missing
          count(t) = 0; numer(t) = 0; denom(t) = 0;
        } else {
          count(t) = id_tmp.n_elem;
          numer(t) = accu(y_tmp(id_tmp))/(id_tmp.n_elem+lambda2);
          denom(t) = id_tmp.n_elem/(id_tmp.n_elem+lambda2);
        }
      }
      arma::rowvec regavg = 1 - denom; regavg *= accu(numer)/accu(denom);
      regavg += numer;
      if (!impute) {
        for (unsigned int t = 0; t < clus_uniq.n_elem; ++t) {
          if (count(t)==0) regavg(t) = 0;
        }
      }
      arma::rowvec regwt(regavg.n_elem, arma::fill::zeros);
      regwt(find(count)) += 1.0;
      // assign regavg to units of val data belonging to the terminal node
      mat_res.rows(filter_val[i]) += repmat(regavg, filter_val[i].n_elem, 1);
      mat_ct.rows(filter_val[i]) += repmat(regwt, filter_val[i].n_elem, 1);
    }
  } else {
    arma::uvec clus_uniq = sort(unique(cluster_trainest));
    mat_res.zeros(X_val.n_rows, clus_uniq.n_elem);
    mat_ct.zeros(X_val.n_rows, clus_uniq.n_elem);
    for (unsigned int i = 0; i < type.size(); ++i) { // go thru each terminal node
      arma::uvec clus_tmp = cluster_trainest(filter_est[i]); // subvector of trt column
      arma::vec y_tmp = y_trainest(filter_est[i]); // subvector of outcome column
      arma::urowvec count(clus_uniq.n_elem);
      arma::rowvec avg(clus_uniq.n_elem);
      for (unsigned int t = 0; t < clus_uniq.n_elem; ++t) {
        arma::uvec id_tmp = find(clus_tmp==clus_uniq(t));
        if (id_tmp.n_elem==0) { // one trt level missing
          count(t) = 0; avg(t) = 0;
        } else {
          count(t) = id_tmp.n_elem;
          avg(t) = mean(y_tmp(id_tmp));
        }
      }
      arma::rowvec wt(clus_uniq.n_elem, arma::fill::zeros);
      wt(find(count)) += 1.0;
      // assign avg to units of val data belonging to the terminal node
      mat_res.rows(filter_val[i]) += repmat(avg, filter_val[i].n_elem, 1);
      mat_ct.rows(filter_val[i]) += repmat(wt, filter_val[i].n_elem, 1);
    }
  }
  return List::create(_["res"]=mat_res, _["ct"]=mat_ct);
}

// [[Rcpp::export]]
List rjaf_cpp(const arma::vec &y_trainest, const arma::vec &y_trainest_resid,
              const arma::mat &X_trainest,
              const arma::uvec &trt_trainest, const arma::vec &prob_trainest,
              const arma::uvec &cluster_trainest, const arma::mat &X_val,
              const unsigned int &ntrts=5, const unsigned int &nvars=3,
              const double &lambda1=0.5, const double &lambda2=0.5,
              const bool &ipw=true, const unsigned int &nodesize=5,
              const unsigned int &ntree=1000,
              const double &prop_train=0.5, const double &eps=0.1,
              const bool &reg=true, const bool &impute=true,
              const bool &setseed=false, const unsigned int &seed=1) {
  if (setseed) set_seed(seed);
  arma::uvec clus_uniq = sort(unique(cluster_trainest));
  unsigned int nclus = clus_uniq.n_elem;
  arma::mat outcome(X_val.n_rows, nclus, arma::fill::zeros), ct(X_val.n_rows, nclus, arma::fill::zeros);
  for (unsigned int i = 0; i < ntree; ++i) {
    List tree_tmp = growTree(y_trainest, y_trainest_resid, X_trainest, trt_trainest,
                             prob_trainest, cluster_trainest,
                             X_val, ntrts, nvars, lambda1,
                             lambda2, ipw, nodesize, prop_train, eps,
                             reg, impute);
    arma::mat outcome_tmp = tree_tmp["res"], ct_tmp = tree_tmp["ct"];
    outcome += outcome_tmp;
    ct += ct_tmp;
  }
  ct.replace(0.0, 1.0);
  outcome /= ct;
  // identify optimal clus idx for each unit in val data
  arma::uvec idx_clus = index_max(outcome, 1);
  arma::uvec clus_pred(idx_clus.n_elem, arma::fill::zeros);
  for (unsigned int t = 0; t < nclus; ++t) {
    clus_pred(find(idx_clus==t)) += clus_uniq(t);
  }
  return List::create(_["Y.cf"]=outcome, _["Y.pred"]=max(outcome, 1),
                      _["trt.rjaf"]=clus_pred);
}
