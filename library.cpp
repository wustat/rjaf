#define ARMA_NO_DEBUG
// #define ARMA_DONT_USE_OPENMP
#define STRICT_R_HEADERS // needed on Windows, not macOS
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h> // for Rcpp::RcppArmadillo::sample
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
List splitting_cpp(const vec &y, const mat &X, const uvec &trt, const vec &prob,
                   const double &lambda=0.5, const bool &ipw=true,
                   const unsigned int &nodesize=5) {
  uvec trt_uniq = unique(trt);
  unsigned int ntrt = trt_uniq.n_elem;
  unsigned int nvar = X.n_cols;
  vector<uvec> idx_trt;
  vector<vec> y_trt;
  vector<mat> X_trt;
  rowvec probs(ntrt);
  for (unsigned int t = 0; t < ntrt; ++t) {
    uvec tmp = find(trt==trt_uniq(t));
    idx_trt.push_back(tmp);
    y_trt.push_back(y.elem(tmp));
    probs(t) = mean(prob.elem(tmp));
    X_trt.push_back(X.rows(tmp));
  }
  mat utility(nvar, 2, fill::zeros);
  vec cutoffs_var(nvar, fill::zeros);
  uvec qualified(nvar);
  // umat treatments(nvar, 2);
  if (ipw) {
    for (unsigned int i = 0; i < nvar; ++i) { // go through each covariate
      vec x_uniq = unique(X.col(i)), cutoffs;
      if (x_uniq.n_elem <= 10) {
        cutoffs = x_uniq;
      } else {
        cutoffs = quantile(x_uniq, regspace(0.1,0.1,0.9));
      }
      mat util(cutoffs.n_elem, 2);
      // umat trts(cutoffs.n_elem, 2);
      uvec qual(cutoffs.n_elem);
      for (unsigned int j = 0; j < cutoffs.n_elem; ++j) {
        // go through each cutoff
        mat numer(ntrt,2), denom(ntrt,2);
        umat count(ntrt,2);
        for (unsigned int t = 0; t < ntrt; ++t) {
          vec ya = y_trt[t](find(X_trt[t].col(i)<cutoffs(j))); 
          vec yb = y_trt[t](find(X_trt[t].col(i)>=cutoffs(j)));
          count(t,0) = ya.n_elem;
          count(t,1) = yb.n_elem;
          numer(t,0) = accu(ya)/(ya.n_elem+lambda);
          numer(t,1) = accu(yb)/(yb.n_elem+lambda);
          denom(t,0) = ya.n_elem/(ya.n_elem+lambda);
          denom(t,1) = yb.n_elem/(yb.n_elem+lambda);
        }
        mat regavg = 1 - denom;
        regavg.each_row() %= sum(numer)/sum(denom);
        regavg += numer;
        uvec ids = index_max(regavg).t();
        // trts(j,0) = trt_uniq(ids(0));
        // trts(j,1) = trt_uniq(ids(1));
        urowvec count_opt = {count(ids(0),0), count(ids(1),1)};
        urowvec count_split = sum(count);
        rowvec regavg_opt = max(regavg), prob_opt = probs(ids);
        util.row(j) = regavg_opt % count_opt / prob_opt;
        qual(j) = static_cast<unsigned int>(min(count_split)>=nodesize &&
          min(sum(count!=0))>=2 && ids(0)!=ids(1));
      }
      uvec id_notqual = find(qual==0);
      util.shed_rows(id_notqual);
      // trts.shed_rows(id_notqual);
      cutoffs.shed_rows(id_notqual);
      if (util.n_rows==0) { // no qualified record for ith var
        qualified(i) = 0;
      } else {
        unsigned int id_max = index_max(sum(util, 1));
        utility.row(i) = util.row(id_max);
        // treatments.row(i) = trts.row(id_max);
        cutoffs_var(i) = cutoffs(id_max);
        qualified(i) = 1;
      }
    }
  } else {
    for (unsigned int i = 0; i < nvar; ++i) { // go through each covariate
      vec x_uniq = unique(X.col(i)), cutoffs;
      if (x_uniq.n_elem <= 10) {
        cutoffs = x_uniq;
      } else {
        cutoffs = quantile(x_uniq, regspace(0.1,0.1,0.9));
      }
      mat util(cutoffs.n_elem, 2);
      // umat trts(cutoffs.n_elem, 2);
      uvec qual(cutoffs.n_elem);
      for (unsigned int j = 0; j < cutoffs.n_elem; ++j) {
        mat numer(ntrt,2), denom(ntrt,2);
        umat count(ntrt,2);
        for (unsigned int t = 0; t < ntrt; ++t) {
          vec ya = y_trt[t](find(X_trt[t].col(i)<cutoffs(j))); 
          vec yb = y_trt[t](find(X_trt[t].col(i)>=cutoffs(j)));
          count(t,0) = ya.n_elem;
          count(t,1) = yb.n_elem;
          numer(t,0) = accu(ya)/(ya.n_elem+lambda);
          numer(t,1) = accu(yb)/(yb.n_elem+lambda);
          denom(t,0) = ya.n_elem/(ya.n_elem+lambda);
          denom(t,1) = yb.n_elem/(yb.n_elem+lambda);
        }
        mat regavg = 1 - denom;
        regavg.each_row() %= sum(numer)/sum(denom);
        regavg += numer;
        uvec ids = index_max(regavg).t();
        // trts(j,0) = trt_uniq(ids(0));
        // trts(j,1) = trt_uniq(ids(1));
        urowvec count_opt = {count(ids(0),0), count(ids(1),1)};
        urowvec count_split = sum(count);
        rowvec regavg_opt = max(regavg);
        util.row(j) = regavg_opt % count_split;
        qual(j) = static_cast<unsigned int>(min(count_split)>=nodesize &&
          min(sum(count!=0))>=2 && ids(0)!=ids(1));
      }
      uvec id_notqual = find(qual==0);
      util.shed_rows(id_notqual);
      // trts.shed_rows(id_notqual);
      cutoffs.shed_rows(id_notqual);
      if (util.n_rows==0) { // no qualified record for ith var
        qualified(i) = 0;
      } else {
        unsigned int id_max = index_max(sum(util, 1));
        utility.row(i) = util.row(id_max);
        // treatments.row(i) = trts.row(id_max);
        cutoffs_var(i) = cutoffs(id_max);
        qualified(i) = 1;
      }
    }
  }
  if (any(qualified)) {
    mat utility_qualified = utility.rows(find(qualified==1));
    vec cutoffs_var_qualified = cutoffs_var.rows(find(qualified==1));
    unsigned int idx_var = index_max(sum(utility_qualified, 1));
    return List::create(_["var"]=idx_var,
                        _["cutoff"]=cutoffs_var_qualified(idx_var),
                        // _["trt"]=treatments.row(idx_var),
                        _["util"]=utility_qualified.row(idx_var));
  } else {
    return List::create(_["var"]=-1);
  }
}

template <typename T>
uvec index_subset(T v, T sub) {
  uvec ids = find(v==sub(0));
  for (unsigned int i=1; i < sub.n_elem; ++i) {
    ids.insert_rows(0, find(v==sub(i)));
  }
  return sort(ids);
}

bool newsplit(vector<unsigned int> &vars, vector<double> &cutoffs,
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

// [[Rcpp::export]]
mat growTree_cpp(const vec &y, const mat &X, const uvec &trt, const vec &prob,
                 const mat &X_est, const mat &X_val, const uvec &ids_train,
                 const uvec &ids_est, const uvec &ids_val,
                 const unsigned int &ntrts=5, const unsigned int &nvars=3,
                 const double &lambda=0.5, const bool &ipw=true,
                 const unsigned int &nodesize=5, const double &epi=0.1) {
  // X: training matrix
  // utility at root
  uvec trt_uniq = unique(trt);
  unsigned int ntrt = trt_uniq.n_elem;
  vec numer(ntrt), denom(ntrt), probs(ntrt);
  uvec count(ntrt);
  for (unsigned int t = 0; t < ntrt; ++t) {
    uvec tmp = find(trt==trt_uniq(t));
    vec y_trt = y(tmp);
    count(t) = y_trt.n_elem;
    probs(t) = mean(prob(tmp));
    numer(t) = accu(y_trt)/(y_trt.n_elem+lambda);
    denom(t) = y_trt.n_elem/(y_trt.n_elem+lambda);
  }
  vec regavg = 1 - denom;
  regavg *= accu(numer)/accu(denom);
  regavg += numer;
  unsigned int id = index_max(regavg);
  double util_root;
  if (ipw) {
    util_root = max(regavg)*count(id)/probs(id);
  } else {
    util_root = max(regavg)*accu(count);
  }
  // grow tree
  IntegerVector parentnode = {0}, node = {1};
  set<unsigned int> nodes2split; nodes2split.insert(1);
  // filter records the row indices of training data for each node
  vector<uvec> filter;
  filter.push_back(regspace<uvec>(0, X.n_rows-1));
  vector<uvec> filter_est;
  filter_est.push_back(regspace<uvec>(0, X_est.n_rows-1));
  vector<uvec> filter_val;
  filter_val.push_back(regspace<uvec>(0, X_val.n_rows-1));
  CharacterVector type = {"split"};
  NumericVector util = {util_root};
  vector<unsigned int> vars; vector<double> cutoffs;
  // push-backs below are for root node only
  vars.push_back(X.n_cols); cutoffs.push_back(0.0);
  double mingain = epi*stddev(y);
  do {
    set<unsigned int> nodes2split_tmp = nodes2split;
    for (set<unsigned int>::iterator i=nodes2split_tmp.begin();
         i!=nodes2split_tmp.end(); ++i) {
      unsigned int node2split = *i;
      uvec ids = filter[node2split-1];
      uvec ids_est = filter_est[node2split-1];
      uvec ids_val = filter_val[node2split-1];
      uvec trt_tmp = trt(ids);
      uvec trt_sub = Rcpp::RcppArmadillo::sample(trt_uniq, ntrts, false);
      uvec var_sub = Rcpp::RcppArmadillo::sample(regspace<uvec>(0, X.n_cols-1),
                                                 nvars, false);
      uvec ids4split = ids(index_subset(trt_tmp, trt_sub));
      List split = splitting_cpp(y(ids4split), X.submat(ids4split, var_sub), 
                                 trt(ids4split), prob(ids4split),
                                 lambda, ipw, nodesize);
      int var_id = split["var"];
      if (var_id==-1) { // split is null
        type[node2split-1] = "leaf";
        nodes2split.erase(node2split);
      } else {
        rowvec utils_split = split["util"];
        if (accu(utils_split) - util[node2split-1] < mingain) {
          type[node2split-1] = "leaf";
          nodes2split.erase(node2split);
        } else {
          unsigned int var = var_sub(static_cast<unsigned int>(var_id));
          double cutoff = split["cutoff"];
          if (newsplit(vars, cutoffs, var, cutoff)) { // new split
            parentnode.push_back(node2split); parentnode.push_back(node2split);
            nodes2split.insert(max(node)+1); node.push_back(max(node)+1);
            nodes2split.insert(max(node)+1); node.push_back(max(node)+1);
            uvec var_tmp(1); var_tmp.fill(var);
            filter.push_back(ids(find(X(ids,var_tmp).as_col()<cutoff)));
            filter.push_back(ids(find(X(ids,var_tmp).as_col()>=cutoff)));
            filter_est.push_back(ids_est(find(X_est(ids_est,var_tmp).as_col()<cutoff)));
            filter_est.push_back(ids_est(find(X_est(ids_est,var_tmp).as_col()>=cutoff)));
            filter_val.push_back(ids_val(find(X_val(ids_val,var_tmp).as_col()<cutoff)));
            filter_val.push_back(ids_val(find(X_val(ids_val,var_tmp).as_col()>=cutoff)));
            type[node2split-1] = "parent";
            nodes2split.erase(node2split);
            type.push_back("split"); type.push_back("split");
            util.push_back(utils_split(0)); util.push_back(utils_split(1));
          } else {
            type[node2split-1] = "leaf";
            nodes2split.erase(node2split);
          }
        }
      }
    }
  } while (nodes2split.size() > 0); // still at least 1 node to split
  for (unsigned int i = 0; i < type.size(); ++i) {
    if (type[i]=="leaf") {
      // filter.insert(filter.begin()+i, ids_train(filter[i]));
      // filter.erase(filter.begin()+i+1);
      filter_est.insert(filter_est.begin()+i, ids_est(filter_est[i]));
      filter_est.erase(filter_est.begin()+i+1);
      filter_val.insert(filter_val.begin()+i, ids_val(filter_val[i]));
      filter_val.erase(filter_val.begin()+i+1);
    } else {
      type.erase(i);
      // filter.erase(filter.begin()+i);
      filter_est.erase(filter_est.begin()+i);
      filter_val.erase(filter_val.begin()+i);
      --i;
    }
  }
  // create weight matrix
  mat mat_wt(ids_val.n_elem, ids_train.n_elem+ids_est.n_elem, fill::zeros);
  for (unsigned int i = 0; i < type.size(); ++i) {
    mat_wt(filter_val[i], filter_est[i]) += 1.0/filter_est[i].n_elem;
  }
  return mat_wt;
}