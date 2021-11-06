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

template <typename T>
uvec index_subset(T v, T sub) {
  uvec ids = find(v==sub(0));
  for (unsigned int i = 1; i < sub.n_elem; ++i) {
    uvec tmp = find(v==sub(i));
    if (tmp.n_elem>0) ids.insert_rows(0, tmp);
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

uvec setdiff(uvec &a, uvec &b) {
  a = sort(a), b = sort(b);
  vector<unsigned int> avec = conv_to<vector<unsigned int>>::from(a),
    bvec = conv_to<vector<unsigned int>>::from(b), cvec;
  
  set_difference(avec.begin(), avec.end(), bvec.begin(), bvec.end(), 
                 inserter(cvec, cvec.end()));
  uvec c = conv_to<uvec>::from(cvec);
  return c;
}

List slice_sample(const uvec &ids, const uvec &trt, const double &prop=0.5) {
  uvec trt_uniq = sort(unique(trt));
  unsigned int ntrt = trt_uniq.n_elem;
  uvec ids_train, ids_est;
  for (unsigned int t = 0; t < ntrt; ++t) {
    uvec ids_tmp = ids(find(trt==trt_uniq(t)));
    unsigned int n_tmp = floor(prop*ids_tmp.n_elem);
    uvec ids_tmp_train = Rcpp::RcppArmadillo::sample(ids_tmp, n_tmp, false);
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
List splitting_cpp(const vec &y, const mat &X, const uvec &trt, const vec &prob,
                   const double &lambda=0.5, const bool &ipw=true,
                   const unsigned int &nodesize=5) {
  Function Rquantile("quantile"); // call R's quantile within Rcpp
  uvec trt_uniq = sort(unique(trt));
  unsigned int ntrt = trt_uniq.n_elem;
  unsigned int nvar = X.n_cols;
  // group ids, y, X, probs by trt
  vector<uvec> idx_trt; vector<vec> y_trt;
  vector<mat> X_trt; rowvec probs(ntrt);
  for (unsigned int t = 0; t < ntrt; ++t) {
    uvec tmp = find(trt==trt_uniq(t));
    idx_trt.push_back(tmp);
    y_trt.push_back(y.elem(tmp));
    probs(t) = mean(prob.elem(tmp));
    X_trt.push_back(X.rows(tmp));
  }
  // for each covariate, record two values of optimal utility, an optimal cutoff,
  // qualification status, and two optimal treatments
  mat utility(nvar, 2, fill::zeros); vec cutoffs_var(nvar, fill::zeros);
  uvec qualified(nvar); umat treatments(nvar, 2);
  if (ipw) {
    for (unsigned int i = 0; i < nvar; ++i) { // go through each covariate
      vec x_uniq = sort(unique(X.col(i))), cutoffs;
      if (x_uniq.n_elem < 10) { // fewer than 10 unique values 
        cutoffs = x_uniq;
        // adjust the least cutoff to guarantee non-empty nodes
        cutoffs(0) = mean(cutoffs.subvec(0,1)); 
      } else { // at least 10 unique values 
        // cutoffs = quantile(x_uniq, regspace(0.1,0.1,0.9)); // Armadillo
        cutoffs = as<vec>(Rquantile(x_uniq, regspace(0.1,0.1,0.9),
                                    _["type"]=7)); // R
      }
      // for each cutoff, record two values of utility, two optimal treatments,
      // and qualification status
      mat util(cutoffs.n_elem, 2); umat trts(cutoffs.n_elem, 2);
      uvec qual(cutoffs.n_elem);
      for (unsigned int j = 0; j < cutoffs.n_elem; ++j) { // go thru each cutoff
        // record numer, denom, and count for each treatment
        mat numer(ntrt,2), denom(ntrt,2); umat count(ntrt,2);
        for (unsigned int t = 0; t < ntrt; ++t) {
          vec ya = y_trt[t](find(X_trt[t].col(i)<cutoffs(j))); 
          vec yb = y_trt[t](find(X_trt[t].col(i)>=cutoffs(j)));
          count(t,0) = ya.n_elem; count(t,1) = yb.n_elem;
          numer(t,0) = accu(ya)/(ya.n_elem+lambda);
          numer(t,1) = accu(yb)/(yb.n_elem+lambda);
          denom(t,0) = ya.n_elem/(ya.n_elem+lambda);
          denom(t,1) = yb.n_elem/(yb.n_elem+lambda);
        }
        mat regavg = 1 - denom;
        regavg.each_row() %= sum(numer)/sum(denom);
        regavg += numer; // regularized avg outcomes
        rowvec regavg_opt = max(regavg); // max regavg for each column
        // record optimal treatments for both branches (may not be unique)
        uvec id0 = find(regavg.col(0)==regavg_opt(0) && count.col(0)!=0);
        uvec id1 = find(regavg.col(1)==regavg_opt(1) && count.col(1)!=0);
        uvec ids(2);
        // pick a single optimal treatment for each branch
        for (unsigned int k = 0; k < id0.n_elem; ++k) {
          ids(0) = id0(k);
          for (unsigned int l = 0; l < id1.n_elem; ++l) {
            ids(1) = id1(l);
            if (ids(0)!=id0(1)) break;
          }
          if (ids(0)!=id0(1)) break;
        }
        trts(j,0) = trt_uniq(ids(0)); trts(j,1) = trt_uniq(ids(1));
        urowvec count_opt = {count(ids(0),0), count(ids(1),1)};
        rowvec prob_opt = probs(ids);
        util.row(j) = regavg_opt % count_opt / prob_opt; // ipw adjusted
        urowvec count_split = sum(count); // column sum
        // check conditions 1 & 2 for recursive splitting
        qual(j) = static_cast<unsigned int>(min(count_split)>=nodesize &&
          min(sum(count!=0))>=2 && ids(0)!=ids(1));
      }
      uvec id_notqual = find(qual==0);
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
      vec x_uniq = sort(unique(X.col(i))), cutoffs;
      if (x_uniq.n_elem < 10) {
        cutoffs = x_uniq;
        cutoffs(0) = mean(cutoffs.subvec(0,1));
      } else {
        // cutoffs = quantile(x_uniq, regspace(0.1,0.1,0.9));
        cutoffs = as<vec>(Rquantile(x_uniq, regspace(0.1,0.1,0.9),
                                    _["type"]=7));
      }
      mat util(cutoffs.n_elem, 2);
      umat trts(cutoffs.n_elem, 2);
      uvec qual(cutoffs.n_elem);
      for (unsigned int j = 0; j < cutoffs.n_elem; ++j) {
        mat numer(ntrt,2), denom(ntrt,2);
        umat count(ntrt,2);
        for (unsigned int t = 0; t < ntrt; ++t) {
          vec ya = y_trt[t](find(X_trt[t].col(i)<cutoffs(j))); 
          vec yb = y_trt[t](find(X_trt[t].col(i)>=cutoffs(j)));
          count(t,0) = ya.n_elem; count(t,1) = yb.n_elem;
          numer(t,0) = accu(ya)/(ya.n_elem+lambda);
          numer(t,1) = accu(yb)/(yb.n_elem+lambda);
          denom(t,0) = ya.n_elem/(ya.n_elem+lambda);
          denom(t,1) = yb.n_elem/(yb.n_elem+lambda);
        }
        mat regavg = 1 - denom; regavg.each_row() %= sum(numer)/sum(denom);
        regavg += numer;
        rowvec regavg_opt = max(regavg);
        uvec id0 = find(regavg.col(0)==regavg_opt(0) && count.col(0)!=0);
        uvec id1 = find(regavg.col(1)==regavg_opt(1) && count.col(1)!=0);
        uvec ids(2);
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
        urowvec count_opt = {count(ids(0),0), count(ids(1),1)};
        urowvec count_split = sum(count);
        util.row(j) = regavg_opt % count_split;
        qual(j) = static_cast<unsigned int>(min(count_split)>=nodesize &&
          min(sum(count!=0))>=2 && ids(0)!=ids(1));
      }
      uvec id_notqual = find(qual==0);
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
    uvec ids = find(qualified==1);
    mat util_qualified = utility.rows(ids);
    umat trt_qualified = treatments.rows(ids);
    vec cutoffs_qualified = cutoffs_var.rows(ids);
    unsigned int idx_var = index_max(sum(util_qualified, 1));
    return List::create(_["var"]=ids(idx_var),
                        _["cutoff"]=cutoffs_qualified(idx_var),
                        _["trt"]=trt_qualified.row(idx_var),
                        _["util"]=util_qualified.row(idx_var));
  } else {
    return List::create(_["var"]=-1);
  }
}

// [[Rcpp::export]]
mat growTree_simpleavg(const vec &y_trainest, const mat &X_trainest,
                       const uvec &trt_trainest, const vec &prob_trainest,
                       const mat &X_val, const unsigned int &ntrts=5,
                       const unsigned int &nvars=3, const double &lambda1=0.5,
                       const double &lambda2=0.5, const bool &ipw=true,
                       const unsigned int &nodesize=5, const double &prop_train=0.5,
                       const double &epi=0.1, const bool &reg=true,
                       const bool &setseed=false, const unsigned int &seed=1) {
  if (setseed) set_seed(seed);
  // analogous to slice_sample in dplyr: sampling within each treatment slice
  List list_ids = slice_sample(regspace<uvec>(0, X_trainest.n_rows-1),
                               trt_trainest, prop_train);
  const uvec ids_train = list_ids["ids_train"], ids_est = list_ids["ids_est"];
  const uvec trt = trt_trainest(ids_train);
  const vec y = y_trainest(ids_train), prob = prob_trainest(ids_train);
  const mat X = X_trainest.rows(ids_train), X_est = X_trainest.rows(ids_est);
  // X: training matrix
  // utility at root
  uvec trt_uniq = sort(unique(trt));
  unsigned int ntrt = trt_uniq.n_elem;
  vec numer(ntrt), denom(ntrt), probs(ntrt);
  uvec count(ntrt);
  for (unsigned int t = 0; t < ntrt; ++t) {
    uvec tmp = find(trt==trt_uniq(t));
    vec y_trt = y(tmp);
    count(t) = y_trt.n_elem;
    probs(t) = mean(prob(tmp));
    numer(t) = accu(y_trt)/(y_trt.n_elem+lambda1);
    denom(t) = y_trt.n_elem/(y_trt.n_elem+lambda1);
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
  set<unsigned int> nodes2split; nodes2split.insert(1); // start from root node
  // filter records the row indices of X
  vector<uvec> filter;
  filter.push_back(regspace<uvec>(0, X.n_rows-1)); // all rows of training data
  vector<uvec> filter_est;
  filter_est.push_back(regspace<uvec>(0, X_est.n_rows-1)); // all rows of est data
  vector<uvec> filter_val;
  filter_val.push_back(regspace<uvec>(0, X_val.n_rows-1)); // all rows of val data
  CharacterVector type = {"split"}; // start from root node as split
  NumericVector util = {util_root}; // util at root node
  vector<unsigned int> vars; vector<double> cutoffs;
  // push-backs below are for root node only
  vars.push_back(X.n_cols); cutoffs.push_back(0.0); // X.n_cols larger than any index
  double mingain = epi*stddev(y);
  do {
    set<unsigned int> nodes2split_tmp = nodes2split;
    for (set<unsigned int>::iterator i=nodes2split_tmp.begin();
         i!=nodes2split_tmp.end(); ++i) {
      unsigned int node2split = *i;
      uvec ids = filter[node2split-1]; // zero-index in C++
      uvec ids_est = filter_est[node2split-1];
      uvec ids_val = filter_val[node2split-1];
      uvec trt_tmp = trt(ids); // trt column of the subsample
      uvec trt_sub = Rcpp::RcppArmadillo::sample(trt_uniq, ntrts, false);
      uvec var_sub = Rcpp::RcppArmadillo::sample(regspace<uvec>(0, X.n_cols-1),
                                                 nvars, false);
      // ids only with selected treatment levels
      uvec ids4split = ids(index_subset(trt_tmp, trt_sub)); 
      List split; int var_id;
      if (ids4split.n_elem==0) { // trt_tmp and trt_sub are disjoint--nothing to split
        var_id = -1; 
      } else { // ids4split is nonempty
        split = splitting_cpp(y(ids4split), X.submat(ids4split, var_sub),
                              trt(ids4split), prob(ids4split),
                              lambda1, ipw, nodesize);
        var_id = split["var"];
      }
      if (var_id==-1) { // split is null
        type[node2split-1] = "leaf";
        nodes2split.erase(node2split); // split unavailable; drop the node
      } else {
        rowvec utils_split = split["util"];
        if (accu(utils_split) - util[node2split-1] < mingain) { // no welfare gain
          type[node2split-1] = "leaf";
          nodes2split.erase(node2split); // split is not beneficial; drop the node
        } else { // with welfare gain
          unsigned int var = var_sub(static_cast<unsigned int>(var_id)); // var for splitting
          double cutoff = split["cutoff"];
          if (newsplit(vars, cutoffs, var, cutoff)) { // new split
            uvec var_tmp(1); var_tmp.fill(var);
            uvec ids_est_left = ids_est(find(X_est(ids_est,var_tmp).as_col()<cutoff));
            uvec ids_est_right = ids_est(find(X_est(ids_est,var_tmp).as_col()>=cutoff));
            uvec ids_val_left = ids_val(find(X_val(ids_val,var_tmp).as_col()<cutoff));
            uvec ids_val_right = ids_val(find(X_val(ids_val,var_tmp).as_col()>=cutoff));
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
              nodes2split.erase(node2split);
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
  for (unsigned int i = 0; i < type.size(); ++i) {
    if (type[i]=="leaf") {
      filter_est.insert(filter_est.begin()+i, ids_est(filter_est[i]));
      filter_est.erase(filter_est.begin()+i+1);
    } else { // drop nonterminal nodes
      type.erase(i);
      filter_est.erase(filter_est.begin()+i);
      filter_val.erase(filter_val.begin()+i);
      --i;
    }
  }
  mat mat_res;
  if (reg) { // regularized final outcomes
    uvec trts_uniq = sort(unique(trt_trainest));
    mat_res.zeros(X_val.n_rows, trts_uniq.n_elem);
    for (unsigned int i = 0; i < type.size(); ++i) { // go thru each terminal node
      uvec trt_tmp = trt_trainest(filter_est[i]); // subvector of trt column
      vec y_tmp = y_trainest(filter_est[i]); // subvector of outcome column
      urowvec count(trts_uniq.n_elem);
      rowvec numer(trts_uniq.n_elem), denom(trts_uniq.n_elem);
      for (unsigned int t = 0; t < trts_uniq.n_elem; ++t) {
        uvec id_tmp = find(trt_tmp==trts_uniq(t));
        if (id_tmp.n_elem==0) { // one trt level missing
          count(t) = 0; numer(t) = 0; denom(t) = 0;
        } else {
          numer(t) = accu(y_tmp(id_tmp))/(id_tmp.n_elem+lambda2);
          denom(t) = id_tmp.n_elem/(id_tmp.n_elem+lambda2);
        }
      }
      rowvec regavg = 1 - denom; regavg *= accu(numer)/accu(denom);
      regavg += numer;
      // assign regavg to units of val data belonging to the terminal node
      mat_res.rows(filter_val[i]) += repmat(regavg, filter_val[i].n_elem, 1);
    }
  } else { // unregularized final outcomes
    mat_res.zeros(X_val.n_rows, X_trainest.n_rows);
    for (unsigned int i = 0; i < type.size(); ++i) {
      mat_res(filter_val[i], filter_est[i]) += 1.0/filter_est[i].n_elem;
    }
  }
  return mat_res;
}

// [[Rcpp::export]]
List growTree_wtavg(const vec &y_trainest, const mat &X_trainest,
                    const uvec &trt_trainest, const vec &prob_trainest,
                    const mat &X_val, const unsigned int &ntrts=5,
                    const unsigned int &nvars=3, const double &lambda1=0.5,
                    const double &lambda2=0.5, const bool &ipw=true,
                    const unsigned int &nodesize=5, const double &prop_train=0.5,
                    const double &epi=0.1, const bool &reg=true,
                    const bool &setseed=false, const unsigned int &seed=1) {
  if (setseed) set_seed(seed);
  // analogous to slice_sample in dplyr: sampling within each treatment slice
  List list_ids = slice_sample(regspace<uvec>(0, X_trainest.n_rows-1),
                               trt_trainest, prop_train);
  const uvec ids_train = list_ids["ids_train"], ids_est = list_ids["ids_est"];
  const uvec trt = trt_trainest(ids_train);
  const vec y = y_trainest(ids_train), prob = prob_trainest(ids_train);
  const mat X = X_trainest.rows(ids_train), X_est = X_trainest.rows(ids_est);
  // X: training matrix
  // utility at root
  uvec trt_uniq = sort(unique(trt));
  unsigned int ntrt = trt_uniq.n_elem;
  vec numer(ntrt), denom(ntrt), probs(ntrt);
  uvec count(ntrt);
  for (unsigned int t = 0; t < ntrt; ++t) {
    uvec tmp = find(trt==trt_uniq(t));
    vec y_trt = y(tmp);
    count(t) = y_trt.n_elem;
    probs(t) = mean(prob(tmp));
    numer(t) = accu(y_trt)/(y_trt.n_elem+lambda1);
    denom(t) = y_trt.n_elem/(y_trt.n_elem+lambda1);
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
  set<unsigned int> nodes2split; nodes2split.insert(1); // start from root node
  // filter records the row indices of X
  vector<uvec> filter;
  filter.push_back(regspace<uvec>(0, X.n_rows-1)); // all rows of training data
  vector<uvec> filter_est;
  filter_est.push_back(regspace<uvec>(0, X_est.n_rows-1)); // all rows of est data
  vector<uvec> filter_val;
  filter_val.push_back(regspace<uvec>(0, X_val.n_rows-1)); // all rows of val data
  CharacterVector type = {"split"}; // start from root node as split
  NumericVector util = {util_root}; // util at root node
  vector<unsigned int> vars; vector<double> cutoffs;
  // push-backs below are for root node only
  vars.push_back(X.n_cols); cutoffs.push_back(0.0); // X.n_cols larger than any index
  double mingain = epi*stddev(y);
  do {
    set<unsigned int> nodes2split_tmp = nodes2split;
    for (set<unsigned int>::iterator i=nodes2split_tmp.begin();
         i!=nodes2split_tmp.end(); ++i) {
      unsigned int node2split = *i;
      uvec ids = filter[node2split-1]; // zero-index in C++
      uvec ids_est = filter_est[node2split-1];
      uvec ids_val = filter_val[node2split-1];
      uvec trt_tmp = trt(ids); // trt column of the subsample
      uvec trt_sub = Rcpp::RcppArmadillo::sample(trt_uniq, ntrts, false);
      uvec var_sub = Rcpp::RcppArmadillo::sample(regspace<uvec>(0, X.n_cols-1),
                                                 nvars, false);
      // ids only with selected treatment levels
      uvec ids4split = ids(index_subset(trt_tmp, trt_sub)); 
      List split; int var_id;
      if (ids4split.n_elem==0) { // trt_tmp and trt_sub are disjoint--nothing to split
        var_id = -1; 
      } else { // ids4split is nonempty
        split = splitting_cpp(y(ids4split), X.submat(ids4split, var_sub),
                              trt(ids4split), prob(ids4split),
                              lambda1, ipw, nodesize);
        var_id = split["var"];
      }
      if (var_id==-1) { // split is null
        type[node2split-1] = "leaf";
        nodes2split.erase(node2split); // split unavailable; drop the node
      } else {
        rowvec utils_split = split["util"];
        if (accu(utils_split) - util[node2split-1] < mingain) { // no welfare gain
          type[node2split-1] = "leaf";
          nodes2split.erase(node2split); // split is not beneficial; drop the node
        } else { // with welfare gain
          unsigned int var = var_sub(static_cast<unsigned int>(var_id)); // var for splitting
          double cutoff = split["cutoff"];
          if (newsplit(vars, cutoffs, var, cutoff)) { // new split
            uvec var_tmp(1); var_tmp.fill(var);
            uvec ids_est_left = ids_est(find(X_est(ids_est,var_tmp).as_col()<cutoff));
            uvec ids_est_right = ids_est(find(X_est(ids_est,var_tmp).as_col()>=cutoff));
            uvec ids_val_left = ids_val(find(X_val(ids_val,var_tmp).as_col()<cutoff));
            uvec ids_val_right = ids_val(find(X_val(ids_val,var_tmp).as_col()>=cutoff));
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
              nodes2split.erase(node2split);
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
  for (unsigned int i = 0; i < type.size(); ++i) {
    if (type[i]=="leaf") {
      filter_est.insert(filter_est.begin()+i, ids_est(filter_est[i]));
      filter_est.erase(filter_est.begin()+i+1);
    } else { // drop nonterminal nodes
      type.erase(i);
      filter_est.erase(filter_est.begin()+i);
      filter_val.erase(filter_val.begin()+i);
      --i;
    }
  }
  mat mat_res, mat_ct;
  if (reg) {
    uvec trts_uniq = sort(unique(trt_trainest));
    mat_res.zeros(X_val.n_rows, trts_uniq.n_elem);
    mat_ct.zeros(X_val.n_rows, trts_uniq.n_elem);
    for (unsigned int i = 0; i < type.size(); ++i) { // go thru each terminal node
      uvec trt_tmp = trt_trainest(filter_est[i]); // subvector of trt column
      vec y_tmp = y_trainest(filter_est[i]); // subvector of outcome column
      urowvec count(trts_uniq.n_elem);
      rowvec numer(trts_uniq.n_elem), denom(trts_uniq.n_elem);
      for (unsigned int t = 0; t < trts_uniq.n_elem; ++t) {
        uvec id_tmp = find(trt_tmp==trts_uniq(t));
        if (id_tmp.n_elem==0) { // one trt level missing
          count(t) = 0; numer(t) = 0; denom(t) = 0;
        } else {
          numer(t) = accu(y_tmp(id_tmp))/(id_tmp.n_elem+lambda2);
          denom(t) = id_tmp.n_elem/(id_tmp.n_elem+lambda2);
        }
      }
      rowvec regavg = 1 - denom; regavg *= accu(numer)/accu(denom);
      regavg += numer;
      for (unsigned int t = 0; t < trts_uniq.n_elem; ++t) {
        if (count(t)==0) regavg(t) = 0;
      }
      rowvec regwt(regavg.n_elem, fill::zeros);
      regwt(find(count)) += 1.0;
      // assign regavg to units of val data belonging to the terminal node
      mat_res.rows(filter_val[i]) += repmat(regavg, filter_val[i].n_elem, 1);
      mat_ct.rows(filter_val[i]) += repmat(regwt, filter_val[i].n_elem, 1);
    }
  } else {
    mat_res.zeros(X_val.n_rows, X_trainest.n_rows);
    mat_ct.zeros(X_val.n_rows, X_trainest.n_rows);
    for (unsigned int i = 0; i < type.size(); ++i) {
      mat_res(filter_val[i], filter_est[i]) += 1.0/filter_est[i].n_elem;
      mat_ct(filter_val[i], filter_est[i]) += 1.0;
    }
  }
  return List::create(_["res"]=mat_res, _["ct"]=mat_ct);
}

// [[Rcpp::export]]
List growForest_cpp(const vec &y_trainest, const mat &X_trainest,
                    const uvec &trt_trainest, const vec &prob_trainest,
                    const mat &X_val,
                    const unsigned int &ntrts=5, const unsigned int &nvars=3,
                    const double &lambda1=0.5, const double &lambda2=0.5,
                    const bool &ipw=true, const unsigned int &nodesize=5,
                    const unsigned int &ntree=1000,
                    const double &prop_train=0.5, const double &epi=0.1,
                    const bool &reg=true, const bool &simpleavg=false,
                    const bool &setseed=false, const unsigned int &seed=1) {
  if (setseed) set_seed(seed);
  uvec trt_uniq = sort(unique(trt_trainest));
  unsigned int ntrt = trt_uniq.n_elem;
  mat outcome(X_val.n_rows, ntrt, fill::zeros);
  if (simpleavg) {
    if (reg) {
      for (unsigned int i = 0; i < ntree; ++i) {
        outcome += growTree_simpleavg(y_trainest, X_trainest, trt_trainest,
                                      prob_trainest, X_val, ntrts, nvars, lambda1,
                                      lambda2, ipw, nodesize, prop_train, epi, reg);
      }
      outcome /= ntree;
    } else {
      mat mat_ct(X_val.n_rows, X_trainest.n_rows, fill::zeros);
      for (unsigned int i = 0; i < ntree; ++i) {
        mat_ct += growTree_simpleavg(y_trainest, X_trainest, trt_trainest,
                                     prob_trainest, X_val, ntrts, nvars, lambda1,
                                     lambda2, ipw, nodesize, prop_train, epi, reg);
      }
      mat_ct /= ntree;
      for (unsigned int t = 0; t < ntrt; ++t) {
        uvec tmp = find(trt_trainest==trt_uniq(t));
        // weighted outcomes based on trainest data
        outcome.col(t) = mat_ct.cols(tmp)*y_trainest(tmp)/sum(mat_ct.cols(tmp),1);
      }
    }
  } else {
    if (reg) {
      mat ct(X_val.n_rows, ntrt, fill::zeros);
      for (unsigned int i = 0; i < ntree; ++i) {
        List tree_tmp = growTree_wtavg(y_trainest, X_trainest, trt_trainest,
                                       prob_trainest, X_val, ntrts, nvars, lambda1,
                                       lambda2, ipw, nodesize, prop_train, epi, reg);
        mat outcome_tmp = tree_tmp["res"], ct_tmp = tree_tmp["ct"];
        outcome += outcome_tmp;
        ct += ct_tmp;
      }
      ct.replace(0.0, 1.0);
      outcome /= ct;
    } else {
      mat mat_ct(X_val.n_rows, X_trainest.n_rows, fill::zeros);
      mat ct(X_val.n_rows, X_trainest.n_rows, fill::zeros);
      for (unsigned int i = 0; i < ntree; ++i) {
        List tree_tmp = growTree_wtavg(y_trainest, X_trainest, trt_trainest,
                                       prob_trainest, X_val, ntrts, nvars, lambda1,
                                       lambda2, ipw, nodesize, prop_train, epi, reg);
        mat outcome_tmp = tree_tmp["res"], ct_tmp = tree_tmp["ct"];
        mat_ct += outcome_tmp;
        ct += ct_tmp;
      }
      ct.replace(0.0, 1.0);
      mat_ct /= ct;
      for (unsigned int t = 0; t < ntrt; ++t) {
        uvec tmp = find(trt_trainest==trt_uniq(t));
        // weighted outcomes based on trainest data
        outcome.col(t) = mat_ct.cols(tmp)*y_trainest(tmp)/sum(mat_ct.cols(tmp),1);
      }
    }
  }
  // identify optimal trt idx for each unit in val data
  uvec idx_trt = index_max(outcome, 1);
  uvec trt_pred(idx_trt.n_elem, fill::zeros);
  for (unsigned int t = 0; t < ntrt; ++t) {
    trt_pred(find(idx_trt==t)) += trt_uniq(t);
  }
  return List::create(_["Y.pred"]=max(outcome, 1), _["trt.dof"]=trt_pred);
}