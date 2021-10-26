
/**
 * We will be parallelising the critical regions on our own, so we will
 * disable Eigen's omp implementation.
 */
//#define EIGEN_DONT_PARALLELIZE
/**
 * On the other hand, after benchmarking it, it seems like it can be faster
 * if we disabled Eigen's parallelize, but we obtain more consistent results
 * with it being on, likely because the small thread benefits helps in the
 * non-tight loop portion (also when K is less than the max number of threads.
 */

#include "gmm.hpp"
#include <cmath>

//#define STATS_ENABLE_EIGEN_WRAPPERS
//#include "stats/include/stats.hpp"

#include "soraxas_cpp_toolbox/main.h"

#ifdef SXS_GMM_USE_AUTODIFF
#include "soraxas_cpp_toolbox/batched_autodiff.hpp"
#endif

#define fast_gmm_assert(condition, message)                                    \
                                                                               \
  if (!(condition)) {                                                          \
    std::stringstream ss;                                                      \
    ss << message;                                                             \
    throw std::runtime_error(ss.str());                                        \
  }

#define svmpc_eigen_shape_as_str(mat)                                          \
  "[" << mat.rows() << "," << mat.cols() << "]"

#define assert_eigen_mat_shape__msg(mat, r, c)                                 \
  "In " << __FUNCTION__ << ": Expect " #mat " to have shape [" << r << ","     \
        << c << "], but was " << svmpc_eigen_shape_as_str(mat)

#define assert_eigen_mat_shape(mat, r, c)                                      \
  fast_gmm_assert(mat.rows() == r && mat.cols() == c,                          \
                                                                               \
                  assert_eigen_mat_shape__msg(mat, r, c))

#define assert_eigen_mat_shape_row(mat, r)                                     \
  fast_gmm_assert(mat.rows() == r, assert_eigen_mat_shape__msg(mat, r, "_"))

#define assert_eigen_mat_shape_col(mat, c)                                     \
  fast_gmm_assert(mat.cols() == c, assert_eigen_mat_shape__msg(mat, "_", c))

// helper alias to easier-to-understand name
#define assert_eigen_mat_shape_data_t(mat, K)                                  \
  fast_gmm_assert(mat.rows() == 1 && mat.cols() == K,                          \
                                                                               \
                  assert_eigen_mat_shape__msg(mat, 1, K))

#define assert_eigen_mat_shape_batched_data_t(mat, K)                          \
  fast_gmm_assert(mat.cols() == K, assert_eigen_mat_shape__msg(mat, 1, K))

namespace gmm {

template <typename eT>
TemplateBatchedData_t<eT> chol(const TemplateBatchedData_t<eT> &X) {
  return X.llt().matrixL();
}

template <typename eT> eT log_det(const TemplateBatchedData_t<eT> &X) {
  return (chol(X).diagonal().array().log() * 2.0).sum();
}

template <typename dtype>
MultivariateGaussian<dtype>::MultivariateGaussian(
    const Data_t &mu_par, const BatchedData_t &sigma_par)
    : K(mu_par.cols()) {
  m_mu = mu_par;
  set_sigma(sigma_par);
}

template <typename dtype>
dtype MultivariateGaussian<dtype>::_get_sigma_log_det() const {
  return log_det(m_sigma);
}

template <typename dtype>
const typename MultivariateGaussian<dtype>::Data_t &
MultivariateGaussian<dtype>::get_mu() const {
  return m_mu;
}

template <typename dtype>
const typename MultivariateGaussian<dtype>::BatchedData_t &
MultivariateGaussian<dtype>::get_sigma() const {
  return m_sigma;
}

template <typename dtype>
void MultivariateGaussian<dtype>::set_mu(const Data_t &mu) {
  assert_eigen_mat_shape_data_t(mu, K);
  m_mu = mu;
}

template <typename dtype>
void MultivariateGaussian<dtype>::set_sigma(const BatchedData_t &sigma) {
  assert_eigen_mat_shape_batched_data_t(sigma, K);
  // update the stored sigma
  m_sigma = sigma;
  _set_sigma_post_process();
}

template <typename dtype>
void MultivariateGaussian<dtype>::_set_sigma_post_process() {
  // update the cached QR decomposition of a matrix with column-pivoting.
  // https://eigen.tuxfamily.org/dox/classEigen_1_1ColPivHouseholderQR.html
  // rank-revealing QR decomposition of a matrix A into matrices P, Q and R
  // such that AP = QR
  //  m_cached_sigma_piv = m_sigma.colPivHouseholderQr();
  m_cached_sigma_inv = m_sigma.inverse(); // might be numerically unstable.

#ifdef SXS_GMM_USE_AUTODIFF
  // cache a dual version for auto-diff purpose.
  m_cached_dual_sigma_inv = m_cached_sigma_inv;
#endif

  // cache the log determinate of sigma
  m_sigma_log_det = _get_sigma_log_det();
}

/*
template <typename dtype>
dtype MultivariateGaussian<dtype>::log_prob_single(const Data_t &X) const {
  //    return _internal_log_prob_single<dtype>(m_cached_sigma_piv, X);
  return _internal_log_prob_single<dtype>(m_cached_sigma_inv, X);
}

template <typename dtype>
std::pair<typename MultivariateGaussian<dtype>::Data_t, dtype>
MultivariateGaussian<dtype>::log_prob_single_grad(const Data_t &x) const {
  // copy data to dual
  DataDual_t dual_x = x;

  // perform diff
  dual u;
  Eigen::Matrix<dtype, Eigen::Dynamic, 1> dudx = gradient(
      [this](auto &&_x) {
        return _internal_log_prob_single<dual>(m_cached_dual_sigma_inv, _x);
      },
      wrt(dual_x), at(dual_x), u);

  return {dudx, autodiff::val(u)};
}
*/

template <typename dtype>
template <typename eT>
TemplateBatch_t<eT> MultivariateGaussian<dtype>::log_prob(
    const TemplateBatchedData_t<eT> &X) const {
  if constexpr (std::is_same<eT, dtype>::value) {
    return _internal_log_prob<dtype>(m_cached_sigma_inv, X);
  }
#ifdef SXS_GMM_USE_AUTODIFF
  else if constexpr (std::is_same<eT, dual>::value) {
    return _internal_log_prob<dual>(m_cached_dual_sigma_inv, X);
  }
#endif
  throw std::runtime_error("Not implemented");
}

template <typename dtype>
std::pair<typename MultivariateGaussian<dtype>::BatchedData_t,
          typename MultivariateGaussian<dtype>::Batch_t>
MultivariateGaussian<dtype>::log_prob_and_grad(const BatchedData_t &X) const {

  const auto cons_term = K * neg_HALF_LOG_2PI;
  BatchedData_t X_minus_mu = X.rowwise() - m_mu;

  // the following is the gradient (except we haven't negated it yet.)
  BatchedData_t non_negated_grad = X_minus_mu * m_cached_sigma_inv;

  // quad form is X * inv * X.transpose()
  // which is = non_negated_grad * X.transpose()

  const Batch_t logprob =
      cons_term - 0.5 * (m_sigma_log_det +
                         (non_negated_grad * X_minus_mu.transpose()).array());

  return {-non_negated_grad, logprob};
}

template <typename dtype>
typename MultivariateGaussian<dtype>::BatchedData_t
MultivariateGaussian<dtype>::log_prob_grad(const BatchedData_t &X) const {

#define FASTGMM_MN_USE_CLOSED_FORM
  // https://stackoverflow.com/questions/13299642/how-to-calculate-derivative-of-multivariate-normal-probability-density-function

#ifdef FASTGMM_MN_USE_CLOSED_FORM

  auto X_minus_mu = X.rowwise() - m_mu;

  return -(X_minus_mu * m_cached_sigma_inv);

#else

  // copy data to dual
  BatchedDataDual_t dual_X = X;

  // perform diff
  BatchDual_t U(X.rows());
  BatchedData_t dudx = sxs::batched_gradient_single_var(
      [this](auto &&_X) {
        return _internal_log_prob<dual>(m_cached_dual_sigma_inv, _X);
      },
      dual_X, at(dual_X), U);

  return dudx;

#endif
}

#ifdef SXS_GMM_USE_AUTODIFF
template <typename dtype>
typename MultivariateGaussian<dtype>::BatchedData_t
MultivariateGaussian<dtype>::NotWorking_log_prob_grad(
    const BatchedDataDual_t &X) const {
  // The following doesn't work as autodiff doesn't work in a batch settings
  // >> return _internal_log_prob<dual>(m_cached_dual_sigma_piv, X);

  BatchedData_t grads(X.rows(), X.cols());

  //// Initialize the static x in all threads
  //#pragma omp parallel for
  //    for (int i = 0; i < X.rows(); ++i) {
  //      DataDual_t one_x = X.row(i);
  //
  //      dual uu;
  //      grads.row(i) =
  //          gradient([this](auto &&a) { return log_prob_single(a); },
  //          wrt(one_x),
  //                   at(one_x)); // evaluate the function value u and its
  //                   gradient
  //    }
  return grads;
}
#endif

template <typename dtype>
template <typename eT>
eT MultivariateGaussian<dtype>::_internal_log_prob_single(
    const TemplateBatchedData_t<eT> &inv, const TemplateData_t<eT> &X) const {
  /*
   * Return the log likelihood of x with the stored mu and sigma.
   */
  assert_eigen_mat_shape_data_t(X, K);

  const eT cons_term = K * neg_HALF_LOG_2PI; // -0.5 * K * log(2pi)

  const eT quad_term = quad_form<eT>(inv, X - m_mu);

  const auto ret = cons_term - 0.5 * (m_sigma_log_det + quad_term);

  return ret;
}

/*
// explicit instanciation to generate for `dual` type and `double` type:
template double MultivariateGaussian<dtype>::_internal_log_prob_single<double>(
    const BatchedData_t &, const TemplateData_t<double> &) const;

template dual MultivariateGaussian<dtype>::_internal_log_prob_single<dual>(
    const BatchedDataDual_t &, const TemplateData_t<dual> &) const;
*/

template <typename dtype>
template <typename eT>
TemplateBatch_t<eT> MultivariateGaussian<dtype>::_internal_log_prob(
    const TemplateBatchedData_t<eT> &inv,
    const TemplateBatchedData_t<eT> &X) const {

  // X [rows,cols] where rows is a batch.
  //    assert_eigen_mat_shape_batched_data_t(X, K);

  // K is the number of events (i.e. K=1 for univariate distributions)
  const auto cons_term = K * neg_HALF_LOG_2PI;

  //    const auto X_minus_mu = X.rowwise() - m_mu;

  /*
  TemplateArrayData_t<eT> quad_term =
      TemplateArrayData_t<eT>::Zero(X_minus_mu.rows());

  // this is a tight loop, we will run it in parallel.
#pragma omp parallel for
  for (size_t i = 0; i < X_minus_mu.rows(); ++i) {
    quad_term[i] = quad_form<eT>(inv, X_minus_mu.row(i));
  }
   */
  /*
#pragma omp parallel // num_threads(2)
  {
    // work from tid to batch size
    const long tid = omp_get_thread_num();
    const long batch_size = (N / omp_get_num_threads()) + 1;

    const int from = (tid * batch_size);
    const int to = std::min(from + batch_size, N);

    if (from < to) {

      BatchedData_t vvv = X_minus_mu.block(from, 0, to, K);

      quad_form_batch<eT>(inv, vvv);
    }
  }
 */

  // the function returns a row vector. we will covert it to a column
  // vector by transpose
  const TemplateBatch_t<eT> ret =
      cons_term -
      0.5 * (m_sigma_log_det + quad_form_batch<eT>(inv, X.rowwise() - m_mu));

  return ret;
}

/*
// explicit instanciation to generate for `dual` type and `double` type:
template Batch_t
MultivariateGaussian<dtype>::log_prob<dtype>(const BatchedData_t &X) const;

template BatchDual_t
MultivariateGaussian<dtype>::log_prob<dual>(const BatchedDataDual_t &X) const;
*/

// MultivariateGaussian<dtype>::_internal_log_prob_single<double>;

// double MultivariateGaussian<dtype>::_quad_form(const Data_t &x) const {
//  // performs A^-1 with the cached QR decomposition (A as the covariance mat)
//
//  // we are using row-vector so we will swap the transpose ops. The following
//  // is what would have happened if we were using col-vec
//  // >> const auto Ax = m_cached_sigma_piv.solve(x);
//  // >> return static_cast<double>((x.transpose() * Ax)(0));
//
//  const auto Ax = m_cached_sigma_piv.solve(x.transpose());
//
//  return static_cast<double>((x * Ax)(0));
//}

template <typename dtype>
MultivariateGaussianDiagCov<dtype>::MultivariateGaussianDiagCov(
    const Data_t &mu_par, const Data_t &sigma_as_diagonal)
    : MultivariateGaussian<dtype>(mu_par.cols()) {
  this->m_mu = mu_par;
  set_sigma(sigma_as_diagonal);
}

template <typename dtype>
dtype MultivariateGaussianDiagCov<dtype>::_get_sigma_log_det() const {
  // using naive method to compute log det for diagonal matrix
  return std::log(m_sigma_diag.prod());
}

template <typename dtype>
void MultivariateGaussianDiagCov<dtype>::set_sigma(const Data_t &sigma) {
  assert_eigen_mat_shape_data_t(sigma, this->K);

  // update the stored sigma
  m_sigma_diag = sigma;
  this->m_sigma = BatchedData_t(sigma.asDiagonal());

  this->_set_sigma_post_process();
}

template <typename dtype>
const typename MultivariateGaussianDiagCov<dtype>::Data_t
MultivariateGaussianDiagCov<dtype>::get_sigma() const {
  return m_sigma_diag;
}

template <typename dtype>
MultivariateGaussianMixtureModelDiagCov<
    dtype>::MultivariateGaussianMixtureModelDiagCov(BatchedData_t mu,
                                                    BatchedData_t sigma_diag,
                                                    ArrayBatch_t phi)
    : K(mu.cols()) {
  using namespace std;

  fast_gmm_assert(mu.cols() == sigma_diag.cols() &&
                      mu.rows() == sigma_diag.rows(),
                  "size mis-match: mu[n,m] sigma_diag[n,m], but was "
                      << svmpc_eigen_shape_as_str(mu) << " and "
                      << svmpc_eigen_shape_as_str(sigma_diag));
  fast_gmm_assert(mu.rows() == phi.rows() && phi.cols() == 1,
                  "size mis-match: mu[n,_] ... phi[n,1] but was "
                      << svmpc_eigen_shape_as_str(mu) << " and "
                      << svmpc_eigen_shape_as_str(phi));
  fast_gmm_assert(std::abs(phi.sum() - 1) < 1e-5,
                  "phi must sums to 1, but was: " << phi.sum() << " from "
                                                  << phi.transpose());

  m_gaus.reserve(mu.rows());
  for (size_t i = 0; i < mu.rows(); ++i) {
    // each row represents a multivariate gaussian
    m_gaus.emplace_back(mu.row(i), sigma_diag.row(i));
  }
  m_phi = phi;
  m_log_phi = phi.log();
}

template <typename dtype>
typename MultivariateGaussianMixtureModelDiagCov<dtype>::BatchedData_t
MultivariateGaussianMixtureModelDiagCov<dtype>::e_step(
    const BatchedData_t &x) const {
  // very similar to log-prob, except we keep track of heuristic

  // a 2d array of K by num-data-points
  ArrayBatchedData_t log_p_y_x(m_gaus.size(), x.rows());

  // for each mixture, compute (in a batch) the log probability of the
  // batched data-points
  // the following is performing log(phi) + log(pdf(x)), across all xs

  // This is the tight loop and take the longest
  for (size_t i = 0; i < m_gaus.size(); ++i) {
    log_p_y_x.block(i, 0, 1, x.rows()) = m_gaus[i].log_prob(x).transpose();
  }
  log_p_y_x.colwise() += m_log_phi;

  ArrayData_t log_p_y_x_norm = log_p_y_x.exp().colwise().sum().log();

  return (log_p_y_x.rowwise() - log_p_y_x_norm).exp();
}

template <typename dtype>
typename MultivariateGaussianMixtureModelDiagCov<dtype>::BatchedData_t
MultivariateGaussianMixtureModelDiagCov<dtype>::em_step_for_mu(
    const BatchedData_t &X) const {

  // compute heuristics from input data-point xs, in a batch
  BatchedData_t heuristics = e_step(X);

  // sum across each datapoint x (row is pts and col is multivariate-gauss)
  ArrayData_t sum_heuristics = heuristics.rowwise().sum();

  // heuristics matrix dot product with data points in X
  return (heuristics * X).array() / sum_heuristics;
}

/*
template <typename dtype>
dtype MultivariateGaussianMixtureModelDiagCov<dtype>::log_prob_single(
    const Data_t &x) const {
  assert_eigen_mat_shape_batched_data_t(x, K);

  // performs log-sum-exp
  dtype result;
  for (size_t i = 0; i < m_gaus.size(); ++i) {
    result += std::exp(m_gaus[i].log_prob_single(x) + m_log_phi[i]);
  }
  return std::log(result);
}
*/

template <typename dtype>
template <typename eT>
TemplateData_t<eT> MultivariateGaussianMixtureModelDiagCov<dtype>::log_prob(
    const TemplateBatchedData_t<eT> &X) const {
  // Data_t MultivariateGaussianMixtureModelDiagCov<dtype>::log_prob(
  //    const BatchedData_t &x) const {
  assert_eigen_mat_shape_batched_data_t(X, K);

  // performs log-sum-exp
  TemplateArrayData_t<eT> results = TemplateArrayData_t<eT>::Zero(X.rows());

  // define implementation to let omp knows how to reduce the Eigen array type
#pragma omp declare reduction( + : TemplateArrayData_t<eT> : omp_out += omp_in) \
  initializer( omp_priv = TemplateArrayData_t<eT>::Zero(omp_orig.size()) )

  // loop through all mixtures
#pragma omp parallel for default(none) shared(X) reduction(+ : results)
  for (size_t i = 0; i < m_gaus.size(); ++i) {
    results += (m_gaus[i].log_prob(X).array() + m_log_phi[i]).exp();
  }

  return results.log();
}

/*
// explicit instanciation to generate for `dual` type and `double` type:
template TemplateData_t<double>
MultivariateGaussianMixtureModelDiagCov<dtype>::log_prob<double>(
    const TemplateBatchedData_t<double> &) const;

template TemplateData_t<dual>
MultivariateGaussianMixtureModelDiagCov<dtype>::log_prob<dual>(
    const TemplateBatchedData_t<dual> &) const;
*/

template <typename dtype>
TemplateBatchedData_t<dtype>
MultivariateGaussianMixtureModelDiagCov<dtype>::log_prob_grad(
    const BatchedData_t &X) const {

#define FASTGMM_GMM_USE_HAND_DERIVED

#ifdef FASTGMM_GMM_USE_HAND_DERIVED

  ////////////
  ArrayBatch_t gmm_prob_accumulator = ArrayBatch_t::Zero(X.rows());
  // define implementation to let omp knows how to reduce the Eigen array type
#pragma omp declare reduction(+ : ArrayBatch_t : omp_out +=omp_in) \
  initializer( omp_priv = ArrayBatch_t::Zero(omp_orig.size()) )

  ////////////
  ArrayBatchedData_t dudx = ArrayBatchedData_t::Zero(X.rows(), X.cols());
  // define implementation to let omp knows how to reduce the Eigen array type
#pragma omp declare reduction(+ : ArrayBatchedData_t : omp_out +=omp_in) \
  initializer( omp_priv = ArrayBatchedData_t::Zero(omp_orig.rows(), omp_orig.cols()) )

  const auto cons_term = K * neg_HALF_LOG_2PI;

// loop through all mixtures
#pragma omp parallel for default(none) shared(cons_term, X) \
    reduction(+: dudx) reduction(+ : gmm_prob_accumulator)
  for (size_t i = 0; i < m_gaus.size(); ++i) {
    /**
     * Here we are computing all necessary info within the same loop.
     * VERY messy... :L
     */

    // Common variables
    // For individual mvn
    const BatchedData_t X_minus_mu = X.rowwise() - m_gaus[i].m_mu;
    const BatchedData_t X_minus_mu__mul__sigma_inv =
        (X_minus_mu * m_gaus[i].m_cached_sigma_inv);

    // For individual mvn calculation
    const ArrayBatchedData_t quad_form =
        (X_minus_mu__mul__sigma_inv.array() * X_minus_mu.array())
            .rowwise()
            .sum();

    // mvn logprob for gmm logprob and gmm grad
    const ArrayBatch_t mvn_logprob =
        cons_term - 0.5 * (m_gaus[i].m_sigma_log_det + quad_form);

    // compute logprob together, as we need to divide the (un-log) prob
    // for the gmm's final gradient
    gmm_prob_accumulator += (mvn_logprob + m_log_phi[i]).exp();

    // the actual gradient that we are interested.
    dudx +=
        /* we need to negate `X_minus_mu__mul__sigma_inv` for it to be
         * the actual `mvn_logprob_grad` that we need. */
        -X_minus_mu__mul__sigma_inv.array()
        /* --I'm a separator-- */
        * (mvn_logprob.array() + m_log_phi[i]).exp();
  }

  // notice that `gmm_prob_accumulator` is NOT log-ed, as part of the formula
  TemplateArrayBatchedData_t<dtype> result =
      dudx.colwise() / gmm_prob_accumulator;

  /**
   * Note that with the above approach, it sometimes makes this function
   * produce a different results than PyTorch's autograd; in particular, for
   * situation where we would have produce NaN, PyTorch produce extremely
   * large gradient (~1000) as opposite to the usual range (<1). Conceptually
   * I think this approach is correct.
   */
  /*
   * Replace any entry with NaN to 0. Happens when item within
   * `gmm_prob_accumulator` is zero, which occurs if it has 0 probability.
   */
  return result.isFinite().select(result, 0);

  /**
   * The following is a naive easy-to-follow approach
   *
   * for (size_t i = 0; i < m_gaus.size(); ++i) {
   *     dudx += m_gaus[i].log_prob_grad(X).array() *
   *         (m_gaus[i].log_prob(X).array() + m_log_phi[i]).exp();
   * }
   * return (dudx.array() / log_prob(X).array().exp()).transpose();
   */

#else

  // copy data to dual
  BatchedDataDual_t dual_X = X;

  // perform diff
  BatchDual_t U(X.cols());
  BatchedData_t dudx = sxs::batched_gradient_single_var(
      [this](auto &&_X) { return log_prob<dual>(_X); },

      dual_X, at(dual_X), U);

  return dudx;

#endif
}

template <typename dtype>
MultivariateGaussianDiagCov<dtype> &
MultivariateGaussianMixtureModelDiagCov<dtype>::get_gaus(size_t i) {
  fast_gmm_assert(i >= 0 && i < m_gaus.size(),
                  "Given input is out of range [0," << m_gaus.size() << ")");

  return m_gaus[i];
}

} // namespace gmm