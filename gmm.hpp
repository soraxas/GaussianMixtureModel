#pragma once

#include <Eigen/Dense>

//#define SXS_GMM_USE_AUTODIFF

#ifdef SXS_GMM_USE_AUTODIFF
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#endif

namespace gmm {

/**
 *
 * Note that RowVector shape is 1, Dynamic,
 * and     [Col]Vector shape is Dynamic, 1.
 *
 * We go against Eigen's convention because we are building this mainly for
 * python, and Pyhton uses row-major (rather than column-major). So using
 * row-major will be more memory and cache friendly.
 *
 * More over, because numpy and torch tends to be BxN where B is batch, so
 * it's better to use row in [row,col] to denote batch concept.
 */

using namespace Eigen;
#ifdef SXS_GMM_USE_AUTODIFF
using namespace autodiff;
#endif

/**
 * The following defines typedef used throughout the codebase
 */

// all matrix type
template <typename eT>
using TemplateData_t = Matrix<eT, 1, Dynamic, RowMajor>; // Row-vector in Eigen
template <typename eT> using TemplateBatch_t = Matrix<eT, Dynamic, 1>;
template <typename eT>
using TemplateBatchedData_t = Matrix<eT, Dynamic, Dynamic, RowMajor>;

// all array type
template <typename eT>
using TemplateArrayData_t =
    Array<eT, 1, Dynamic, RowMajor>; // Row-vector in Eigen
template <typename eT> using TemplateArrayBatch_t = Array<eT, Dynamic, 1>;
template <typename eT>
using TemplateArrayBatchedData_t = Array<eT, Dynamic, Dynamic, RowMajor>;

////////////////////////////////////////////////////////////////////////
// Helpers

// The following is for non-square matrix, or numerically unstable covariance
// matrix.
/*
 * Reasons of using/not-using pseudo inverse like Moore-Penrose or QR:
 * https://stats.stackexchange.com/questions/394734/which-is-more-numerically-stable-for-ols-pinv-vs-qr
 "That said, in most cases it is not good practice to use the Moore-Penrose
 Pseudo-inverse unless we have a very good reason (e.g. our procedure
 consistently employs small and potentially rank-degenerate
 covariance matrices)"
 */
/*
template <typename eT>
eT quad_form(
    const ColPivHouseholderQR<TemplateBatchedData_t<eT>> &decomposed_qr,
    const TemplateData_t<eT> &x) {
  // performs A^-1 with the cached QR decomposition (A as the covariance mat)

  // we are using row-vector so we will swap the transpose ops. The following
  // is what would have happened if we were using col-vec
  // >> const auto Ax = m_cached_sigma_piv.solve(x);
  // >> return static_cast<double>((x.transpose() * Ax)(0));

  const auto Ax = decomposed_qr.solve(x.transpose());

  //  return static_cast<eT>((x * Ax)(0));
  return static_cast<eT>(x * Ax);
}
 */

// constants as compile time constant
constexpr long double LOG_2PI = 1.8378770664093454835606594728112352797228L;
constexpr double neg_HALF_LOG_2PI = -0.5 * LOG_2PI;

/**
 * The following is for square matrix and compute quad_form using inverse mat
 *
 * @tparam eT - value type
 * @param A_inv - inversed square matrix
 * @param x - a row vector
 * @return real value of the resulting quad form
 */
template <typename eT>
eT quad_form(const TemplateBatchedData_t<eT> &A_inv,
             const TemplateData_t<eT> &x) {
  // we are using row-vector so we will swap the transpose ops. The following
  // is what would have happened if we were using col-vec
  // >> const auto Ax = m_cached_sigma_piv.solve(x);
  // >> return static_cast<double>((x.transpose() * Ax)(0));

  return x * A_inv * x.transpose();
}

/**
 * The following is for square matrix and compute quad_form using inverse mat,
 * and performing quad in a batch
 *
 * @tparam eT - value type
 * @param A_inv - inversed square matrix
 * @param x - a row vector
 * @return batched values of the resulting quad form
 */
template <typename eT>
auto quad_form_batch(const TemplateBatchedData_t<eT> &A_inv,
                     const TemplateBatchedData_t<eT> &X) {
  // Note that X * A_inv (i.e. X being mat rather than column vector)
  // will produce a matrix. So XA will only need to do a dot product along row
  // to output the actual result, as a way to BATCH the quad_form computation.

  return ((X * A_inv).array() * X.array()).rowwise().sum();
}

////////////////////////////////////////////////////////////////////////

#define SXS_DECLARE_DTYPE_DATA(dtype)                                          \
  /* all matrix type */                                                        \
  /* scalar */                                                                 \
  using Data_t = TemplateData_t<dtype>; /* Row-vector in Eigen */              \
  using Batch_t = TemplateBatch_t<dtype>;                                      \
  using BatchedData_t = TemplateBatchedData_t<dtype>;                          \
                                                                               \
  /* all array type */                                                         \
  /* scalar */                                                                 \
  using ArrayData_t = TemplateArrayData_t<dtype>;                              \
  using ArrayBatch_t = TemplateArrayBatch_t<dtype>;                            \
  using ArrayBatchedData_t = TemplateArrayBatchedData_t<dtype>;

#ifdef SXS_GMM_USE_AUTODIFF
#define SXS_DECLARE_DTYPE_DUALDATA(dtype)                                      \
  /* dual matrix type */                                                       \
  using DataDual_t = TemplateData_t<autodiff::forward::Dual<dtype, dtype>>;    \
  using BatchDual_t = TemplateBatch_t<autodiff::forward::Dual<dtype, dtype>>;  \
  using BatchedDataDual_t =                                                    \
      TemplateBatchedData_t<autodiff::forward::Dual<dtype, dtype>>;            \
                                                                               \
  /* dual array type */                                                        \
  using ArrayDataDual_t =                                                      \
      TemplateArrayData_t<autodiff::forward::Dual<dtype, dtype>>;              \
  using ArrayBatchDual_t =                                                     \
      TemplateArrayBatch_t<autodiff::forward::Dual<dtype, dtype>>;             \
  using ArrayBatchedDataDual_t =                                               \
      TemplateArrayBatchedData_t<autodiff::forward::Dual<dtype, dtype>>;
#else
#define SXS_DECLARE_DTYPE_DUALDATA(dtype) ;
#endif

// forward declaration
template <typename dtype> class MultivariateGaussianMixtureModelDiagCov;

/**
 * Multivariate Gaussian with row vector as `mu`, and square matrix as `sigma`
 */
template <typename dtype = double> class MultivariateGaussian {
public:
  SXS_DECLARE_DTYPE_DATA(dtype)
  SXS_DECLARE_DTYPE_DUALDATA(dtype)

  /**
   * Public interface
   *
   * @param mu_par - mean vector
   * @param sigma_par - square matrix as covariance
   */

  MultivariateGaussian(const Data_t &mu_par, const BatchedData_t &sigma_par);

  /**
   * @return current mean vector
   */
  const Data_t &get_mu() const;

  /**
   * @return current covariance matrix
   */
  const BatchedData_t &get_sigma() const;

  /**
   * @param mu - new mean vector
   */
  void set_mu(const Data_t &mu);

  /**
   * @param sigma - new covariance matrix
   */
  void set_sigma(const BatchedData_t &sigma);

  /**
   * Internal function to post-process sigma (e.g. inversing the matrix)
   */
  void _set_sigma_post_process();

  /**
   * Get log probability at a single point
   *
   * @param x - a vector that represent the data point. Must be same shape as
   *            mu
   * @return log probability of x
   */
  dtype log_prob_single(const Data_t &x) const;

  /**
   * Get the gradient of log probability at a single point
   *
   * @param x - a vector that represent the data point. Must be same shape as
   *            mu
   * @return a pair of gradient vector and the log probability of x
   */
  std::pair<Data_t, dtype> log_prob_single_grad(const Data_t &x) const;

  /**
   * Get the gradient of log probability at a batch of points
   *
   * @param X - a batch of data points in the shape [B,K] where B is the batch
   * @return a batch of log probability at X
   */
  template <typename eT>
  TemplateBatch_t<eT> log_prob(const TemplateBatchedData_t<eT> &X) const;

  /**
   * Get the gradient of log probability at a batch of points
   *
   * @param X - a batch of data points in the shape [B,K] where B is the batch
   * @return a pair of batch of gradients and a batch of log probability of X
   */
  BatchedData_t log_prob_grad(const BatchedData_t &X) const;

  /**
   * Get the gradient and log probability at a batch of points
   *
   * @param X - a batch of data points in the shape [B,K] where B is the batch
   * @return a pair of batch of gradients and a batch of log probability of X
   */
  std::pair<BatchedData_t, Batch_t>
  log_prob_and_grad(const BatchedData_t &X) const;

#ifdef SXS_GMM_USE_AUTODIFF
  // deprecated
  BatchedData_t NotWorking_log_prob_grad(const BatchedDataDual_t &X) const;
#endif

protected:
  /**
   * Internal implementation of single log prob
   *
   * @tparam eT - value type
   * @param inv - inversed matrix
   * @param X - a data point
   * @return log probability at x
   */
  template <typename eT>
  eT _internal_log_prob_single(const TemplateBatchedData_t<eT> &inv,
                               const TemplateData_t<eT> &X) const;

  /**
   * Internal implementation of log prob
   *
   * @tparam eT - value type
   * @param inv - inversed matrix
   * @param X - a data point
   * @return log probability at x
   */
  template <typename eT>
  inline TemplateBatch_t<eT>
  _internal_log_prob(const TemplateBatchedData_t<eT> &inv,
                     const TemplateBatchedData_t<eT> &X) const;

  // K is the number of events (i.e. K=1 for univariate distributions)
  const size_t K;
  Data_t m_mu;
  BatchedData_t m_sigma;
  dtype m_sigma_log_det;

  //  ColPivHouseholderQR<BatchedData_t> m_cached_sigma_piv;
  //  ColPivHouseholderQR<BatchedDataDual_t> m_cached_dual_sigma_piv;

  BatchedData_t m_cached_sigma_inv;
#ifdef SXS_GMM_USE_AUTODIFF
  BatchedDataDual_t m_cached_dual_sigma_inv;
#endif

  /**
   * Called by derived class
   *
   * @param k - the dimension of data point
   */
  MultivariateGaussian(size_t k) : K(k) {}

private:
  // allow the GMM to access private variables
  friend class MultivariateGaussianMixtureModelDiagCov<dtype>;

  /**
   * @return sigma log det for is derived class
   */
  virtual dtype _get_sigma_log_det() const;
};

/**
 * Multivariate Gaussian with Diagonal covariance. Row vector as `mu`, and
 * square matrix as `sigma`
 */
template <typename dtype = double>
class MultivariateGaussianDiagCov : public MultivariateGaussian<dtype> {
public:
  SXS_DECLARE_DTYPE_DATA(dtype)
  SXS_DECLARE_DTYPE_DUALDATA(dtype)

  /**
   * Implementation of a multivariate gaussian with diagonal covariance
   *
   * @param mu_par - mean vector
   * @param sigma_as_diagonal - covariance vector that represents the diagonal
   * values
   */
  MultivariateGaussianDiagCov(const Data_t &mu_par,
                              const Data_t &sigma_as_diagonal);

  /**
   * Sigma setter that only takes a row vector
   *
   * @param sigma - row vector that represents the diagonal values
   */
  void set_sigma(const Data_t &sigma);

  /**
   * Sigma getter that only takes a row vector
   *
   * @param sigma - row vector that represents the diagonal values
   */
  const Data_t get_sigma() const;

protected:
  /**
   * A diagonal sigma for this class
   */
  Data_t m_sigma_diag;

private:
  /**
   * @return sigma log det for is derived class
   */
  dtype _get_sigma_log_det() const override;
};

/**
 * Implementation of Gaussian Mixture Model, with diagonal covariance
 */
template <typename dtype = double>
class MultivariateGaussianMixtureModelDiagCov {
public:
  SXS_DECLARE_DTYPE_DATA(dtype)
  SXS_DECLARE_DTYPE_DUALDATA(dtype)

  /**
   * Public constructor
   *
   * @param mu - a batch of mean vectors [B,K]
   * @param sigma_diag - a batch of covariance diagonal vectors [B,K]
   * @param phi - a vector of weights [
   */
  MultivariateGaussianMixtureModelDiagCov(BatchedData_t mu,
                                          BatchedData_t sigma_diag,
                                          ArrayBatch_t phi);

  /**
   * Expectation step of the Expectation-Maximise procedure
   *
   * @param X - a batch of data points
   * @return A batch of heuristic values
   */
  BatchedData_t e_step(const BatchedData_t &X) const;

  /**
   * Expectation-Maximise step procedure for mu
   *
   * @param X - a batch of data points
   * @return A batch of values for each x \in X
   */
  BatchedData_t em_step_for_mu(const BatchedData_t &X) const;

  /**
   * Log probability of the GMM
   *
   * @param x - a data point
   * @return log probability of x
   */
  dtype log_prob_single(const Data_t &X) const;

  /**
   * Log probability of the GMM
   *
   * @param X - A batch of data points
   * @return log probability of x \in X
   */
  //  Data_t log_prob(const BatchedData_t &x) const;
  template <typename eT>
  TemplateData_t<eT> log_prob(const TemplateBatchedData_t<eT> &x) const;

  /**
   * Get the gradient of log probability at a batch of points
   *
   * @param X - a batch of data points in the shape [B,K] where B is the batch
   * @return a pair of batch of gradients and a batch of log probability of X
   */
  BatchedData_t log_prob_grad(const BatchedData_t &X) const;

  //  /**
  //   * Internal implementation of log prob
  //   *
  //   * @tparam eT - value type
  //   * @param inv - inversed matrix
  //   * @param X - data points
  //   * @return log probability at X
  //   */
  //  template <typename eT>
  //  inline TemplateBatch_t<eT>
  //  _internal_log_prob(const TemplateBatchedData_t<eT> &inv,
  //                     const TemplateBatchedData_t<eT> &X) const;

  /**
   * Getter of the internal multivariate gaussian
   *
   * @param i - index of the Gaussian
   * @return the Gaussian distribution
   */
  MultivariateGaussianDiagCov<dtype> &get_gaus(size_t i);

protected:
  const size_t K;
  ArrayBatch_t m_phi;
  ArrayBatch_t m_log_phi; // log of weights

  std::vector<MultivariateGaussianDiagCov<dtype>> m_gaus;
};

template class MultivariateGaussian<double>;
template class MultivariateGaussian<float>;
template class MultivariateGaussianDiagCov<double>;
template class MultivariateGaussianDiagCov<float>;
template class MultivariateGaussianMixtureModelDiagCov<double>;
template class MultivariateGaussianMixtureModelDiagCov<float>;

} // namespace gmm
