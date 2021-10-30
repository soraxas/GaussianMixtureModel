
// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

#ifdef SXS_GMM_USE_AUTODIFF
// autodiff include
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
using namespace autodiff;
#endif

#include <Eigen/Dense>

#include "cppm.hpp"
#include "soraxas_toolbox/globals.h"
#include "soraxas_toolbox/main.h"

#include "gmm.hpp"
#include "gmm.tpp"

#include <cmath>

#include <iostream>

using namespace Eigen;
using namespace std;

#include <typeinfo>

#ifdef SXS_GMM_USE_AUTODIFF
// The scalar function for which the gradient is needed
dual f(const VectorXdual &x, dual p) {
  return x.cwiseProduct(x).sum() *
         exp(p); // sum([x(i) * x(i) for i = 1:5]) * exp(p)
}

int test_gaus_single_log_prob() {

  using dtype = double;
  using Data_t = gmm::MultivariateGaussianDiagCov<dtype>::Data_t;
  using DataDual_t = gmm::MultivariateGaussianDiagCov<dtype>::DataDual_t;

  sxs::println("hi");

  Data_t mu(2);
  Data_t sigma_as_diagonal(2);

  mu << 2, 4;
  sigma_as_diagonal << 1.42, 91;

  gmm::MultivariateGaussianDiagCov<dtype> gauss(mu, sigma_as_diagonal);

  DataDual_t pt(2);
  pt << 1.5, 0.3;
  Data_t pst(2);
  pst << 1.5, 0.3;

  sxs::println("=============");
  sxs::println(gauss.log_prob_single(pst));
  sxs::println("=============");
  //  sxs::println(gauss.log_prob_single_grad(pt));

  sxs::println("========= log prob ====");

  //  sxs::println(gauss.log_prob_single_grad(pt));
  //  dual uu;
  //  VectorXd vvgpx = gradient(
  //      [gauss](auto &&a) { return gauss.log_prob_single(a); }, wrt(pt),
  //      at(pt), uu); // evaluate the function value u and its gradient
  //           // vector gp = [du/dp, du/dx]
  //
  //  sxs::println(vvgpx);
  //  sxs::println(uu);
  sxs::println("================   uu  ==============");

  //  VectorXd gpx = gradient(f, wrt(x), at(x, p),
  //                          u); // evaluate the function value u and its
  //                          gradient
  //                              // vector gp = [du/dp, du/dx]

  Data_t v(2);
  v << 2, 3;

  sxs::println(gauss.log_prob_single(v));

  ////////////////////////////////////
  ////////////////////////////////////
  ////////////////////////////////////

  VectorXdual x(5);   // the input vector x with 5 variables
  x << 1, 2, 3, 4, 5; // x = [1, 2, 3, 4, 5]

  // the input parameter vector p with 3 variables
  dual p = 3;

  // the output scalar u = f(x, p) evaluated together with gradient below
  dual u;

  VectorXd gpx = gradient(f, wrt(x), at(x, p),
                          u); // evaluate the function value u and its gradient
                              // vector gp = [du/dp, du/dx]

  sxs::println(u);

  cout << "u = " << u << endl; // print the evaluated output u
  cout << "gpx = \n"
       << gpx
       << endl; // print the evaluated gradient vector gp = [du/dp, du/dx]
}
#endif

template <typename T> T f(T arg) {
  //  std::cout << "======" << arg.transpose() << "========" << std::endl;
  //  std::cout << "======" << arg.transpose() << "========" << std::endl;
  //  std::cout << "======" << arg.transpose() << "========" << std::endl;
  Eigen::VectorXd vv(2);
  vv << 4, 5;

  T aa = arg.array() * vv.array();
  //  aa(0) = 1;
  //  aa(1) = 3;

  std::cout << "======" << aa.transpose() << "========" << std::endl;

  aa *= 3;
  std::cout << "======" << aa.transpose() << "========" << std::endl;
  aa(0) += 0.1 * aa(1);
  std::cout << "======" << aa.transpose() << "========" << std::endl;

  aa(0) += aa(1);
  std::cout << "======" << aa.transpose() << "========" << std::endl;
  std::cout << "==============" << std::endl;
  std::cout << "======" << aa.transpose() << "========" << std::endl;
  aa(1) *= aa(0);
  aa(1) += aa(0);
  aa(1) *= aa(0);
  std::cout << "======" << aa.transpose() << "========" << std::endl;
  std::cout << "======" << aa(1).grad.adjoints() << "========" << std::endl;
  //
  //  aa(0) -= aa(1) * .1;
  //  //  aa(0) *= aa(1);
  //
  //  aa(0) += 1;
  //  aa(0) *= 1.1;

  //  aa(0) /= aa(1);
  //  aa(0) = -aa(1);
  //  aa(0) = -aa(0);
  //  aa(0) = -aa(0);
  return aa;
  //  return arg * 2;
}

int main() {

  using gmm_t = gmm::MultivariateGaussianDiagCov<double>;

  using Data_t = gmm_t::Data_t;
  using BatchedData_t = gmm_t::BatchedData_t;

  size_t K = 4;

  Data_t mu = Data_t::Random(K);
  Data_t sigma_as_diagonal = Data_t::Random(K);

  gmm_t gauss(mu, sigma_as_diagonal);

  size_t N = 100;
  for (auto &&i : cppm::range(200)) {
    BatchedData_t pst = BatchedData_t::Random(N, K);
    gauss.log_prob(pst);
  }

  for (auto &&i : cppm::range(200)) {
    BatchedData_t pts = BatchedData_t::Random(N, K);
    //    gmm::BatchedDataDual_t pts = gmm::BatchedDataDual_t::Random(N, K);
    //    dual uu;
    //    volatile VectorXd vvgpx = gradient(
    //        [gauss](auto &&a) { return gauss.log_prob_single_grad(a); },
    //        wrt(pts.row(i)), at(pts.row(i)), uu);

    gauss.log_prob(pts);
  }

  BatchedData_t dual_pts = BatchedData_t::Random(N, K);

  // loop through each pt
  std::cout << dual_pts.rows() << std::endl;

  const int num_variables = 1; // in our cas

  BatchedData_t g(dual_pts.rows(), dual_pts.cols());

  auto ret = gauss.log_prob(dual_pts);
  sxs::println(ret);

  std::cout << "...." << std::endl;
  std::cout << gauss.log_prob_and_grad(dual_pts).first << std::endl;
  sxs::println("================   ____uu  ==============");
  std::cout << gauss.log_prob_and_grad(dual_pts).second << std::endl;

  //          for(auto j = 0; j < dual_pts.rows(); ++j)
  //        {
  //            dual_pts.row().grad = 1.0;
  //            u = std::apply(f, args);
  //            w[j].grad = 0.0;
  //            g[j + current_index_pos] = val(u.grad);
  //        }

  //  mu << 2, 4;
  //  sigma_as_diagonal << 1.42, 91;
  //
  //  gmm::MultivariateGaussianDiagCov gauss(mu, sigma_as_diagonal);
  //
  //  BatchedData_t pst(4, 2);
  //  pst << 1.5, 0.3, 5, 3, 6, 7, 8, 1;
  //  gmm::BatchedDataDual_t pt(4, 2);
  //  pt << 1.5, 0.3, 5, 3, 6, 7, 8, 1;
  //
  //  std::cout << pst << std::endl;
  //  sxs::println("--------------");
  //  std::cout << pt << std::endl;
  //  sxs::println("--------------");
  //
  //  sxs::println("=============");
  //  sxs::println(gauss.log_prob(pst));
  //  sxs::println("=============");
  ////  sxs::println(gauss.log_prob(pt));
  //  sxs::println(gauss.log_prob_grad(pst));

  sxs::println("========= log prob ====");

  //  dual uu;
  //  VectorXd vvgpx = gradient(
  //      [gauss](auto &&a) { return gauss.log_prob_single(a); }, wrt(pt),
  //      at(pt), uu); // evaluate the function value u and its gradient
  //           // vector gp = [du/dp, du/dx]

  //  sxs::println(vvgpx);
  //  sxs::println(uu);
  sxs::println("================   uu  ==============");

  //  sxs::println(gauss.log_prob(pt));
  return 0;
}
