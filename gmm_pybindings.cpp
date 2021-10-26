#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// header file
#include "gmm.hpp"
// templated implementation
#include "gmm.tpp"

namespace py = pybind11;

using namespace gmm;

template <typename dtype> void _declare_py_class_helper(py::module &m) {
  /**
   * This helper method aid to declare the same set of class with both
   * Float and Double version.
   */

  using Data_t = TemplateData_t<dtype>;
  using BatchedData_t = TemplateBatchedData_t<dtype>;

  std::string suffix;
  if constexpr (std::is_same_v<dtype, double>)
    suffix = "";
  else
    suffix = "Float";

  /**
   * Definition for Multivariate Gaussian
   */
  py::class_<MultivariateGaussian<dtype>>(
      m, (std::string("MultivariateGaussian") + suffix).c_str())
      .def(py::init<const Data_t &, const BatchedData_t &>(), py::arg("mu"),
           py::arg("sigma"))
      .def_property("mu", &MultivariateGaussian<dtype>::get_mu,
                    &MultivariateGaussian<dtype>::set_mu,
                    py::return_value_policy::copy)
      .def_property("sigma", &MultivariateGaussian<dtype>::get_sigma,
                    &MultivariateGaussian<dtype>::set_sigma,
                    py::return_value_policy::copy)
      //      .def("log_prob_single",
      //           py::overload_cast<const Data_t &>(
      //               &MultivariateGaussian<dtype>::log_prob_single,
      //               py::const_))
      .def("log_prob",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussian<dtype>::template log_prob<dtype>,
               py::const_))
      //      .def("log_prob_single_grad",
      //           py::overload_cast<const Data_t &>(
      //               &MultivariateGaussian<dtype>::log_prob_single_grad,
      //               py::const_))
      .def("log_prob_grad",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussian<dtype>::log_prob_grad, py::const_))
      .def("log_prob_and_grad",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussian<dtype>::log_prob_and_grad, py::const_));

  /**
   * Definition for Multivariate Gaussian with Diagonal Covariance
   */
  py::class_<MultivariateGaussianDiagCov<dtype>>(
      m, (std::string("MultivariateGaussianDiagCov") + suffix).c_str())
      .def(py::init<const Data_t &, const Data_t &>(), py::arg("mu"),
           py::arg("sigma_diag"))
      .def_property("mu", &MultivariateGaussianDiagCov<dtype>::get_mu,
                    &MultivariateGaussianDiagCov<dtype>::set_mu,
                    py::return_value_policy::copy)
      .def_property("sigma", &MultivariateGaussianDiagCov<dtype>::get_sigma,
                    &MultivariateGaussianDiagCov<dtype>::set_sigma,
                    py::return_value_policy::copy)
      //      .def(
      //          "log_prob_single",
      //          py::overload_cast<const Data_t &>(
      //              &MultivariateGaussianDiagCov<dtype>::log_prob_single,
      //              py::const_))
      .def("log_prob",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussianDiagCov<dtype>::template log_prob<dtype>,
               py::const_))
      //      .def("log_prob_single_grad",
      //           py::overload_cast<const Data_t &>(
      //               &MultivariateGaussianDiagCov<dtype>::log_prob_single_grad,
      //               py::const_))
      .def("log_prob_grad",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussianDiagCov<dtype>::log_prob_grad, py::const_))
      .def("log_prob_and_grad",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussianDiagCov<dtype>::log_prob_and_grad,
               py::const_));

  /**
   * Definition for Multivariate Gaussian Mixture Model
   */
  py::class_<MultivariateGaussianMixtureModelDiagCov<dtype>>(
      m,
      (std::string("MultivariateGaussianMixtureModelDiagCov") + suffix).c_str())
      .def(py::init<Matrix<dtype, -1, -1>, Matrix<dtype, -1, -1>,
                    Matrix<dtype, -1, 1>>(),
           py::arg("mu"), py::arg("sigma_diag"), py::arg("phi"))
      .def("e_step",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussianMixtureModelDiagCov<dtype>::e_step,
               py::const_))
      .def("em_step_for_mu",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussianMixtureModelDiagCov<dtype>::em_step_for_mu,
               py::const_))
      //      .def("log_prob_single",
      //           py::overload_cast<const Data_t &>(
      //               &MultivariateGaussianMixtureModelDiagCov<dtype>::log_prob_single,
      //               py::const_))
      .def("log_prob", py::overload_cast<const BatchedData_t &>(
                           &MultivariateGaussianMixtureModelDiagCov<
                               dtype>::template log_prob<dtype>,
                           py::const_))
      .def("log_prob_grad",
           py::overload_cast<const BatchedData_t &>(
               &MultivariateGaussianMixtureModelDiagCov<dtype>::log_prob_grad,
               py::const_))
      .def("get_gaus",
           py::overload_cast<size_t>(
               &MultivariateGaussianMixtureModelDiagCov<dtype>::get_gaus));
}

void init_gmm_diff_module(py::module &m) {

  _declare_py_class_helper<double>(m);

  /*
   * The following is the float version of the above class. Directly
   * mirroring by copying the above.
   * */
  _declare_py_class_helper<float>(m);
}

PYBIND11_MODULE(fast_gmm_diff, m) {
  // Optional docstring
  m.doc() = "Fast implementation of Multivariate Gaussian and GMM model with "
            "gradient supports in C++";

  init_gmm_diff_module(m);
}
