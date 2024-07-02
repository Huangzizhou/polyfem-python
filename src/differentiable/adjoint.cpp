#include <polyfem/solver/AdjointTools.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/State.hpp>
#include "binding.hpp"
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace polyfem;
using namespace polyfem::solver;

void define_adjoint(py::module_ &m)
{
  m.def("shape_derivative", [](State &state) {
    Eigen::VectorXd term;
    if (state.problem->is_time_dependent())
      AdjointTools::dJ_shape_transient_adjoint_term(
          state, state.get_adjoint_mat(1), state.get_adjoint_mat(0), term);
    else
      AdjointTools::dJ_shape_static_adjoint_term(
          state, state.diff_cached.u(0), state.get_adjoint_mat(0), term);
    return utils::unflatten(term, state.mesh->dimension());
  }, py::arg("solver"));
}
