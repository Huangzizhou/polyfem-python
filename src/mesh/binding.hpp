#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void define_mesh(py::module_ &m);
