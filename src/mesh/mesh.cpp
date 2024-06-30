#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include "binding.hpp"
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace polyfem;
using namespace polyfem::mesh;

void define_mesh(py::module_ &m)
{
  py::class_<Mesh>(m, "Mesh", "Mesh")

      .def("n_elements", &Mesh::n_elements, "Number of elements")

      .def("n_boundary_elements", &Mesh::n_boundary_elements,
           "Number of boundary elements (faces in 3D, edges in 2D)")

      .def("n_vertices", &Mesh::n_vertices, "Number of vertices")

      .def("n_cell_vertices", &Mesh::n_cell_vertices,
           "Number of vertices in one cell", py::arg("cell_id"))

      .def("element_vertex", &Mesh::element_vertex,
           "Global vertex ID of a local vertex in an element",
           py::arg("cell_id"), py::arg("local_vertex_id"))

      .def("boundary_element_vertex", &Mesh::boundary_element_vertex,
           "Global vertex ID of a local vertex in a boundary element",
           py::arg("boundary_cell_id"), py::arg("local_vertex_id"))

      .def("is_boundary_vertex", &Mesh::is_boundary_vertex,
           "Check if a vertex is on boundary", py::arg("vertex_id"))

      .def(
          "bounding_box",
          [](const Mesh &mesh) {
            RowVectorNd min, max;
            mesh.bounding_box(min, max);
            return py::make_tuple(min, max);
          },
          "Get bounding box")

      .def("set_boundary_ids", &Mesh::set_boundary_ids,
           "Set boundary IDs with an array", py::arg("ids"))

      .def("set_body_ids", &Mesh::set_body_ids, "Set body IDs with an array",
           py::arg("ids"));

  py::class_<CMesh2D, Mesh>(m, "Mesh2D", "");
  py::class_<CMesh3D, Mesh>(m, "Mesh3D", "");
}
