#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/YamlToJson.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/GeogramUtils.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/State.hpp>

// #include "raster.hpp"

#include <igl/boundary_facets.h>
#include <igl/remove_unreferenced.h>

#include <stdexcept>

#include <pybind11_json/pybind11_json.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "differentiable/binding.hpp"

namespace py = pybind11;
using namespace polyfem;

typedef std::function<Eigen::MatrixXd(double x, double y, double z)> BCFuncV;
typedef std::function<double(double x, double y, double z)> BCFuncS;

class Assemblers
{
};

class PDEs
{
};

// TODO add save_time_sequence

namespace
{

  bool load_json(const std::string &json_file, json &out)
  {
    std::ifstream file(json_file);

    if (!file.is_open())
      return false;

    file >> out;

    if (!out.contains("root_path"))
      out["root_path"] = json_file;

    return true;
  }

  bool load_yaml(const std::string &yaml_file, json &out)
  {
    try
    {
      out = io::yaml_file_to_json(yaml_file);
      if (!out.contains("root_path"))
        out["root_path"] = yaml_file;
    }
    catch (...)
    {
      return false;
    }
    return true;
  }

  void init_globals(State &state)
  {
    static bool initialized = false;

    if (!initialized)
    {
      state.set_max_threads(1);
      state.init_logger("", spdlog::level::level_enum::info,
                        spdlog::level::level_enum::debug, false);

      initialized = true;
    }
  }

} // namespace

// const auto rendering_lambda =
//     [](const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &faces, int
//     width,
//        int height, const Eigen::MatrixXd &camera_positionm,
//        const double camera_fov, const double camera_near,
//        const double camera_far, const bool is_perspective,
//        const Eigen::MatrixXd &lookatm, const Eigen::MatrixXd &upm,
//        const Eigen::MatrixXd &ambient_lightm,
//        const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>
//        &lights, const Eigen::MatrixXd &diffuse_colorm, const Eigen::MatrixXd
//        &specular_colorm, const double specular_exponent) {
//       using namespace renderer;
//       using namespace Eigen;

//       Eigen::Vector3d camera_position(0, 0, 1);
//       Eigen::Vector3d lookat(0, 0, 0);
//       Eigen::Vector3d up(0, 0, 1);
//       Eigen::Vector3d ambient_light(0.1, 0.1, 0.1);
//       Eigen::Vector3d diffuse_color(1, 0, 0);
//       Eigen::Vector3d specular_color(1, 0, 0);

//       if (camera_positionm.size() > 0 && camera_positionm.size() != 3)
//         throw pybind11::value_error("camera_position have size 3");
//       if (camera_positionm.size() == 3)
//         camera_position << camera_positionm(0), camera_positionm(1),
//             camera_positionm(2);
//       if (lookatm.size() > 0 && lookatm.size() != 3)
//         throw pybind11::value_error("lookat have size 3");
//       if (lookatm.size() == 3)
//         lookat << lookatm(0), lookatm(1), lookatm(2);
//       if (upm.size() > 0 && upm.size() != 3)
//         throw pybind11::value_error("up have size 3");
//       if (upm.size() == 3)
//         up << upm(0), upm(1), upm(2);
//       if (ambient_lightm.size() > 0 && ambient_lightm.size() != 3)
//         throw pybind11::value_error("ambient_light have size 3");
//       if (ambient_lightm.size() == 3)
//         ambient_light << ambient_lightm(0), ambient_lightm(1),
//             ambient_lightm(2);
//       if (diffuse_colorm.size() > 0 && diffuse_colorm.size() != 3)
//         throw pybind11::value_error("diffuse_color have size 3");
//       if (diffuse_colorm.size() == 3)
//         diffuse_color << diffuse_colorm(0), diffuse_colorm(1),
//             diffuse_colorm(2);
//       if (specular_colorm.size() > 0 && specular_colorm.size() != 3)
//         throw pybind11::value_error("specular_color have size 3");
//       if (specular_colorm.size() == 3)
//         specular_color << specular_colorm(0), specular_colorm(1),
//             specular_colorm(2);

//       Material material;
//       material.diffuse_color = diffuse_color;
//       material.specular_color = specular_color;
//       material.specular_exponent = specular_exponent;

//       Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>
//           frameBuffer(width, height);
//       UniformAttributes uniform;

//       const Vector3d gaze = lookat - camera_position;
//       const Vector3d w = -gaze.normalized();
//       const Vector3d u = up.cross(w).normalized();
//       const Vector3d v = w.cross(u);

//       Matrix4d M_cam_inv;
//       M_cam_inv << u(0), v(0), w(0), camera_position(0), u(1), v(1), w(1),
//           camera_position(1), u(2), v(2), w(2), camera_position(2), 0, 0, 0,
//           1;

//       uniform.M_cam = M_cam_inv.inverse();

//       {
//         const double camera_ar = double(width) / height;
//         const double tan_angle = tan(camera_fov / 2);
//         const double n = -camera_near;
//         const double f = -camera_far;
//         const double t = tan_angle * n;
//         const double b = -t;
//         const double r = t * camera_ar;
//         const double l = -r;

//         uniform.M_orth << 2 / (r - l), 0, 0, -(r + l) / (r - l), 0, 2 / (t -
//         b),
//             0, -(t + b) / (t - b), 0, 0, 2 / (n - f), -(n + f) / (n - f), 0,
//             0, 0, 1;
//         Matrix4d P;
//         if (is_perspective)
//         {
//           P << n, 0, 0, 0, 0, n, 0, 0, 0, 0, n + f, -f * n, 0, 0, 1, 0;
//         }
//         else
//         {
//           P << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
//         }

//         uniform.M = uniform.M_orth * P * uniform.M_cam;
//       }

//       Program program;
//       program.VertexShader = [&](const VertexAttributes &va,
//                                  const UniformAttributes &uniform) {
//         VertexAttributes out;
//         out.position = uniform.M * va.position;
//         Vector3d color = ambient_light;

//         Vector3d hit(va.position(0), va.position(1), va.position(2));
//         for (const auto &l : lights)
//         {
//           Vector3d Li = (l.first - hit).normalized();
//           Vector3d N = va.normal;
//           Vector3d diffuse =
//               va.material.diffuse_color * std::max(Li.dot(N), 0.0);
//           Vector3d H;
//           if (is_perspective)
//           {
//             H = (Li + (camera_position - hit).normalized()).normalized();
//           }
//           else
//           {
//             H = (Li - gaze.normalized()).normalized();
//           }
//           const Vector3d specular = va.material.specular_color
//                                     * std::pow(std::max(N.dot(H), 0.0),
//                                                va.material.specular_exponent);
//           const Vector3d D = l.first - hit;
//           color +=
//               (diffuse + specular).cwiseProduct(l.second) / D.squaredNorm();
//         }
//         out.color = color;
//         return out;
//       };

//       program.FragmentShader = [](const VertexAttributes &va,
//                                   const UniformAttributes &) {
//         FragmentAttributes out(va.color(0), va.color(1), va.color(2));
//         out.depth = -va.position(2);
//         return out;
//       };

//       program.BlendingShader = [](const FragmentAttributes &fa,
//                                   const FrameBufferAttributes &previous) {
//         if (fa.depth < previous.depth)
//         {
//           FrameBufferAttributes out(fa.color[0] * 255, fa.color[1] * 255,
//                                     fa.color[2] * 255, fa.color[3] * 255);
//           out.depth = fa.depth;
//           return out;
//         }
//         else
//         {
//           return previous;
//         }
//       };

//       Eigen::MatrixXd vnormals(vertices.rows(), 3);
//       //    Eigen::MatrixXd areas(tmp.rows(), 1);
//       vnormals.setZero();
//       //    areas.setZero();
//       Eigen::MatrixXd fnormals(faces.rows(), 3);

//       for (int i = 0; i < faces.rows(); ++i)
//       {
//         const Vector3d l1 =
//             vertices.row(faces(i, 1)) - vertices.row(faces(i, 0));
//         const Vector3d l2 =
//             vertices.row(faces(i, 2)) - vertices.row(faces(i, 0));
//         const auto nn = l1.cross(l2);
//         const double area = nn.norm();
//         fnormals.row(i) = nn / area;

//         for (int j = 0; j < 3; j++)
//         {
//           int vid = faces(i, j);
//           vnormals.row(vid) += nn;
//           //    areas(vid) += area;
//         }
//       }

//       std::vector<VertexAttributes> vertex_attributes;
//       for (int i = 0; i < faces.rows(); ++i)
//       {
//         for (int j = 0; j < 3; j++)
//         {
//           int vid = faces(i, j);
//           VertexAttributes va(vertices(vid, 0), vertices(vid, 1),
//                               vertices(vid, 2));
//           va.material = material;
//           va.normal = vnormals.row(vid).normalized();
//           vertex_attributes.push_back(va);
//         }
//       }

//       rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);

//       std::vector<uint8_t> image;
//       framebuffer_to_uint8(frameBuffer, image);

//       return image;
//     };

void define_pde_types(py::module_ &m)
{
  const auto &pdes = py::class_<PDEs>(m, "PDEs");

  const std::vector<std::string> materials = {"LinearElasticity",
                                              "HookeLinearElasticity",
                                              "SaintVenant",
                                              "NeoHookean",
                                              "MooneyRivlin",
                                              "MooneyRivlin3Param",
                                              "MooneyRivlin3ParamSymbolic",
                                              "UnconstrainedOgden",
                                              "IncompressibleOgden",
                                              "Stokes",
                                              "NavierStokes",
                                              "OperatorSplitting",
                                              "IncompressibleLinearElasticity",
                                              "Laplacian",
                                              "Helmholtz",
                                              "Bilaplacian",
                                              "AMIPS",
                                              "FixedCorotational"};

  for (const auto &a : materials)
    pdes.attr(a.c_str()) = a;

  pdes.doc() = "List of supported partial differential equations";

  m.def(
      "is_tensor",
      [](const std::string &pde) {
        if (pde == "Laplacian" || pde == "Helmholtz" || pde == "Bilaplacian")
          return false;
        return true;
      },
      "returns true if the pde is tensorial", py::arg("pde"));
}

void define_solver(py::module_ &m)
{
  const auto setting_lambda = [](State &self, const py::object &settings,
                                 bool strict_validation) {
    using namespace polyfem;

    init_globals(self);
    // py::scoped_ostream_redirect output;
    const std::string json_string = py::str(settings);
    self.init(json::parse(json_string), strict_validation);
  };

  py::class_<State>(m, "Solver")
      .def(py::init<>())

      .def("is_tensor", [](const State &s) { return s.assembler->is_tensor(); })

      .def(
          "settings", [](const State &s) { return s.args; },
          "get PDE and problem parameters from the solver")

      .def("set_settings", setting_lambda,
           "load PDE and problem parameters from the settings", py::arg("json"),
           py::arg("strict_validation") = false)

      .def("ndof", &State::ndof, "Dimension of the solution")

      .def(
          "set_log_level",
          [](State &s, int log_level) {
            init_globals(s);
            //    py::scoped_ostream_redirect output;
            log_level = std::max(0, std::min(6, log_level));
            s.set_log_level(static_cast<spdlog::level::level_enum>(log_level));
          },
          "sets polyfem log level, valid value between 0 (all logs) and 6 (no logs)",
          py::arg("log_level"))

      .def(
          "mesh", [](State &s) -> mesh::Mesh& { return *s.mesh.get(); },
          "Get mesh in simulator", py::return_value_policy::reference)

      .def(
          "load_mesh_from_settings",
          [](State &s, const bool normalize_mesh, const double vismesh_rel_area,
             const int n_refs, const double boundary_id_threshold) {
            init_globals(s);
            s.args["geometry"][0]["advanced"]["normalize_mesh"] =
                normalize_mesh;
            s.args["geometry"][0]["surface_selection"] =
                R"({ "threshold": 0.0, "id_offset": 0 })"_json;
            s.args["geometry"][0]["surface_selection"]["threshold"] =
                boundary_id_threshold;

            s.args["output"]["paraview"]["vismesh_rel_area"] = vismesh_rel_area;
            s.args["geometry"][0]["n_refs"] = n_refs;
            s.load_mesh();
          },
          "Loads a mesh from the 'mesh' field of the json and 'bc_tag' if any bc tags",
          py::arg("normalize_mesh") = bool(false),
          py::arg("vismesh_rel_area") = double(0.00001),
          py::arg("n_refs") = int(0),
          py::arg("boundary_id_threshold") = double(-1))

      .def(
          "load_mesh_from_path",
          [](State &s, const std::string &path, const bool normalize_mesh,
             const double vismesh_rel_area, const int n_refs,
             const double boundary_id_threshold) {
            init_globals(s);
            s.args["geometry"] = R"([{ }])"_json;
            s.args["geometry"][0]["mesh"] = path;
            s.args["geometry"][0]["advanced"]["normalize_mesh"] =
                normalize_mesh;
            s.args["geometry"][0]["surface_selection"] =
                R"({ "threshold": 0.0 })"_json;
            s.args["geometry"][0]["surface_selection"]["threshold"] =
                boundary_id_threshold;
            s.args["geometry"][0]["n_refs"] = n_refs;
            s.args["output"]["paraview"]["vismesh_rel_area"] = vismesh_rel_area;
            s.load_mesh();
          },
          "Loads a mesh from the path and 'bc_tag' from the json if any bc tags",
          py::arg("path"), py::arg("normalize_mesh") = bool(false),
          py::arg("vismesh_rel_area") = double(0.00001),
          py::arg("n_refs") = int(0),
          py::arg("boundary_id_threshold") = double(-1))

      .def(
          "load_mesh_from_path_and_tags",
          [](State &s, const std::string &path, const std::string &bc_tag,
             const bool normalize_mesh, const double vismesh_rel_area,
             const int n_refs, const double boundary_id_threshold) {
            init_globals(s);
            s.args["geometry"] = R"([{ }])"_json;
            s.args["geometry"][0]["mesh"] = path;
            s.args["bc_tag"] = bc_tag;
            s.args["geometry"][0]["advanced"]["normalize_mesh"] =
                normalize_mesh;
            s.args["geometry"][0]["surface_selection"] =
                R"({ "threshold": 0.0 })"_json;
            s.args["geometry"][0]["surface_selection"]["threshold"] =
                boundary_id_threshold;
            s.args["geometry"][0]["n_refs"] = n_refs;
            s.args["output"]["paraview"]["vismesh_rel_area"] = vismesh_rel_area;
            s.load_mesh();
          },
          "Loads a mesh and bc_tags from path", py::arg("path"),
          py::arg("bc_tag_path"), py::arg("normalize_mesh") = bool(false),
          py::arg("vismesh_rel_area") = double(0.00001),
          py::arg("n_refs") = int(0),
          py::arg("boundary_id_threshold") = double(-1))

      .def(
          "set_mesh",
          [](State &s, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
             const int n_refs, const double boundary_id_threshold) {
            init_globals(s);
            s.mesh = mesh::Mesh::create(V, F);
            s.args["geometry"] = R"([{ }])"_json;
            s.args["geometry"][0]["n_refs"] = n_refs;
            s.args["geometry"][0]["surface_selection"] =
                R"({ "threshold": 0.0 })"_json;
            s.args["geometry"][0]["surface_selection"]["threshold"] =
                boundary_id_threshold;

            s.load_mesh();
          },
          "Loads a mesh from vertices and connectivity", py::arg("vertices"),
          py::arg("connectivity"), py::arg("n_refs") = int(0),
          py::arg("boundary_id_threshold") = double(-1))

      .def(
          "set_high_order_mesh",
          [](State &s, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
             const Eigen::MatrixXd &nodes_pos,
             const std::vector<std::vector<int>> &nodes_indices,
             const bool normalize_mesh, const double vismesh_rel_area,
             const int n_refs, const double boundary_id_threshold) {
            init_globals(s);
            //    py::scoped_ostream_redirect output;

            s.mesh = mesh::Mesh::create(V, F);
            s.mesh->attach_higher_order_nodes(nodes_pos, nodes_indices);

            s.args["geometry"][0]["advanced"]["normalize_mesh"] =
                normalize_mesh;
            s.args["geometry"][0]["n_refs"] = n_refs;
            s.args["geometry"][0]["surface_selection"] =
                R"({ "threshold": 0.0 })"_json;
            s.args["geometry"][0]["surface_selection"]["threshold"] =
                boundary_id_threshold;
            s.args["output"]["paraview"]["vismesh_rel_area"] = vismesh_rel_area;

            s.load_mesh();
          },
          "Loads an high order mesh from vertices, connectivity, nodes, and node indices mapping element to nodes",
          py::arg("vertices"), py::arg("connectivity"), py::arg("nodes_pos"),
          py::arg("nodes_indices"), py::arg("normalize_mesh") = bool(false),
          py::arg("vismesh_rel_area") = double(0.00001),
          py::arg("n_refs") = int(0),
          py::arg("boundary_id_threshold") = double(-1))

      .def(
          "set_boundary_side_set_from_bary",
          [](State &s,
             const std::function<int(const RowVectorNd &)> &boundary_marker) {
            init_globals(s);
            //    py::scoped_ostream_redirect output;
            s.mesh->compute_boundary_ids(boundary_marker);
          },
          "Sets the side set for the boundary conditions, the functions takes the barycenter of the boundary (edge or face)",
          py::arg("boundary_marker"))
      .def(
          "set_boundary_side_set_from_bary_and_boundary",
          [](State &s, const std::function<int(const RowVectorNd &, bool)>
                           &boundary_marker) {
            init_globals(s);
            //    py::scoped_ostream_redirect output;
            s.mesh->compute_boundary_ids(boundary_marker);
          },
          "Sets the side set for the boundary conditions, the functions takes the barycenter of the boundary (edge or face) and a flag that says if the element is boundary",
          py::arg("boundary_marker"))
      .def(
          "set_boundary_side_set_from_v_ids",
          [](State &s, const std::function<int(const std::vector<int> &, bool)>
                           &boundary_marker) {
            init_globals(s);
            //    py::scoped_ostream_redirect output;
            s.mesh->compute_boundary_ids(boundary_marker);
          },
          "Sets the side set for the boundary conditions, the functions takes the sorted list of vertex id and a flag that says if the element is boundary",
          py::arg("boundary_marker"))

      .def("nl_problem", [](State &s) { return s.solve_data.nl_problem; })

      .def(
          "solve",
          [](State &s) {
            init_globals(s);
            //    py::scoped_ostream_redirect output;
            s.stats.compute_mesh_stats(*s.mesh);

            s.build_basis();

            s.assemble_rhs();
            s.assemble_mass_mat();

            Eigen::MatrixXd sol, pressure;
            s.solve_problem(sol, pressure);

            s.compute_errors(sol);

            s.save_json(sol);
            s.export_data(sol, pressure);

            return py::make_tuple(sol, pressure);
          },
          "solve the pde")
      .def(
          "build_basis",
          [](State &s) {
            if (!s.mesh)
              throw std::runtime_error("Load mesh first!");

            s.build_basis();
          },
          "build finite element basis")
      .def(
          "assemble",
          [](State &s) {
            if (s.bases.size() == 0)
              throw std::runtime_error("Call build_basis() first!");

            s.assemble_rhs();
            s.assemble_mass_mat();
          },
          "assemble RHS and mass matrix if needed")
      .def(
          "init_timestepping",
          [](State &s, const double t0, const double dt) {
            if (!s.solve_data.rhs_assembler || s.mass.size() == 0)
              throw std::runtime_error("Call assemble() first!");

            s.solution_frames.clear();
            Eigen::MatrixXd sol, pressure;
            s.init_solve(sol, pressure);
            s.init_nonlinear_tensor_solve(sol, t0 + dt);
            s.cache_transient_adjoint_quantities(
                0, sol,
                Eigen::MatrixXd::Zero(s.mesh->dimension(),
                                      s.mesh->dimension()));
            return sol;
          },
          "initialize timestepping", py::arg("t0"), py::arg("dt"))
      .def(
          "step_in_time",
          [](State &s, Eigen::MatrixXd &sol, const double t0, const double dt,
             const int t) {
            if (s.assembler->name() == "NavierStokes"
                || s.assembler->name() == "OperatorSplitting"
                || s.is_problem_linear() || s.is_homogenization())
              throw std::runtime_error("Formulation " + s.assembler->name()
                                       + " is not supported!");

            s.solve_tensor_nonlinear(sol, t);
            s.cache_transient_adjoint_quantities(
                t, sol,
                Eigen::MatrixXd::Zero(s.mesh->dimension(),
                                      s.mesh->dimension()));

            s.solve_data.time_integrator->update_quantities(sol);
            s.solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);
            s.solve_data.update_dt();
            s.solve_data.update_barrier_stiffness(sol);
            return sol;
          },
          "step in time", py::arg("solution"), py::arg("t0"), py::arg("dt"),
          py::arg("t"))

      .def(
          "solve_adjoint",
          [](State &s, const Eigen::MatrixXd &adjoint_rhs) {
            if (adjoint_rhs.cols() != s.diff_cached.size()
                || adjoint_rhs.rows() != s.diff_cached.u(0).size())
              throw std::runtime_error("Invalid adjoint_rhs shape!");
            if (!s.problem->is_time_dependent()
                && !s.lin_solver_cached) // nonlinear static solve only
            {
              Eigen::MatrixXd reduced;
              for (int i = 0; i < adjoint_rhs.cols(); i++)
              {
                Eigen::VectorXd reduced_vec =
                    s.solve_data.nl_problem->full_to_reduced_grad(
                        adjoint_rhs.col(i));
                if (i == 0)
                  reduced.setZero(reduced_vec.rows(), adjoint_rhs.cols());
                reduced.col(i) = reduced_vec;
              }
              return s.solve_adjoint_cached(reduced);
            }
            else
              return s.solve_adjoint_cached(adjoint_rhs);
          },
          "Solve the adjoint equation given the gradient of objective wrt. PDE solution")

      .def(
          "set_cache_level",
          [](State &s, solver::CacheLevel level) {
            s.optimization_enabled = level;
          },
          "Set solution caching level", py::arg("cache_level"))

      .def(
          "get_solution_cache", [](State &s) { return s.diff_cached; },
          "get the cached solution after simulation, this function requires setting CacheLevel before the simulation")

      .def(
          "compute_errors",
          [](State &s, Eigen::MatrixXd &sol) { s.compute_errors(sol); },
          "compute the error", py::arg("solution"))

      .def(
          "export_data",
          [](State &s, const Eigen::MatrixXd &sol,
             const Eigen::MatrixXd &pressure) { s.export_data(sol, pressure); },
          "exports all data specified in the settings")
      .def(
          "export_vtu",
          [](State &s, const Eigen::MatrixXd &sol,
             const Eigen::MatrixXd &pressure, const double time,
             const double dt, std::string &path, bool boundary_only) {
            s.args["output"]["advanced"]["vis_boundary_only"] = boundary_only;
            s.out_geom.save_vtu(
                s.resolve_output_path(path), s, sol, pressure, time, dt,
                io::OutGeometryData::ExportOptions(s.args, s.mesh->is_linear(),
                                                   s.problem->is_scalar(),
                                                   s.solve_export_to_file),
                s.is_contact_enabled(), s.solution_frames);
          },
          "exports the solution as vtu", py::arg("solution"),
          py::arg("pressure"), py::arg("time"), py::arg("dt"), py::arg("path"),
          py::arg("boundary_only") = bool(false))
      //    .def("export_wire", [](State &s, std::string &path, bool
      //    isolines) { py::scoped_ostream_redirect output; s.save_wire(path,
      //    isolines); }, "exports wireframe of the mesh", py::arg("path"),
      //    py::arg("isolines") = false)

      //    .def(
      // 	   "get_sampled_solution", [](State &s, const
      // Eigen::MatrixXd &sol, bool boundary_only) {
      // 		   //    py::scoped_ostream_redirect output;
      // 		   Eigen::MatrixXd points;
      // 		   Eigen::MatrixXi tets;
      // 		   Eigen::MatrixXi el_id;
      // 		   Eigen::MatrixXd discr;
      // 		   Eigen::MatrixXd fun;

      // 		   s.out_geom.build_vis_mesh(*s.mesh, s.disc_orders,
      // s.geom_bases(), s.polys, s.polys_3d, boundary_only, points, tets,
      // el_id, discr);

      // 		   Eigen::MatrixXd ids(points.rows(), 1);
      // 		   for (int i = 0; i < points.rows(); ++i)
      // 		   {
      // 			   ids(i) = s.mesh->get_body_id(el_id(i));
      // 		   }
      // 			const bool use_sampler = true;
      // 		   io::Evaluator::interpolate_function(
      // 			*s.mesh, s.problem->is_scalar(), s.bases, s.disc_orders,
      // s.polys, s.polys_3d, s.out_geom.ref_element_sampler, points.rows(),
      // sol, fun, use_sampler, boundary_only);

      // 		   return py::make_tuple(points, tets, el_id, ids, fun); },
      // 	   "returns the solution on a densly sampled mesh, use
      // 'vismesh_rel_area' to control density", py::arg("solution"),
      // py::arg("boundary_only") = bool(false))

      //    .def(
      // 	   "get_stresses", [](State &s, const Eigen::MatrixXd &sol,
      // const double t, bool boundary_only) {
      // 		   //    py::scoped_ostream_redirect output;
      // 		   Eigen::MatrixXd points;
      // 		   Eigen::MatrixXi tets;
      // 		   Eigen::MatrixXi el_id;
      // 		   Eigen::MatrixXd discr;
      // 			auto& vis_bnd =
      // s.args["output"]["advanced"]["vis_boundary_only"]; 		  const bool
      // tmp = vis_bnd; 		  vis_bnd = boundary_only;

      // 		   s.out_geom.build_vis_mesh(*s.mesh, s.disc_orders,
      // s.geom_bases(), s.polys, s.polys_3d, boundary_only, points, tets,
      // el_id, discr); 		   const bool use_sampler = true;

      // 			std::vector<assembler::Assembler::NamedMatrix>
      // tvals; 			io::Evaluator::compute_tensor_value(
      // *s.mesh, s.problem->is_scalar(), s.bases, s.geom_bases(),
      // s.disc_orders, 				s.polys, s.polys_3d, *s.assembler,
      // s.out_geom.ref_element_sampler, 				points.rows(), sol, t,
      // tvals, use_sampler, boundary_only);

      // 			vis_bnd = tmp;
      // 			for (const auto &[name, fun] : tvals)
      // 				if (name == "cauchy_stess")
      // 					return fun;
      // 			throw std::runtime_error("Cauchy stress not found!");
      // 			return Eigen::MatrixXd(); },
      // 	   "returns the stress tensor on a densly sampled mesh, use
      // 'vismesh_rel_area' to control density", py::arg("solution"),
      // py::arg("t"), py::arg("boundary_only") = bool(false))

      //    .def(
      // 	   "get_sampled_mises", [](State &s, const Eigen::MatrixXd
      // &sol, const double t, bool boundary_only) {
      // 		   //    py::scoped_ostream_redirect output;
      // 		   Eigen::MatrixXd points;
      // 		   Eigen::MatrixXi tets;
      // 		   Eigen::MatrixXi el_id;
      // 		   Eigen::MatrixXd discr;

      // 			auto& vis_bnd =
      // s.args["output"]["advanced"]["vis_boundary_only"]; 		  const bool
      // tmp = vis_bnd; 		  vis_bnd = boundary_only;

      // 		   s.out_geom.build_vis_mesh(*s.mesh, s.disc_orders,
      // s.geom_bases(), s.polys, s.polys_3d, boundary_only, points, tets,
      // el_id, discr); 		   	const bool use_sampler = true;

      // 			std::vector<assembler::Assembler::NamedMatrix>
      // tvals; 			io::Evaluator::compute_scalar_value(
      // *s.mesh, s.problem->is_scalar(), s.bases, s.geom_bases(),
      // s.disc_orders, 				s.polys, s.polys_3d, *s.assembler,
      // s.out_geom.ref_element_sampler, 				points.rows(), sol, t,
      // tvals, use_sampler, boundary_only);

      // 			vis_bnd = tmp;
      // 			return tvals[0].second; },
      // 	   "returns the von mises stresses on a densly sampled mesh, use
      // 'vismesh_rel_area' to control density", py::arg("solution"),
      // py::arg("t"), py::arg("boundary_only") = bool(false))

      //    .def(
      // 	   "get_sampled_mises_avg", [](State &s, const
      // Eigen::MatrixXd &sol, const double t, bool boundary_only) {
      // 		   //    py::scoped_ostream_redirect output;
      // 		   Eigen::MatrixXd points;
      // 		   Eigen::MatrixXi tets;
      // 		   Eigen::MatrixXi el_id;
      // 		   Eigen::MatrixXd discr;
      // 		   std::vector<assembler::Assembler::NamedMatrix> fun,
      // tfun;

      // 			auto& vis_bnd =
      // s.args["output"]["advanced"]["vis_boundary_only"]; 		  const bool
      // tmp = vis_bnd; 		  vis_bnd = boundary_only;

      // 		   s.out_geom.build_vis_mesh(*s.mesh, s.disc_orders,
      // s.geom_bases(), s.polys, s.polys_3d, boundary_only, points, tets,
      // el_id, discr); 		   const bool use_sampler = true;
      // 		   io::Evaluator::average_grad_based_function(
      // 			*s.mesh, s.problem->is_scalar(), s.n_bases, s.bases,
      // s.geom_bases(), s.disc_orders, 				s.polys, s.polys_3d,
      // *s.assembler, s.out_geom.ref_element_sampler, t,
      // points.rows(), sol, fun, tfun, use_sampler, boundary_only);

      // 			vis_bnd = tmp;

      // 		   return py::make_tuple(fun[0].second, tfun[0].second); },
      // 	   "returns the von mises stresses and stress tensor averaged around
      // a vertex on a densly sampled mesh, use 'vismesh_rel_area' to control
      // density", py::arg("solution"), py::arg("t"), py::arg("boundary_only") =
      // bool(false))

      //    .def(
      // 	   "get_sampled_traction_forces", [](State &s, const bool
      // apply_displacement, const bool compute_avg) {
      // 		   //    py::scoped_ostream_redirect output;

      // 		   if (!s.mesh)
      // 			   throw pybind11::value_error("Load the mesh first!");
      // 		   if (!s.mesh->is_volume())
      // 			   throw pybind11::value_error("This function works only on
      // volumetric meshes!"); 		   if (s.problem->is_scalar()) throw
      // pybind11::value_error("Define a tensor problem!");

      // 		   Eigen::MatrixXd result, stresses, mises;

      // 		   Eigen::MatrixXd v_surf;
      // 		   Eigen::MatrixXi f_surf;
      // 		   const double epsilon = 1e-10;

      // 		   {
      // 			   const mesh::CMesh3D &mesh3d =
      // *dynamic_cast<mesh::CMesh3D *>(s.mesh.get());
      // Eigen::MatrixXd points(mesh3d.n_vertices(), 3); Eigen::MatrixXi
      // tets(mesh3d.n_cells(), 4);

      // 			   for (int t = 0; t < mesh3d.n_cells(); ++t)
      // 			   {
      // 				   if (mesh3d.n_cell_vertices(t) != 4)
      // 					   throw pybind11::value_error("Works only with tet
      // meshes!");

      // 				   for (int i = 0; i < 4; ++i)
      // 					   tets(t, i) = mesh3d.cell_vertex(t, i);
      // 			   }

      // 			   for (int p = 0; p < mesh3d.n_vertices(); ++p)
      // 				   points.row(p) << mesh3d.point(p);

      // 			   Eigen::MatrixXi f_surf_tmp, _;
      // 			   igl::boundary_facets(tets, f_surf_tmp);
      // 			   igl::remove_unreferenced(points, f_surf_tmp, v_surf,
      // f_surf, _);
      // 		   }

      // 		   if (apply_displacement)
      // 			   s.interpolate_boundary_tensor_function(v_surf, f_surf,
      // s.sol, s.sol, compute_avg, result, stresses, mises, true); else
      // 			   s.interpolate_boundary_tensor_function(v_surf, f_surf,
      // s.sol, compute_avg, result, stresses, mises, true);

      // 		   return py::make_tuple(v_surf, f_surf, result, stresses,
      // mises); }, 	   "returns the traction forces, stresses, and von mises
      // computed on the surface", py::arg("apply_displacement") = bool(false),
      // py::arg("compute_avg") = bool(true))

      ////////////////////////////////////////////////////////////////////////////////////////////
      .def(
          "get_sampled_points_frames",
          [](State &s) {
            //    py::scoped_ostream_redirect output;
            assert(!s.solution_frames.empty());

            std::vector<Eigen::MatrixXd> pts;

            for (const auto &sol : s.solution_frames)
            {
              pts.push_back(sol.points);
            }

            return pts;
          },
          "returns the points frames for a time dependent problem on a densly sampled mesh, use 'vismesh_rel_area' to control density")

      .def(
          "get_sampled_connectivity_frames",
          [](State &s) {
            //    py::scoped_ostream_redirect output;
            assert(!s.solution_frames.empty());

            std::vector<Eigen::MatrixXi> tets;

            for (const auto &sol : s.solution_frames)
              tets.push_back(sol.connectivity);

            return tets;
          },
          "returns the connectivity frames for a time dependent problem on a densly sampled mesh, use 'vismesh_rel_area' to control density")

      .def(
          "get_sampled_solution_frames",
          [](State &s) {
            //    py::scoped_ostream_redirect output;
            assert(!s.solution_frames.empty());

            std::vector<Eigen::MatrixXd> fun;

            for (const auto &sol : s.solution_frames)
            {
              fun.push_back(sol.solution);
            }

            return fun;
          },
          "returns the solution frames for a time dependent problem on a densly sampled mesh, use 'vismesh_rel_area' to control density")

      .def(
          "get_sampled_mises_frames",
          [](State &s) {
            //    py::scoped_ostream_redirect output;
            assert(!s.solution_frames.empty());

            std::vector<Eigen::MatrixXd> mises;

            for (const auto &sol : s.solution_frames)
              mises.push_back(sol.scalar_value);

            return mises;
          },
          "returns the von mises stresses frames on a densly sampled mesh, use 'vismesh_rel_area' to control density")

      .def(
          "get_sampled_mises_avg_frames",
          [](State &s) {
            //    py::scoped_ostream_redirect output;
            assert(!s.solution_frames.empty());

            std::vector<Eigen::MatrixXd> mises;

            for (const auto &sol : s.solution_frames)
              mises.push_back(sol.scalar_value_avg);

            return mises;
          },
          "returns the von mises stresses per frame averaged around a vertex on a densly sampled mesh, use 'vismesh_rel_area' to control density")

      .def(
          "get_boundary_sidesets",
          [](State &s) {
            //    py::scoped_ostream_redirect output;
            Eigen::MatrixXd points;
            Eigen::MatrixXi faces;
            Eigen::MatrixXd sidesets;

            io::Evaluator::get_sidesets(*s.mesh, points, faces, sidesets);

            return py::make_tuple(points, faces, sidesets);
          },
          "exports get the boundary sideset, edges in 2d or trangles in 3d")
      //    .def(
      // 	   "get_body_ids", [](State &s, bool boundary_only) {
      // 		   //    py::scoped_ostream_redirect output;
      // 		   Eigen::MatrixXd points;
      // 		   Eigen::MatrixXi tets;
      // 		   Eigen::MatrixXi el_id;
      // 		   Eigen::MatrixXd discr;

      // 		   s.out_geom.build_vis_mesh(*s.mesh, s.disc_orders,
      // s.geom_bases(), s.polys, s.polys_3d, boundary_only, points, tets,
      // el_id, discr);

      // 		   Eigen::MatrixXd ids(points.rows(), 1);

      // 		   for (int i = 0; i < points.rows(); ++i)
      // 		   {
      // 			   ids(i) = s.mesh->get_body_id(el_id(i));
      // 		   }

      // 		   return py::make_tuple(points, tets, el_id, ids); },
      // 	   "exports get the body ids", py::arg("boundary_only") =
      // bool(false))
      .def(
          "update_dirichlet_boundary",
          [](State &self, const int id, const py::object &val, const bool isx,
             const bool isy, const bool isz, const std::string &interp) {
            using namespace polyfem;
            // py::scoped_ostream_redirect output;
            const json json_val = val;
            if (assembler::GenericTensorProblem *prob0 =
                    dynamic_cast<assembler::GenericTensorProblem *>(
                        self.problem.get()))
            {
              prob0->update_dirichlet_boundary(id, json_val, isx, isy, isz,
                                               interp);
            }
            else if (assembler::GenericScalarProblem *prob1 =
                         dynamic_cast<assembler::GenericScalarProblem *>(
                             self.problem.get()))
            {
              prob1->update_dirichlet_boundary(id, json_val, interp);
            }
            else
            {
              throw "Updating BC works only for GenericProblems";
            }
          },
          "updates dirichlet boundary", py::arg("id"), py::arg("val"),
          py::arg("isx") = bool(true), py::arg("isy") = bool(true),
          py::arg("isz") = bool(true), py::arg("interp") = std::string(""))
      .def(
          "update_neumann_boundary",
          [](State &self, const int id, const py::object &val,
             const std::string &interp) {
            using namespace polyfem;
            // py::scoped_ostream_redirect output;
            const json json_val = val;

            if (auto prob0 = dynamic_cast<assembler::GenericTensorProblem *>(
                    self.problem.get()))
            {
              prob0->update_neumann_boundary(id, json_val, interp);
            }
            else if (auto prob1 =
                         dynamic_cast<assembler::GenericScalarProblem *>(
                             self.problem.get()))
            {
              prob1->update_neumann_boundary(id, json_val, interp);
            }
            else
            {
              throw "Updating BC works only for GenericProblems";
            }
          },
          "updates neumann boundary", py::arg("id"), py::arg("val"),
          py::arg("interp") = std::string(""))
      .def(
          "update_pressure_boundary",
          [](State &self, const int id, const py::object &val,
             const std::string &interp) {
            using namespace polyfem;
            // py::scoped_ostream_redirect output;
            const json json_val = val;

            if (auto prob = dynamic_cast<assembler::GenericTensorProblem *>(
                    self.problem.get()))
            {
              prob->update_pressure_boundary(id, json_val, interp);
            }
            else
            {
              throw "Updating BC works only for Tensor GenericProblems";
            }
          },
          "updates pressure boundary", py::arg("id"), py::arg("val"),
          py::arg("interp") = std::string(""))
      .def(
          "update_obstacle_displacement",
          [](State &self, const int oid, const py::object &val,
             const std::string &interp) {
            using namespace polyfem;
            // py::scoped_ostream_redirect output;
            const json json_val = val;
            self.obstacle.change_displacement(oid, json_val, interp);
          },
          "updates obstacle displacement", py::arg("oid"), py::arg("val"),
          py::arg("interp") = std::string(""));
  //    .def(
  // 	   "render", [](State &self, const Eigen::MatrixXd &sol, int
  // width, int height, const Eigen::MatrixXd &camera_positionm, const double
  // camera_fov, const double camera_near, const double camera_far, const bool
  // is_perspective, const Eigen::MatrixXd &lookatm, const Eigen::MatrixXd &upm,
  // const Eigen::MatrixXd &ambient_lightm, const
  // std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &lights, const
  // Eigen::MatrixXd &diffuse_colorm, const Eigen::MatrixXd &specular_colorm,
  // const double specular_exponent) { 		   using namespace Eigen;

  // 		   const int problem_dim = self.problem->is_scalar() ? 1 :
  // self.mesh->dimension();

  // 		   Eigen::MatrixXd tmp = self.collision_mesh.rest_positions();
  // 		   assert(tmp.rows() * problem_dim == sol.size());
  // 		   for (int i = 0; i < sol.size(); i += problem_dim)
  // 		   {
  // 			   for (int d = 0; d < problem_dim; ++d)
  // 			   {
  // 				   tmp(i / problem_dim, d) += sol(i + d);
  // 			   }
  // 		   }

  // 		   return rendering_lambda(tmp, self.collision_mesh.faces(), width,
  // height, camera_positionm, camera_fov, camera_near, camera_far,
  // is_perspective, lookatm, upm, ambient_lightm, lights, diffuse_colorm,
  // specular_colorm, specular_exponent); }, 	   "renders the scene",
  // py::arg("solution"), py::kw_only(), py::arg("width"), py::arg("height"),
  // py::arg("camera_position") = Eigen::MatrixXd(), py::arg("camera_fov") =
  // double(0.8), py::arg("camera_near") = double(1), py::arg("camera_far") =
  // double(10), py::arg("is_perspective") = bool(true), py::arg("lookat") =
  // Eigen::MatrixXd(), py::arg("up") = Eigen::MatrixXd(),
  // py::arg("ambient_light") = Eigen::MatrixXd(), py::arg("lights") =
  // std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>(),
  // py::arg("diffuse_color") = Eigen::MatrixXd(), py::arg("specular_color") =
  // Eigen::MatrixXd(), py::arg("specular_exponent") = double(1))
  //    .def(
  // 	   "render_extrinsic", [](State &, const
  // std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &vertex_face, int
  // width, int height, const Eigen::MatrixXd &camera_positionm, const double
  // camera_fov, const double camera_near, const double camera_far, const bool
  // is_perspective, const Eigen::MatrixXd &lookatm, const Eigen::MatrixXd &upm,
  // const Eigen::MatrixXd &ambient_lightm, const
  // std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &lights, const
  // Eigen::MatrixXd &diffuse_colorm, const Eigen::MatrixXd &specular_colorm,
  // const double specular_exponent) { 		   int v_count = 0; 		   int
  // f_count = 0; 		   for (const auto& vf_pair : vertex_face) {
  // v_count += vf_pair.first.rows(); 			   f_count +=
  // vf_pair.second.rows();
  // 		   }
  // 		   Eigen::MatrixXd vertices(v_count, 3);
  // 		   Eigen::MatrixXi faces(f_count, 3);
  // 		   v_count = 0;
  // 		   f_count = 0;
  // 		   for (const auto& vf_pair : vertex_face) {
  // 			   vertices.block(v_count, 0, vf_pair.first.rows(), 3) =
  // vf_pair.first; 			   faces.block(f_count, 0,
  // vf_pair.second.rows(), 3) = (vf_pair.second.array() + v_count).matrix();
  // v_count += vf_pair.first.rows(); 			   f_count +=
  // vf_pair.second.rows();
  // 		   }
  // 		   return rendering_lambda(vertices, faces, width, height,
  // camera_positionm, camera_fov, camera_near, camera_far, is_perspective,
  // lookatm, upm, ambient_lightm, lights, diffuse_colorm, specular_colorm,
  // specular_exponent); }, 	   "renders the extrinsic scene", py::kw_only(),
  // py::arg("vertex_face") = std::vector<std::pair<Eigen::MatrixXd,
  // Eigen::MatrixXi>>(), py::arg("width"), py::arg("height"),
  // py::arg("camera_position") = Eigen::MatrixXd(), py::arg("camera_fov") =
  // double(0.8), py::arg("camera_near") = double(1), py::arg("camera_far") =
  // double(10), py::arg("is_perspective") = bool(true), py::arg("lookat") =
  // Eigen::MatrixXd(), py::arg("up") = Eigen::MatrixXd(),
  // py::arg("ambient_light") = Eigen::MatrixXd(), py::arg("lights") =
  // std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>(),
  // py::arg("diffuse_color") = Eigen::MatrixXd(), py::arg("specular_color") =
  // Eigen::MatrixXd(), py::arg("specular_exponent") = double(1));
}

void define_solve(py::module_ &m)
{

  m.def(
      "polyfem_command",
      [](const std::string &json_file, const std::string &yaml_file,
         const int log_level, const bool strict_validation,
         const int max_threads, const std::string &output_dir) {
        json in_args = json({});

        const bool ok = !json_file.empty() ? load_json(json_file, in_args)
                                           : load_yaml(yaml_file, in_args);

        if (!ok)
          throw std::runtime_error(
              fmt::format("unable to open {} file", json_file));

        json tmp = json::object();
        tmp["/output/log/level"_json_pointer] = int(log_level);
        tmp["/solver/max_threads"_json_pointer] = max_threads;
        if (!output_dir.empty())
          tmp["/output/directory"_json_pointer] =
              std::filesystem::absolute(output_dir);
        assert(tmp.is_object());
        in_args.merge_patch(tmp);

        std::vector<std::string> names;
        std::vector<Eigen::MatrixXi> cells;
        std::vector<Eigen::MatrixXd> vertices;

        State state;
        state.init(in_args, strict_validation);
        state.load_mesh(/*non_conforming=*/false, names, cells, vertices);

        // Mesh was not loaded successfully; load_mesh() logged the error.
        if (state.mesh == nullptr)
          throw std::runtime_error("Failed to load the mesh!");

        state.stats.compute_mesh_stats(*state.mesh);

        state.build_basis();

        state.assemble_rhs();
        state.assemble_mass_mat();

        Eigen::MatrixXd sol;
        Eigen::MatrixXd pressure;

        state.solve_problem(sol, pressure);

        state.compute_errors(sol);

        state.save_json(sol);
        state.export_data(sol, pressure);
      },
      "runs the polyfem command, internal usage", py::kw_only(),
      py::arg("json"), py::arg("yaml") = std::string(""),
      py::arg("log_level") = int(1), py::arg("strict_validation") = bool(true),
      py::arg("max_threads") = int(1), py::arg("output_dir") = "");

  //   m.def(
  //       "solve_febio",
  //       [](const std::string &febio_file, const std::string &output_path,
  //          const int log_level, const py::kwargs &kwargs) {
  //         if (febio_file.empty())
  //           throw pybind11::value_error("Specify a febio file!");

  //         // json in_args = opts.is_none() ? json({}) : json(opts);
  //         json in_args = json(static_cast<py::dict>(kwargs));

  //         if (!output_path.empty())
  //         {
  //           in_args["export"]["paraview"] = output_path;
  //           in_args["export"]["wire_mesh"] =
  //               utils::StringUtils::replace_ext(output_path, "obj");
  //           in_args["export"]["material_params"] = true;
  //           in_args["export"]["body_ids"] = true;
  //           in_args["export"]["contact_forces"] = true;
  //           in_args["export"]["surface"] = true;
  //         }

  //         const int discr_order =
  //             in_args.contains("discr_order") ? int(in_args["discr_order"]) :
  //             1;

  //         if (discr_order == 1 && !in_args.contains("vismesh_rel_area"))
  //           in_args["output"]["paraview"]["vismesh_rel_area"] = 1e10;

  //         State state;
  //         state.init_logger("", log_level, false);
  //         state.init(in_args);
  //         state.load_febio(febio_file, in_args);
  //         state.stats.compute_mesh_stats(*state.mesh);

  //         state.build_basis();

  //         state.assemble_rhs();
  //         state.assemble_mass_mat();

  //         Eigen::MatrixXd sol, pressure;
  //         state.solve_problem(sol, pressure);

  //         state.save_json();
  //         state.export_data(sol, pressure);
  //       },
  //       "runs FEBio", py::arg("febio_file"),
  //       py::arg("output_path") = std::string(""), py::arg("log_level") = 2);

  //   m.def(
  //       "solve",
  //       [](const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &cells,
  //          const py::object &sidesets_func, const int log_level,
  //          const py::kwargs &kwargs) {
  //         std::string log_file = "";

  //         std::unique_ptr<State> res =
  //             std::make_unique<State>();
  //         State &state = *res;
  //         state.init_logger(log_file, log_level, false);

  //         json in_args = json(static_cast<py::dict>(kwargs));

  //         state.init(in_args);

  //         state.load_mesh(vertices, cells);

  //         [&]() {
  //           if (!sidesets_func.is_none())
  //           {
  //             try
  //             {
  //               const auto fun =
  //                   sidesets_func
  //                       .cast<std::function<int(const RowVectorNd
  //                       &)>>();
  //               state.mesh->compute_boundary_ids(fun);
  //               return true;
  //             }
  //             catch (...)
  //             {
  //               {
  //               }
  //             }
  //             try
  //             {
  //               const auto fun = sidesets_func.cast<
  //                   std::function<int(const RowVectorNd &,
  //                   bool)>>();
  //               state.mesh->compute_boundary_ids(fun);
  //               return true;
  //             }
  //             catch (...)
  //             {
  //             }

  //             try
  //             {
  //               const auto fun = sidesets_func.cast<
  //                   std::function<int(const std::vector<int> &, bool)>>();
  //               state.mesh->compute_boundary_ids(fun);
  //               return true;
  //             }
  //             catch (...)
  //             {
  //             }

  //             throw pybind11::value_error(
  //                 "sidesets_func has invalid type, should be a function
  //                 (p)->int, (p, bool)->int, ([], bool)->int");
  //           }
  //         }();

  //         state.stats.compute_mesh_stats(*state.mesh);

  //         state.build_basis();

  //         state.assemble_rhs();
  //         state.assemble_mass_mat();
  //         state.solve_problem();

  //         return res;
  //       },
  //       "single solve function", py::kw_only(),
  //       py::arg("vertices") = Eigen::MatrixXd(),
  //       py::arg("cells") = Eigen::MatrixXi(),
  //       py::arg("sidesets_func") = py::none(), py::arg("log_level") = 2);
}
