#include "mwf_candidate_index.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using Index = vllm::MWFCandidateIndex;

PYBIND11_MODULE(_mwf_candidate_index, m) {
  py::enum_<Index::Kind>(m, "CandidateKind")
      .value("UNHASHED", Index::Kind::Unhashed)
      .value("MISSING", Index::Kind::Missing)
      .value("TRACKED", Index::Kind::Tracked);

  py::class_<Index>(m, "MWFCandidateIndex")
      .def(py::init<double, double>(),
           py::arg("alpha"), py::arg("beta"))
      .def("add", &Index::add,
           py::arg("block_id"), py::arg("kind"),
           py::arg("key_a") = 0.0, py::arg("key_b") = 0.0)
      .def("update", &Index::update,
           py::arg("block_id"), py::arg("kind"),
           py::arg("key_a") = 0.0, py::arg("key_b") = 0.0)
      .def("erase", &Index::erase, py::arg("block_id"))
      .def("pop_n", &Index::pop_n,
           py::arg("num_blocks"), py::arg("elapsed"))
      .def("clear", &Index::clear)
      .def("size", &Index::size);
}