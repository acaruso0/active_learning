#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ps.h"

PYBIND11_MODULE(ps, m) // name of the output module, variable
{
    m.doc() = "Partridge-Schwenke";
    m.def("energy", ps::pot_nasa, "Retrieves 1B energy water");
}
