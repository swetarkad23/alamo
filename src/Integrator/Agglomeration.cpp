#include "Agglomeration.H"
#include "BC/Constant.H"
#include "IO/ParmParse.H"

namespace Integrator
{
void
Agglomeration::Parse(Agglomeration &value, IO::ParmParse &pp)
{
    // Chemical potential
    pp.query_default("gamma", value.gamma, 0.0005);
    // Surface tension
    pp.query_default("kappa", value.kappa, 0.001);
    // Agglomeration mobility
    pp.query_default("L0", value.L0, 0.5);
    // Agglomeration mobility exponent
    pp.query_default("n", value.n, 2.0);

    // Boundary conditions for aluminum phase order pamarater aluminum
    // order parameter
    pp.select_default<BC::Constant>("alpha.bc", value.bc_alpha, 1);
    value.RegisterNewFab(value.alpha_mf, value.bc_alpha, 1, value.ghost_count, "alpha", true);
}
}
