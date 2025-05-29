#include "Agglomeration.H"
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
}
}
