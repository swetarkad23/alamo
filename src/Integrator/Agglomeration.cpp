#include "Agglomeration.H"
#include "BC/Constant.H"
#include "IC/BMP.H"
#include "IC/Constant.H"
#include "IC/Expression.H"
#include "IC/PNG.H"
#include "IO/ParmParse.H"
#include "Set/Base.H"

#include "Flame.H"

namespace Integrator
{
void
Agglomeration::Parse(Agglomeration &value, IO::ParmParse &pp)
{
    pp.queryclass<Flame>("flame", &value);

    // Chemical potential of the agglomerating material
    pp.query_default("gamma", value.gamma_agglom, 0.0005);
    // Surface tension of the agglomerating material
    pp.query_default("kappa", value.kappa_agglom, 1.0);
    // Agglomeration mobility
    pp.query_default("L0", value.L0, 1.0);
    // Agglomeration mobility exponent
    pp.query_default("n", value.n, 1.0);

    // Boundary conditions for agglomerate order parameter
    pp.select_default<BC::Constant>("alpha_agglom.bc", value.bc_alpha_agglom, 1);
    // Initial conditions for agglomerate order parameter
    pp.select_default<IC::Constant, IC::Expression, IC::BMP, IC::PNG>("alpha_agglom.ic", value.ic_alpha_agglom, value.geom);

    value.RegisterNewFab(value.alphaold_agglom_mf, value.bc_alpha_agglom, 1, 1, "alpha_agglom_old", false);
    value.RegisterNewFab(value.alphanew_agglom_mf, value.bc_alpha_agglom, 1, 1, "alpha_agglom", true);
};

void
Agglomeration::Initialize(int lev)
{
    Flame::Initialize(lev);
    ic_alpha_agglom->Initialize(lev, alphaold_agglom_mf);
    ic_alpha_agglom->Initialize(lev, alphanew_agglom_mf);
}

void
Agglomeration::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    Flame::Advance(lev, time, dt);
    const Set::Scalar *DX = geom[lev].CellSize();
    // TODO: Implement Cahn-Hilliard agglomeration kinetics
}
}
