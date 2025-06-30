#include <cmath>
#include <utility>

#include "AMReX_MultiFabUtil.H"
#include "Agglomeration.H"
#include "Flame.H"

#include "BC/Constant.H"
#include "IC/BMP.H"
#include "IC/BetaDistribution.H"
#include "IC/Constant.H"
#include "IC/Expression.H"
#include "IC/PNG.H"
#include "IC/Random.H"
#include "IO/ParmParse.H"
#include "Numeric/Stencil.H"
#include "Set/Base.H"
#include "Set/Set.H"

#include "AMReX_Array4.H"
#include "AMReX_Box.H"
#include "AMReX_GpuLaunchFunctsC.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MFIter.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_TagBox.H"

namespace Integrator
{
void
Agglomeration::Parse(Agglomeration &value, IO::ParmParse &pp)
{
    pp.queryclass<Flame>("flame", &value);

    // Method to use for agglomeration kinetics
    pp.query_enum_case_insensitive("kinetics_method", value.agglom.kinetics_method);

    // Diffuse interface length of the agglomerate phase
    pp.query_default("epsilon", value.agglom.epsilon, 1e-7);
    // Chemical potential of the agglomerating material
    pp.query_default("gamma", value.agglom.gamma, 0.0005);
    // Surface tension of the agglomerating material
    pp.query_default("kappa", value.agglom.kappa, 1.0);
    // Agglomeration mobility
    pp.query_default("L_0", value.agglom.L_0, 1.0);
    // Agglomeration mobility exponent
    pp.query_default("n", value.agglom.n, 1.0);

    if (value.agglom.kinetics_method == AgglomerationKinetics::ConstrainedAllenCahn)
    {
        // Lagrangian multiplier for the agglomerate volume fraction
        // constraint
        pp.query_default("lambda", value.agglom.lambda, 100.0);
        // Prescribed volume fraction of agglomerate in the domain. If
        // unset, it will default to the initial volume fraction of
        // agglomerate in the domain based on the given initial
        // condition.
        pp.query("V_0", value.agglom.V_0);
    }

    // Regridding criterion
    pp.query_default("refinement_threshold", value.refinement_threshold, 1e100);
    // Smallest non-zero value
    pp.query_default("small", value.small, 1e-4);

    // Initial conditions for agglomerate order parameter
    pp.select_default<IC::Constant, IC::Expression, IC::BMP, IC::PNG, IC::Random, IC::BetaDistribution>("alpha.ic", value.agglom.alpha_ic, value.geom);
    // Boundary conditions for agglomerate order parameter
    pp.select_default<BC::Constant>("alpha.bc", value.agglom.alpha_bc, 1);

    value.RegisterNewFab(value.agglom.alpha_old, value.agglom.alpha_bc, 1, 1, "agglom.alpha_old", false);
    value.RegisterNewFab(value.agglom.alpha, value.agglom.alpha_bc, 1, 1, "agglom.alpha", true);
    value.RegisterNewFab(value.agglom.free_energy_derivative, value.agglom.alpha_bc, 1, 1, "agglom.free_energy_derivative", true);

    if (value.agglom.kinetics_method == AgglomerationKinetics::ConstrainedAllenCahn)
        value.RegisterIntegratedVariable(&value.agglom.V, "agglom.V");
}

void
Agglomeration::Initialize(int lev)
{
    Flame::Initialize(lev);
    agglom.alpha_old[lev]->setVal(0.0);
    agglom.alpha_ic->Initialize(lev, agglom.alpha);
    agglom.free_energy_derivative[lev]->setVal(0.0);

    int nComp = agglom.alpha[lev]->nComp();
    int nGrow = agglom.alpha[lev]->nGrow();
    MultiFab cell_based_phi(agglom.alpha[lev]->boxArray(), agglom.alpha[lev]->DistributionMap(), nComp, nGrow);
    average_node_to_cellcenter(cell_based_phi, 0, *phi_mf[lev], 0, nComp, nGrow);

    scaleByComplement(*agglom.alpha[lev], cell_based_phi, 0, 0, nComp, nGrow);
}

void
Agglomeration::scaleByComplement(MultiFab &dst, const MultiFab &src, int srccomp, int dstcomp, int numcomp, int nghost)
{
    int nCompSrc = src.nComp();
    int nGrowSrc = src.nGrow();

    for (int comp = 0; comp < numcomp; ++comp)
    {
        Util::Assert(INFO, TEST(src.min(srccomp + comp) >= 0.0));
        Util::Assert(INFO, TEST(src.max(srccomp + comp) <= 1.0));
    }

    MultiFab complement(src.boxArray(), src.DistributionMap(), nCompSrc, nGrowSrc);
    complement.setVal(1.0);

    MultiFab::Subtract(complement, src, 0, 0, nCompSrc, nGrowSrc);
    MultiFab::Multiply(dst, complement, srccomp, dstcomp, numcomp, nghost);
}

void
Agglomeration::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    Flame::Advance(lev, time, dt);

    std::swap(agglom.alpha_old[lev], agglom.alpha[lev]);
    const Set::Scalar *dx = geom[lev].CellSize();

    int agglom_volume_fraction = agglom.V / geom[0].ProbSize();

    // I'm not sure if this is thread-safe. But I don't think it
    // matters. @brunnels is this okay? If a prescribed agglom.V_0 is
    // not configured, I want to set the prescribed agglomerate volume
    // fraction based on the initial condition of the
    // agglomerate. This helps me when I'm debugging/testing with
    // random ICs.
    if (time == 0.0 && std::isnan(agglom.V_0))
    {
        agglom.V_0 = agglom_volume_fraction;
    }

    for (amrex::MFIter mfi(*agglom.alpha[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        Set::Patch<const Set::Scalar> alpha_old = agglom.alpha_old.Patch(lev, mfi);
        Set::Patch<Set::Scalar> alpha = agglom.alpha.Patch(lev, mfi);
        Set::Patch<Set::Scalar> free_energy_derivative = agglom.free_energy_derivative.Patch(lev, mfi);
        Set::Patch<Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // calculate the Laplacian of alpha
            Set::Scalar laplacian_alpha = Numeric::Laplacian(alpha_old, i, j, k, 0, dx);

            // calculate the variational derivative
            free_energy_derivative(i, j, k) = 2 * agglom.gamma / agglom.epsilon * alpha_old(i, j, k) * (1 - alpha_old(i, j, k)) * (1 - 2 * alpha_old(i, j, k)) - agglom.epsilon * agglom.kappa * laplacian_alpha;
        });

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // calculate effective mobility L
            Set::Scalar L = agglom.L_0 * std::pow(1 - eta(i, j, k), agglom.n);

            if (agglom.kinetics_method == AgglomerationKinetics::ConstrainedAllenCahn)
                // constrained Allen-Cahn equation
                alpha(i, j, k) = alpha_old(i, j, k) + dt * (-L * free_energy_derivative(i, j, k) + agglom.lambda * (agglom_volume_fraction - agglom.V_0));
            else if (agglom.kinetics_method == AgglomerationKinetics::CahnHilliard)
            {
                // calculate the Laplacian of the variational derivative
                Set::Scalar free_energy_laplacian = Numeric::Laplacian(free_energy_derivative, i, j, k, 0, dx);

                // Cahn-Hilliard equation
                alpha(i, j, k) = alpha_old(i, j, k) + dt * L * free_energy_laplacian;
            }
        });
    }
}

void
Agglomeration::TagCellsForRefinement(int lev, amrex::TagBoxArray &a_tags, Set::Scalar time, int ngrow)
{
    Flame::TagCellsForRefinement(lev, a_tags, time, ngrow);

    const Set::Vector dx(geom[lev].CellSize());
    Set::Scalar dr = dx.lpNorm<2>();

    for (amrex::MFIter mfi(*agglom.alpha[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        Set::Patch<char> tags = a_tags.array(mfi);
        Set::Patch<const Set::Scalar> alpha = agglom.alpha.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector grad = Numeric::Gradient(alpha, i, j, k, 0, dx.data());
            if (grad.lpNorm<2>() * dr > refinement_threshold)
                tags(i, j, k) = amrex::TagBox::SET;
        });
    }
}

void
Agglomeration::Integrate(int amrlev, Set::Scalar time, int step, const amrex::MFIter &mfi, const amrex::Box &box)
{
    Flame::Integrate(amrlev, time, step, mfi, box);

    if (agglom.kinetics_method != AgglomerationKinetics::ConstrainedAllenCahn)
        return;

    const Set::Scalar *dx = geom[amrlev].CellSize();
    Set::Scalar dv = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    Set::Patch<const Set::Scalar> alpha = agglom.alpha.Patch(amrlev, mfi);
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        agglom.V += alpha(i, j, k, 0) * dv;
    });
}
}
