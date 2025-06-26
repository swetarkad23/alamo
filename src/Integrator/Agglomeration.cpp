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

#include "AMReX_Algorithm.H"
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

    // Diffuse interface length of the agglomerate phase
    pp.query_default("eps", value.agglom.eps, 1e-7);
    // Chemical potential of the agglomerating material
    pp.query_default("gamma", value.agglom.gamma, 0.0005);
    // Surface tension of the agglomerating material
    pp.query_default("kappa", value.agglom.kappa, 1.0);
    // Agglomeration mobility
    pp.query_default("L0", value.agglom.L0, 1.0);
    // Agglomeration mobility exponent
    pp.query_default("n", value.agglom.n, 1.0);

    // Regridding criterion
    pp.query_default("refinement_threshold", value.refinement_threshold, 1e100);

    // Initial conditions for agglomerate order parameter
    pp.select_default<IC::Constant, IC::Expression, IC::BMP, IC::PNG, IC::Random, IC::BetaDistribution>("alpha_agglom.ic", value.ic_alpha_agglom, value.geom);
    // Boundary conditions for agglomerate order parameter
    pp.select_default<BC::Constant>("alpha_agglom.bc", value.bc_alpha_agglom, 1);

    value.RegisterNewFab(value.alphaold_agglom_mf, value.bc_alpha_agglom, 1, 1, "alpha_agglom_old", false);
    value.RegisterNewFab(value.alpha_agglom_mf, value.bc_alpha_agglom, 1, 1, "alpha_agglom", true);
    value.RegisterNewFab(value.free_energy_agglom_derivative_mf, value.bc_alpha_agglom, 1, 1, "free_energy_agglom_derivative", true);
};

void
Agglomeration::Initialize(int lev)
{
    Flame::Initialize(lev);
    alphaold_agglom_mf[lev]->setVal(0.0);
    ic_alpha_agglom->Initialize(lev, alpha_agglom_mf);
    free_energy_agglom_derivative_mf[lev]->setVal(0.0);

    int nComp = alpha_agglom_mf[lev]->nComp();
    int nGrow = alpha_agglom_mf[lev]->nGrow();
    MultiFab cell_based_phi(alpha_agglom_mf[lev]->boxArray(), alpha_agglom_mf[lev]->DistributionMap(), nComp, nGrow);
    average_node_to_cellcenter(cell_based_phi, 0, *phi_mf[lev], 0, nComp, nGrow);

    scaleByComplement(*alpha_agglom_mf[lev], cell_based_phi, 0, 0, nComp, nGrow);
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
    std::swap(alphaold_agglom_mf[lev], alpha_agglom_mf[lev]);
    const Set::Scalar *DX = geom[lev].CellSize();
    for (amrex::MFIter mfi(*alpha_agglom_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        Set::Patch<const Set::Scalar> alphaold_agglom = alphaold_agglom_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> alpha_agglom = alpha_agglom_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> free_energy_agglom_derivative = free_energy_agglom_derivative_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar laplacian_alpha = Numeric::Laplacian(alphaold_agglom, i, j, k, 0, DX);

            // calculate the variational derivative
            free_energy_agglom_derivative(i, j, k) = 2 * agglom.eps * agglom.gamma * alphaold_agglom(i, j, k) * (1 - alphaold_agglom(i, j, k)) * (1 - 2 * alphaold_agglom(i, j, k)) - agglom.kappa / agglom.eps * laplacian_alpha;
        });

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // calculate the Laplacian of the variational derivative
            Set::Scalar laplacian = Numeric::Laplacian(free_energy_agglom_derivative, i, j, k, 0, DX);

            // calculate effective mobility L
            Set::Scalar L = agglom.L0 * std::pow(1 - eta(i, j, k), agglom.n);

            // Cahn-Hilliard equation
            alpha_agglom(i, j, k) = alphaold_agglom(i, j, k) + dt * L * laplacian;
            alpha_agglom(i, j, k) = amrex::Clamp(alpha_agglom(i, j, k), small, 1.0);
        });
    }
}

void
Agglomeration::TagCellsForRefinement(int lev, amrex::TagBoxArray &a_tags, Set::Scalar time, int ngrow)
{
    Flame::TagCellsForRefinement(lev, a_tags, time, ngrow);

    const Set::Vector DX(geom[lev].CellSize());
    Set::Scalar dr = DX.lpNorm<2>();

    for (amrex::MFIter mfi(*alpha_agglom_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        Set::Patch<char> tags = a_tags.array(mfi);
        Set::Patch<const Set::Scalar> alpha_agglom = alpha_agglom_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector grad = Numeric::Gradient(alpha_agglom, i, j, k, 0, DX.data());
            if (grad.lpNorm<2>() * dr > refinement_threshold)
                tags(i, j, k) = amrex::TagBox::SET;
        });
    }
}
}
