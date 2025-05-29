//
// This implements a phase field model for agglomeration of
// nanoparticles in an SCP. The chemical potential :math:`W_C[\alpha]`
// is defined as
//
// .. math::
//    W_C[\alpha] = \epsilon\gamma\alpha^2(1-\alpha)^2,
//
// where :math:`\epsilon` is the diffuse interface width and
// :math:`\gamma` is a parameter. To capture the surface tension of
// the agglomerate, we define
//
// .. math::
//    W_B[\alpha] = \frac{1}{2\epsilon}\kappa|\nabla\alpha|^2,
//
// where :math:`\kappa` is a parameter. The free energy is
//
// .. math::
//    F[\alpha] = \int_\Omega(W_c[\alpha]+W_B[\alpha])\mathrm{d}\bm{x}.
//
// The effective mobility :math:`L` is defined as
//
// .. math::
//    L = L_0(1-\eta)^n,
//
// where :math:`L_0` is the mobility parameter, and :math:`n` is some
// constant greater than 0.
//
// The :math:`\alpha` phase is evolved according to Cahn-Hilliard
// kinetics:
//
// .. math::
//    \frac{\partial\alpha}{\partial t} = L\Delta]\left(\frac{\delta F}{\delta\alpha}\right)

#ifndef INTEGRATOR_AGGLOMERATION_H
#define INTEGRATOR_AGGLOMERATION_H

#include "IO/ParmParse.H"
#include "Integrator/Flame.H"

namespace Integrator
{

class Agglomeration : public Flame
{

public:
    static constexpr const char *name = "agglomeration";

    Agglomeration() = default;
    ~Agglomeration() = default;
    Agglomeration(IO::ParmParse &pp) : Agglomeration() { Parse(*this, pp); };

    static void Parse(Agglomeration &value, IO::ParmParse &pp);

private:
    Set::Field<Set::Scalar> alpha_mf;

    Set::Scalar gamma = NAN;
    Set::Scalar kappa = NAN;
    Set::Scalar L0 = NAN;
    Set::Scalar n = NAN;
};
}

#endif
