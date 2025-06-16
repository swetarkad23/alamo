//
// Set each point in the field to a random value between 0 and 1
//

#ifndef IC_RANDOM_H_
#define IC_RANDOM_H_

#include "IC/IC.H"
#include "IO/ParmParse.H"
#include "Set/Set.H"
#include "Util/Util.H"

#include "AMReX_GpuQualifiers.H"
#include "AMReX_Vector.H"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace IC
{
namespace Distributions
{

/// Generate a standard normal random number using Box-Muller transform
/// Note: This is a simplified version without the spare value optimization
/// to avoid thread safety issues on GPU
AMREX_GPU_DEVICE
inline Set::Scalar Normal(Set::Scalar mean = 0.0, Set::Scalar stddev = 1.0)
{
    Set::Scalar u1 = Util::Random();
    Set::Scalar u2 = Util::Random();

    // Avoid log(0)
    if (u1 < 1e-30)
        u1 = 1e-30;

    Set::Scalar z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    return mean + stddev * z0;
}

/// Generate a gamma-distributed random number using Marsaglia-Tsang method
AMREX_GPU_DEVICE
inline Set::Scalar Gamma(Set::Scalar shape, Set::Scalar scale = 1.0)
{
    if (shape <= 0.0)
        return 0.0;

    if (shape < 1.0)
    {
        // For shape < 1, use Johnk's generator
        Set::Scalar u = Util::Random();
        return scale * Gamma(shape + 1.0, 1.0) * std::pow(u, 1.0 / shape);
    }

    // Marsaglia-Tsang method for shape >= 1
    Set::Scalar d = shape - 1.0 / 3.0;
    Set::Scalar c = 1.0 / std::sqrt(9.0 * d);

    while (true)
    {
        Set::Scalar x, v;
        do
        {
            x = Normal();
            v = 1.0 + c * x;
        } while (v <= 0.0);

        v = v * v * v;
        Set::Scalar u = Util::Random();

        if (u < 1.0 - 0.0331 * x * x * x * x)
            return scale * d * v;

        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v)))
            return scale * d * v;
    }
}

/// Generate a beta-distributed random number
AMREX_GPU_DEVICE
inline Set::Scalar Beta(Set::Scalar alpha, Set::Scalar beta)
{
    if (alpha <= 0.0 || beta <= 0.0)
        return 0.5;

    if (alpha == 1.0 && beta == 1.0)
        return Util::Random();

    Set::Scalar x = Gamma(alpha);
    Set::Scalar y = Gamma(beta);

    Set::Scalar sum = x + y;
    if (sum < 1e-30)
        return 0.5;

    return x / sum;
}
}

/// Set each point to a random value.
class Random : public IC<Set::Scalar>
{
public:
    static constexpr const char *name = "random";

    Random(amrex::Vector<amrex::Geometry> &_geom, Set::Scalar a_mult) : IC(_geom), mult(a_mult)
    {
    }
    Random(amrex::Vector<amrex::Geometry> &_geom, IO::ParmParse &pp, std::string name) : IC(_geom)
    {
        pp.queryclass(name, *this);
    }

    void Add(const int &lev, Set::Field<Set::Scalar> &a_field, Set::Scalar)
    {
        for (amrex::MFIter mfi(*a_field[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box bx;
            amrex::IndexType type = a_field[lev]->ixType();
            if (type == amrex::IndexType::TheCellType())
                bx = mfi.growntilebox();
            else if (type == amrex::IndexType::TheNodeType())
                bx = mfi.grownnodaltilebox();
            else
                Util::Abort(INFO, "Unknown index type");

            Set::Patch<Set::Scalar> field = a_field.Patch(lev, mfi);
            for (int n = 0; n < a_field[lev]->nComp(); n++)
            {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    field(i, j, k, n) += offset + mult * Util::Random();
                });
            }
        }
    }

    using IC::Add;

    static void Parse(Random &value, IO::ParmParse &pp)
    {
        // offset from the [0,1] random number range
        pp.query_default("offset", value.offset, 0.0);
        // multiplier for the [0,1] random number range
        pp.query_default("mult", value.mult, 1.0);
    }

protected:
    Set::Scalar mult = NAN;
    Set::Scalar offset = NAN;
};
}
#endif
