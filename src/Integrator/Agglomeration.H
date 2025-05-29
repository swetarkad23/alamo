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
};
}

#endif
