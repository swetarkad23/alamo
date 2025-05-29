#ifndef INTEGRATOR_AGGLOMERATION_H
#define INTEGRATOR_AGGLOMERATION_H

#include "IO/ParmParse.H"
#include "Integrator/Integrator.H"

namespace Integrator
{

class Agglomeration : public Integrator
{

public:
    static constexpr const char *name = "agglomeration";

    Agglomeration();
    ~Agglomeration();
    Agglomeration(IO::ParmParse &pp) : Agglomeration() { Parse(*this, pp); };

    static void Parse(Agglomeration &value, IO::ParmParse &pp);

private:
    Set::Field<Set::Scalar> alpha_mf;
};
}

#endif
