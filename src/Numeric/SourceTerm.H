#ifndef NUMERIC_SOURCETERM_H
#define NUMERIC_SOURCETERM_H

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include "Set/Set.H"
#include "Integrator/ScimitarX.H"
#include "Util/Util.H"
#include "Util/ScimitarX_Util.H"
#include "Model/Fluid/Fluid.H"
#include "Numeric/Stencil.H"
#include "Numeric/NumericTypes.H"
#include "Numeric/SolverCapabilities.H"
#include "Numeric/IntegratorVariableAccessLayer.H"
#include "Numeric/SymmetryPreservingRoeAveragingOperations.H"
#include "Integrator/ScimitarX.H"
#include "Numeric/FluxHandler.H"
#include <memory>
#include <cmath>
#include <algorithm>

//forward declaration for ScimitarX
namespace Integrator {
    class ScimitarX;
}

namespace Numeric {

AMREX_GPU_DEVICE
inline void ClampIndices(int indices[3], const int lower_bounds[3], const int upper_bounds[3]);

template <typename IntegratorDerivedClass>
class SourceTerm {
public:

Set::Scalar R; // Specific gas constant
Set::Scalar mu; // dynamic viscosity
Set::Scalar k_th; // thermal conductivity
static void Parse(SourceTerm& value, IO::ParmParse& pp)
{
pp.query_default("mu", value.mu, 0.0);
pp.query_default("k_th", value.k_th, 0.0);
pp.query_default("R", value.R, 287.0);
}


void ComputeShearStress(int lev, IntegratorDerivedClass* solver) const {

    const int nghosts = solver->number_of_ghost_cells;
    const Set::Scalar gamma = 1.4; // specific heat ratio for diatomic gases   

    IO::ParmParse pp;
    SourceTerm::Parse(*const_cast<SourceTerm*>(this), pp);

    const Set::Scalar dx = solver->geom[lev].CellSize()[0];  // Access the geometry at level `lev`
    const Set::Scalar dy = solver->geom[lev].CellSize()[1];
#if (AMREX_SPACEDIM == 3)
    const Set::Scalar dz = solver->geom[lev].CellSize()[2];
#endif	

    const Set::Scalar inU = solver->variableIndex.UVEL;
    const Set::Scalar inV = solver->variableIndex.VVEL;
#if (AMREX_SPACEDIM == 3)	
	const Set::Scalar inW = solver->variableIndex.WVEL;
#endif	
    const Set::Scalar inE = solver->variableIndex.IE;
    // Loop through the grid
    for (amrex::MFIter mfi(*solver->ShearStress_mf[lev], false); mfi.isValid(); ++mfi) {

        const amrex::Box& bx_with_ghosts = mfi.growntilebox(nghosts);
        //const amrex::Box& bx_with_ghosts = mfi.validbox();

        // Get arrays for pArr and shear stress output
        auto const& pArr = solver->PVec_mf.Patch(lev, mfi);  
        auto const& shearStressArr = solver->ShearStress_mf.Patch(lev, mfi);
        auto const& gradShearStressArr = solver->GradShearStress_mf.Patch(lev, mfi);

        // Parallel loop over cells
        amrex::ParallelFor(bx_with_ghosts, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
             
             
            // Finding pos-neg diagonal stresses on faces
            // "Xpos" means the face center on the positive (right) side of the cell along x-rection, etc.
            // These are the shear stress values on the cell faces from the linearly interpolated velocity gradients.

            // tao_xx 

            Set::Scalar tao_xx_Xpos = 2.0*mu*((pArr(i+1, j, k, inU)-pArr(i, j, k, inU))/dx) 
                                        - 2.0/3.0*mu*((pArr(i, j+1, k, inV)+pArr(i+1, j+1, k, inV)-
					pArr(i, j-1, k, inV)-pArr(i+1, j-1, k, inV))/(4.0*dy) + (pArr(i+1, j, k, inU)-pArr(i, j, k, inU))/dx);
#if (AMREX_SPACEDIM == 3)		
            tao_xx_Xpos = tao_xx_Xpos -2.0/3.0*mu*(pArr(i, j, k+1, inW)+pArr(i+1, j, k+1, inW)-
                                       pArr(i, j, k-1, inW)-pArr(i+1, j, k-1, inW))/(4.0*dz) ;
#endif				
						
            Set::Scalar tao_xx_Xneg = 2.0*mu*((pArr(i, j, k, inU)-pArr(i-1, j, k, inU))/dx) 
                                        - 2.0/3.0*mu*((pArr(i, j+1, k, inV)+pArr(i-1, j+1, k, inV)-
                                        pArr(i, j-1, k, inV)-pArr(i-1, j-1, k, inV))/(4.0*dy) + (pArr(i, j, k, inU)-pArr(i-1, j, k, inU))/dx);
#if (AMREX_SPACEDIM == 3)		
            tao_xx_Xneg = tao_xx_Xneg -2.0/3.0*mu*(pArr(i, j, k+1, inW)+pArr(i-1, j, k+1, inW)-
                                        pArr(i, j, k-1, inW)-pArr(i-1, j, k-1, inW))/(4.0*dz) ;
#endif
            // tao_yy
			
            Set::Scalar tao_yy_Ypos = 2.0*mu*((pArr(i, j+1, k, inV)-pArr(i, j, k, inV))/dy) 
                                        - 2.0/3.0*mu*((pArr(i+1, j, k, inU)+pArr(i+1, j+1, k, inU)-
                                        pArr(i-1, j, k, inU)-pArr(i-1, j+1, k, inU))/(4.0*dx) + (pArr(i, j+1, k, inV)-pArr(i, j, k, inV))/dy);
#if (AMREX_SPACEDIM == 3)		
            tao_yy_Ypos = tao_yy_Ypos -2.0/3.0*mu*(pArr(i, j, k+1, inW)+pArr(i, j+1, k+1, inW)-
										pArr(i, j, k-1, inW)-pArr(i, j+1, k-1, inW))/(4.0*dz) ;
#endif										
            Set::Scalar tao_yy_Yneg = 2.0*mu*((pArr(i, j, k, inV)-pArr(i, j-1, k, inV))/dy) 
                                        - 2.0/3.0*mu*((pArr(i+1, j, k, inU)+pArr(i+1, j-1, k, inU)-
                                        pArr(i-1, j, k, inU)-pArr(i-1, j-1, k, inU))/(4.0*dx) + (pArr(i, j, k, inV)-pArr(i, j-1, k, inV))/dy);
#if (AMREX_SPACEDIM == 3)		
            tao_yy_Yneg = tao_yy_Yneg -2.0/3.0*mu*(pArr(i, j, k+1, inW)+pArr(i, j-1, k+1, inW)-
                                       pArr(i, j, k-1, inW)-pArr(i, j-1, k-1, inW))/(4.0*dz) ;
#endif	
            // tao_zz

            Set::Scalar tao_zz_Zpos = 0.0;
            Set::Scalar tao_zz_Zneg = 0.0;		
#if (AMREX_SPACEDIM == 3)										
            tao_zz_Zpos = 2.0*mu*((pArr(i, j, k+1, inW)-pArr(i, j, k, inW))/dz) 
                                         - 2.0/3.0*mu*((pArr(i+1, j, k, inU)+pArr(i+1, j, k+1, inU)-
                                         pArr(i-1, j, k, inU)-pArr(i-1, j, k+1, inU))/(4.0*dx) + 
                                         (pArr(i, j+1, k, inV)+pArr(i, j+1, k+1, inV)-
                                         pArr(i, j-1, k, inV)-pArr(i, j-1, k+1, inV))/(4.0*dy) +
                                         (pArr(i, j, k+1, inW)-pArr(i, j, k, inW))/dz);
										
            tao_zz_Zneg = 2.0*mu*((pArr(i, j, k, inW)-pArr(i, j, k-1, inW))/dz) 
                                         - 2.0/3.0*mu*((pArr(i+1, j, k, inU)+pArr(i+1, j, k-1, inU)-
                                         pArr(i-1, j, k, inU)-pArr(i-1, j, k-1, inU))/(4.0*dx) + 
                                         (pArr(i, j+1, k, inV)+pArr(i, j+1, k-1, inV)-
                                         pArr(i, j-1, k, inV)-pArr(i, j-1, k-1, inV))/(4.0*dy) +
                                         (pArr(i, j, k, inW)-pArr(i, j, k-1, inW))/dz);
#endif
		    
            // Finding off-diagonal terms one after another:

            // tao_xy
            Set::Scalar tao_xy_Xpos = mu*((pArr(i, j+1, k, inU)+pArr(i+1, j+1, k, inU)-
                                        pArr(i, j-1, k, inU)-pArr(i+1, j-1, k, inU))/(4.0*dy) +
                                        (pArr(i+1, j, k, inV)-pArr(i, j, k, inV))/dx);
										
	    Set::Scalar tao_xy_Xneg = mu*((pArr(i, j+1, k, inU)+pArr(i-1, j+1, k, inU)-
                                        pArr(i, j-1, k, inU)-pArr(i-1, j-1, k, inU))/(4.0*dy) +
                                        (pArr(i, j, k, inV)-pArr(i-1, j, k, inV))/dx);
										
            Set::Scalar tao_xy_Ypos = mu*((pArr(i+1, j, k, inV)+pArr(i+1, j+1, k, inV)-
                                        pArr(i-1, j, k, inV)-pArr(i-1, j+1, k, inV))/(4.0*dx) +
                                        (pArr(i, j+1, k, inU)-pArr(i, j, k, inU))/dy);
            
            Set::Scalar tao_xy_Yneg = mu*((pArr(i+1, j, k, inV)+pArr(i+1, j-1, k, inV)-
                                         pArr(i-1, j, k, inV)-pArr(i-1, j-1, k, inV))/(4.0*dx) +
                                         (pArr(i, j, k, inU)-pArr(i, j-1, k, inU))/dy);
										
	    // tao_yz

            Set::Scalar tao_yz_Ypos = 0.0;
            Set::Scalar tao_yz_Yneg = 0.0;
            Set::Scalar tao_yz_Zpos = 0.0;
            Set::Scalar tao_yz_Zneg = 0.0;

#if (AMREX_SPACEDIM == 3)				
            tao_yz_Ypos = mu*((pArr(i, j, k+1, inV)+pArr(i, j+1, k+1, inV)-
										pArr(i, j, k-1, inV)-pArr(i, j+1, k-1, inV))/(4.0*dz) +
										(pArr(i, j+1, k, inW)-pArr(i, j, k, inW))/dy);	

            tao_yz_Yneg = mu*((pArr(i, j, k+1, inV)+pArr(i, j-1, k+1, inV)-
										pArr(i, j, k-1, inV)-pArr(i, j-1, k-1, inV))/(4.0*dz) +
										(pArr(i, j, k, inW)-pArr(i, j-1, k, inW))/dy);

            tao_yz_Zpos = mu*((pArr(i, j+1, k, inW)+pArr(i, j+1, k+1, inW)-
										pArr(i, j-1, k, inW)-pArr(i, j-1, k+1, inW))/(4.0*dy) +
										(pArr(i, j, k+1, inV)-pArr(i, j, k, inV))/dz);
										
            tao_yz_Zneg = mu*((pArr(i, j+1, k, inW)+pArr(i, j+1, k-1, inW)-
										pArr(i, j-1, k, inW)-pArr(i, j-1, k-1, inW))/(4.0*dy) +
										(pArr(i, j, k, inV)-pArr(i, j, k-1, inV))/dz);	   
#endif						

            // tao_xz

            Set::Scalar tao_xz_Xpos = 0.0;
            Set::Scalar tao_xz_Xneg = 0.0;
            Set::Scalar tao_xz_Zpos = 0.0;
            Set::Scalar tao_xz_Zneg = 0.0;

#if (AMREX_SPACEDIM == 3)				
            tao_xz_Xpos = mu*((pArr(i, j, k+1, inU)+pArr(i+1, j, k+1, inU)-
										pArr(i, j, k-1, inU)-pArr(i+1, j, k-1, inU))/(4.0*dz) +
										(pArr(i+1, j, k, inW)-pArr(i, j, k, inW))/dx);

            tao_xz_Xneg = mu*((pArr(i, j, k+1, inU)+pArr(i-1, j, k+1, inU)-
										pArr(i, j, k-1, inU)-pArr(i-1, j, k-1, inU))/(4.0*dz) +
										(pArr(i, j, k, inW)-pArr(i-1, j, k, inW))/dx);	

            tao_xz_Zpos = mu*((pArr(i+1, j, k, inW)+pArr(i+1, j, k+1, inW)-
										pArr(i-1, j, k, inW)-pArr(i-1, j, k+1, inW))/(4.0*dx) +
										(pArr(i, j, k+1, inU)-pArr(i, j, k, inU))/dz);

            tao_xz_Zneg = mu*((pArr(i+1, j, k, inW)+pArr(i+1, j, k-1, inW)-
										pArr(i-1, j, k, inW)-pArr(i-1, j, k-1, inW))/(4.0*dx) +
										(pArr(i, j, k, inU)-pArr(i, j, k-1, inU))/dz);										
#endif
			
            // Finding gradients and divergence of shear stresses using the above relations

            Set::Scalar tao_xx_x = (tao_xx_Xpos - tao_xx_Xneg)/dx;
            Set::Scalar tao_yy_y = (tao_yy_Ypos - tao_yy_Yneg)/dy;
            Set::Scalar tao_zz_z = 0.0;
#if (AMREX_SPACEDIM == 3)			
            tao_zz_z = 	(tao_zz_Zpos - tao_zz_Zneg)/dz;	
#endif
            // -----
            Set::Scalar tao_xy_x = (tao_xy_Xpos - tao_xy_Xneg)/dx;	
            Set::Scalar tao_xy_y = (tao_xy_Ypos - tao_xy_Yneg)/dy;
			// -----
            Set::Scalar tao_yz_y = 0.0;
            Set::Scalar tao_yz_z = 0.0;
#if (AMREX_SPACEDIM == 3)			
            tao_yz_y =  (tao_yz_Ypos - tao_yz_Yneg)/dy;
            tao_yz_z =  (tao_yz_Zpos - tao_yz_Zneg)/dz;
#endif
            // -----
            Set::Scalar tao_xz_x = 0.0;
            Set::Scalar tao_xz_z = 0.0;
#if (AMREX_SPACEDIM == 3)			
            tao_xz_x =  (tao_xz_Xpos - tao_xz_Xneg)/dx;
            tao_xz_z =  (tao_xz_Zpos - tao_xz_Zneg)/dz;
#endif

            // Assigning values to Shear Stress MultiFAB (Voight Notation)            
            //
            shearStressArr(i,j,k,0) = 0.5*(tao_xx_Xpos + tao_xx_Xneg);   // tao_XX
            shearStressArr(i,j,k,1) = 0.5*(tao_yy_Ypos + tao_yy_Yneg);   // tao_YY
            shearStressArr(i,j,k,2) = 0.5*(tao_zz_Zpos + tao_zz_Zneg);   // tao_ZZ

            shearStressArr(i,j,k,3) = 0.25*(tao_yz_Zpos + tao_yz_Zneg + tao_yz_Ypos + tao_yz_Yneg);   // tao_YZ
            shearStressArr(i,j,k,4) = 0.25*(tao_xz_Xpos + tao_xz_Xneg + tao_xz_Zpos + tao_xz_Zneg);   // tao_XZ
            shearStressArr(i,j,k,5) = 0.25*(tao_xy_Xpos + tao_xy_Xneg + tao_xy_Ypos + tao_xy_Yneg);   // tao_XY

            // Assigning values to Shear Stress Gradient MultiFAB (Row-Major Order)
            //
            gradShearStressArr(i,j,k,0) = tao_xx_x;
            gradShearStressArr(i,j,k,1) = tao_xy_y;
            gradShearStressArr(i,j,k,2) = tao_xz_z;
            gradShearStressArr(i,j,k,3) = tao_xy_x;
            gradShearStressArr(i,j,k,4) = tao_yy_y;
            gradShearStressArr(i,j,k,5) = tao_yz_z;
            gradShearStressArr(i,j,k,6) = tao_xz_x;
            gradShearStressArr(i,j,k,7) = tao_yz_y;
            gradShearStressArr(i,j,k,8) = tao_zz_z;

            }); // index loop end

      } // MFIter loop end

} // ComputeShearStress function end


void ComputeSourceTerm(int lev, IntegratorDerivedClass* solver) const {

    const int nghosts = solver->number_of_ghost_cells;
    const Set::Scalar gamma = 1.4; // specific heat ratio for diatomic gases   

    IO::ParmParse pp;
    SourceTerm::Parse(*const_cast<SourceTerm*>(this), pp);

    const Set::Scalar dx = solver->geom[lev].CellSize()[0];  // Access the geometry at level `lev`
    const Set::Scalar dy = solver->geom[lev].CellSize()[1];
#if (AMREX_SPACEDIM == 3)
    const Set::Scalar dz = solver->geom[lev].CellSize()[2];
#endif	

    const Set::Scalar inU = solver->variableIndex.UVEL;
    const Set::Scalar inV = solver->variableIndex.VVEL;
#if (AMREX_SPACEDIM == 3)	
	const Set::Scalar inW = solver->variableIndex.WVEL;
#endif	
    const Set::Scalar inE = solver->variableIndex.IE;
    // Loop through the grid
    for (amrex::MFIter mfi(*solver->SourceTermVec[lev], false); mfi.isValid(); ++mfi) {

        const amrex::Box& bx_with_ghosts = mfi.growntilebox(nghosts);
        //const amrex::Box& bx_with_ghosts = mfi.validbox();

        // Get arrays for pArr and shear stress output
        auto const& pArr = solver->PVec_mf.Patch(lev, mfi);  
        auto const& shearStressArr = solver->ShearStress_mf.Patch(lev, mfi);
        auto const& gradShearStressArr = solver->GradShearStress_mf.Patch(lev, mfi);
        auto const& sourceTermArr = solver->SourceTermVec.Patch(lev, mfi);

        // Parallel loop over cells
        amrex::ParallelFor(bx_with_ghosts, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

            
            // Assigning values to source term multiFab
			
            sourceTermArr(i,j,k,0) = 0.0;
			
            sourceTermArr(i,j,k,1) = gradShearStressArr(i,j,k,0) + gradShearStressArr(i,j,k,1) + gradShearStressArr(i,j,k,2);

            sourceTermArr(i,j,k,2) = gradShearStressArr(i,j,k,3) + gradShearStressArr(i,j,k,4) + gradShearStressArr(i,j,k,5);
		
            sourceTermArr(i,j,k,3) = gradShearStressArr(i,j,k,6) + gradShearStressArr(i,j,k,7) + gradShearStressArr(i,j,k,8);

            // The final energy term of the source term vector

            // Divide the total energy term into four parts, namely, S_E1, S_E2, S_E3 and S_E4 (S_E4 coming from the thermal contribution)
			
            Set::Scalar S_E1 = shearStressArr(i,j,k,0)*(pArr(i+1, j, k, inU)-pArr(i-1, j, k, inU))/(2.0*dx) + 
                                     pArr(i, j, k, inU)*gradShearStressArr(i,j,k,0) + 
#if (AMREX_SPACEDIM >= 2)
                               shearStressArr(i,j,k,5)*(pArr(i+1, j, k, inV)-pArr(i-1, j, k, inV))/(2.0*dx) +
                                     pArr(i, j, k, inV)*gradShearStressArr(i,j,k,3) +
#endif
#if (AMREX_SPACEDIM == 3)
                               shearStressArr(i,j,k,4)*(pArr(i+1, j, k, inW)-pArr(i-1, j, k, inW))/(2.0*dx) +
                                     pArr(i, j, k, inW)*gradShearStressArr(i,j,k,6)
#endif
                               +0.0;


            // -----

            Set::Scalar S_E2 = 0.0;
#if (AMREX_SPACEDIM >= 2)            
            S_E2 = shearStressArr(i,j,k,5)*(pArr(i, j+1, k, inU)-pArr(i, j-1, k, inU))/(2.0*dy) +
                                     pArr(i, j, k, inU)*gradShearStressArr(i,j,k,1) +
                               
                               shearStressArr(i,j,k,1)*(pArr(i, j+1, k, inV)-pArr(i, j-1, k, inV))/(2.0*dy) +
                                     pArr(i, j, k, inV)*gradShearStressArr(i,j,k,4) +

#if (AMREX_SPACEDIM == 3)
                               shearStressArr(i,j,k,3)*(pArr(i, j+1, k, inW)-pArr(i, j-1, k, inW))/(2.0*dy) +
                                     pArr(i, j, k, inW)*gradShearStressArr(i,j,k,7)
#endif
                               +0.0;
#endif


            // -----
			
            Set::Scalar S_E3 = 0.0;
#if (AMREX_SPACEDIM == 3)			
            S_E3 = shearStressArr(i,j,k,4)*(pArr(i, j, k+1, inU)-pArr(i, j, k-1, inU))/(2.0*dz) +
                                     pArr(i, j, k, inU)*gradShearStressArr(i,j,k,2) +

                               shearStressArr(i,j,k,3)*(pArr(i, j, k+1, inV)-pArr(i, j, k-1, inV))/(2.0*dz) +
                                     pArr(i, j, k, inV)*gradShearStressArr(i,j,k,5) +

                               shearStressArr(i,j,k,2)*(pArr(i, j, k+1, inW)-pArr(i, j, k-1, inW))/(2.0*dz) +
                                     pArr(i, j, k, inW)*gradShearStressArr(i,j,k,8);
#endif

            // ----- Calculation of Temperature and Temperature Gradients for the energy source term

            Set::Scalar S_E4 = 0.0;
            
            // finding double gradients of temperature
            Set::Scalar T_XX = (gamma-1.0)/R * (pArr(i+1,j,k,inE) - 2.0*pArr(i,j,k,inE) + pArr(i-1,j,k,inE))/(dx*dx);
            Set::Scalar T_YY = 0.0;
#if (AMREX_SPACEDIM >= 2)
            T_YY = (gamma-1.0)/R * (pArr(i,j+1,k,inE) - 2.0*pArr(i,j,k,inE) + pArr(i,j-1,k,inE))/(dy*dy);
#endif
            Set::Scalar T_ZZ = 0.0;
#if (AMREX_SPACEDIM == 3)
            T_ZZ = (gamma-1.0)/R * (pArr(i,j,k+1,inE) - 2.0*pArr(i,j,k,inE) + pArr(i,j,k-1,inE))/(dz*dz);
#endif

            S_E4 = k_th*(T_XX + T_YY + T_ZZ); // k_th is thermal conductivity read from the input

            // finally Add all of them to find final form of the source term
			
            sourceTermArr(i,j,k,4) = S_E1 + S_E2 + S_E3 + S_E4;

	
            }); // index loop end
		
      } // MFIter loop end
	
} // ComputeSourceTerm function end

}; // end of class

} // end of Namespace

#endif	
