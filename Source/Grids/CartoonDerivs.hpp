/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef CARTOONDERIVS_HPP_
#define CARTOONDERIVS_HPP_

#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"

/// Generic Cartoon derivative fills for axisymmetric 2D runs
/// (DEFAULT_TENSOR_DIM=3, CH_SPACEDIM=2).
/**
 * Fills the z-direction (tensor index 2) components of first and second
 * derivatives from the field values and their in-plane derivatives, using
 * the Cartoon conditions that follow from L_{d/dphi} T = 0 at z = 0, where
 * d/dphi = -z d/dy + y d/dz is the axisymmetry Killing vector.
 *
 * Convention: index 0 = x (symmetry axis), index 1 = y (cylindrical radius),
 *             index 2 = z (suppressed). Cartoon coordinate rho = y = coords.y.
 *
 * Overloaded on the rank of the variable (scalar / contravariant vector /
 * symmetric covariant tensor), so callers just dispatch each Vars member to
 * the matching overload instead of repeating the algebra per-field. All
 * components left untouched here are zero (from FORSPACEDIM
 * zero-initialisation in FourthOrderDerivatives), the correct value.
 *
 **/

namespace CartoonDerivs
{
// d/dz of a contravariant vector V^i (shift, B, Gamma):
// d/dz V^y = -V^z/y ; d/dz V^z = V^y/y ; d/dz V^x = 0
void fill_cartoon_derivs_d1_vector(const Tensor::Rank1 &V, 
                                   const double one_over_y,Tensor::Rank2 &d1_V)
{
    d1_V(1, 2) = -V(2) * one_over_y;
    d1_V(2, 2) = V(1) * one_over_y;
}

// d/dz of a symmetric covariant tensor T_{ij} (h, A):
// d/dz T_{xy} = -T_{xz}/y ; d/dz T_{xz} = T_{xy}/y
// d/dz T_{yy} = -2 T_{yz}/y ; d/dz T_{zz} = 2 T_{yz}/y
// d/dz T_{yz} = (T_{yy}-T_{zz})/y ; d/dz T_{xx} = 0
void fill_cartoon_derivs_d1_sym_tensor(const Tensor::Sym12Rank2 &T,
                                       const double one_over_y,
                                       Tensor::Sym12Rank3 &d1_T)
{
    // ordering below 0: T11, 1: T12, 2: T13, 3: T22, 4: T23, 5: T33
    d1_T(1, 2) = -T(2) * one_over_y;
    d1_T(2, 2) = T(1) * one_over_y;
    d1_T(3, 2) = -2. * T(4) * one_over_y;
    d1_T(4, 2) = (T(3) - T(5)) * one_over_y;
    d1_T(5, 2) = 2. * T(4) * one_over_y;
}

void fill_cartoon_derivs_d1_tensor(const Tensor::Rank2 &T, 
                                   const double one_over_y,
                                   Tensor::Rank3 &d1_T)
{
    d1_T(1, 0, 2) = -T(0, 2) * one_over_y;
    d1_T(0, 1, 2) = d1_T(1, 0, 2);

    d1_T(0, 2, 2) = T(0, 1) * one_over_y;
    d1_T(2, 0, 2) = d1_T(0, 2, 2);

    d1_T(1, 1, 2) = -2. * T(1, 2) * one_over_y;
    d1_T(2, 2, 2) = 2. * T(1, 2) * one_over_y;

    d1_T(1, 2, 2) = (T(1, 1) - T(2, 2)) * one_over_y;
    d1_T(2, 1, 2) = d1_T(1, 2, 2);

} 

// d/dz d/dz of a scalar S (chi, lapse)
void fill_cartoon_derivs_d2_scalar(const double dy_S, 
                                   const double one_over_y, 
                                   Tensor::Rank2 &d2_S)
{
    d2_S(2, 2) = dy_S * one_over_y;
}

// d/dk d/dz and d/dz d/dz of a contravariant vector V^i (shift, Gamma).
void fill_cartoon_derivs_d2_vector(const Tensor::Rank1 &V,
                                   const Tensor::Rank2 &d1_V,
                                   const double one_over_y,
                                   const double one_over_y2,
                                   Tensor::Sym23Rank3 &d2_V)
{
    FORSPACEDIM(k)
    {
        d2_V(2, k, 2) = d1_V(1, k) * one_over_y;
        d2_V(1, k, 2) = -d1_V(2, k) * one_over_y;
    }
    // 1/y^2 correction from differentiating 1/y in the y-direction (k=1).
    d2_V(2, 1, 2) -= V(1) * one_over_y2;
    d2_V(1, 1, 2) += V(2) * one_over_y2;

    FORSPACEDIM(k)
    {
        d2_V(2, 2, k) = d2_V(2, k, 2);
        d2_V(1, 2, k) = d2_V(1, k, 2);
    }

    d2_V(0, 2, 2) = d1_V(0, 1) * one_over_y;
    d2_V(1, 2, 2) = d1_V(1, 1) * one_over_y - V(1) * one_over_y2;
    d2_V(2, 2, 2) = d1_V(2, 1) * one_over_y - V(2) * one_over_y2;
}

// d/dk d/dz and d/dz d/dz of a  covariant tensor T_{ij} (h, A).
void fill_cartoon_derivs_d2_tensor(const Tensor::Rank2 &T, 
                                   const Tensor::Rank3 &d1_T,
                                   const double one_over_y,
                                   const double one_over_y2,
                                   Tensor::Sym34Rank4 &d2_T)
{
    FORSPACEDIM(k)
    {
        d2_T(1, 2, k, 2) = (d1_T(1, 1, k) - d1_T(2, 2, k)) * one_over_y;
        d2_T(2, 1, k, 2) = d2_T(1, 2, k, 2);

        d2_T(0, 2, k, 2) = d1_T(0, 1, k) * one_over_y;
        d2_T(2, 0, k, 2) = d2_T(0, 2, k, 2);

        d2_T(1, 0, k, 2) = -d1_T(0, 2, k) * one_over_y;
        d2_T(0, 1, k, 2) = d2_T(1, 0, k, 2);

        d2_T(1, 1, k, 2) = -2. * d1_T(1, 2, k) * one_over_y;
        d2_T(2, 2, k, 2) = 2. * d1_T(1, 2, k) * one_over_y;
    }
    // 1/y^2 corrections from differentiating 1/y in the y-direction (k=1).
    d2_T(1, 2, 1, 2) -= (T(1, 1) - T(2, 2)) * one_over_y2;
    d2_T(2, 1, 1, 2) = d2_T(1, 2, 1, 2);

    d2_T(0, 2, 1, 2) -= T(0, 1) * one_over_y2;
    d2_T(2, 0 ,1 ,2) = d2_T(0 ,2 ,1 ,2);

    d2_T(1, 0, 1, 2) += T(0, 2) * one_over_y2;
    d2_T(0, 1, 1, 2)  = d2_T(1, 0, 1, 2);

    d2_T(1, 1, 1, 2) += 2. * T(1, 2) * one_over_y2;
    d2_T(2, 2, 1, 2) -= 2. * T(1, 2) * one_over_y2;

    d2_T(0, 0, 2, 2) = d1_T(0, 0, 1) * one_over_y;
    d2_T(1, 1, 2, 2) = d1_T(1, 1, 1) * one_over_y - 2. * (T(1, 1) - T(2, 2)) * one_over_y2;
    d2_T(2, 2, 2, 2) = d1_T(2, 2, 1) * one_over_y + 2. * (T(1, 1) - T(2, 2)) * one_over_y2;

    d2_T(1, 0, 2, 2) = d1_T(1, 0, 1) * one_over_y - T(1, 0) * one_over_y2;
    d2_T(0, 1, 2, 2) = d2_T(1, 0, 2, 2);

    d2_T(0, 2, 2, 2) = d1_T(0, 2, 1) * one_over_y - T(0, 2) * one_over_y2;
    d2_T(2, 0, 2, 2) = d2_T(0, 2, 2, 2);

    d2_T(1, 2, 2, 2) = d1_T(1, 2, 1) * one_over_y - 4. * T(1, 2) * one_over_y2;
    d2_T(2, 1, 2, 2) = d2_T(1, 2, 2, 2);


} 

void fill_cartoon_derivs_d2_sym_tensor(const Tensor::Sym12Rank2 &T, 
                                   const Tensor::Sym12Rank3 &d1_T,
                                   const double one_over_y,
                                   const double one_over_y2,
                                   Tensor::Sym12Sym34Rank4 &d2_T)
{
    FORSPACEDIM(k)
    {
        d2_T(1, 2, k, 2) = (d1_T(1, 1, k) - d1_T(2, 2, k)) * one_over_y;

        d2_T(0, 2, k, 2) = d1_T(0, 1, k) * one_over_y;

        d2_T(1, 0, k, 2) = -d1_T(0, 2, k) * one_over_y;

        d2_T(1, 1, k, 2) = -2. * d1_T(1, 2, k) * one_over_y;
        d2_T(2, 2, k, 2) = 2. * d1_T(1, 2, k) * one_over_y;
    }
    // 1/y^2 corrections from differentiating 1/y in the y-direction (k=1).
    d2_T(1, 2, 1, 2) -= (T(1, 1) - T(2, 2)) * one_over_y2;

    d2_T(0, 2, 1, 2) -= T(0, 1) * one_over_y2;

    d2_T(0, 1, 1, 2) += T(0, 2) * one_over_y2;

    d2_T(1, 1, 1, 2) += 2. * T(1, 2) * one_over_y2;
    d2_T(2, 2, 1, 2) -= 2. * T(1, 2) * one_over_y2;

    d2_T(0, 0, 2, 2) = d1_T(0, 0, 1) * one_over_y;
    d2_T(1, 1, 2, 2) = d1_T(1, 1, 1) * one_over_y - 2. * (T(1, 1) - T(2, 2)) * one_over_y2;
    d2_T(2, 2, 2, 2) = d1_T(2, 2, 1) * one_over_y + 2. * (T(1, 1) - T(2, 2)) * one_over_y2;

    d2_T(0, 1, 2, 2) = d1_T(1, 0, 1) * one_over_y - T(1, 0) * one_over_y2;

    d2_T(0, 2, 2, 2) = d1_T(0, 2, 1) * one_over_y - T(0, 2) * one_over_y2;

    d2_T(1, 2, 2, 2) = d1_T(1, 2, 1) * one_over_y - 4. * T(1, 2) * one_over_y2;

}

}// namespace CartoonDerivs

 #endif /* CARTOONDERIVS_HPP_ */