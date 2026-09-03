/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef BOOSTEDHEADONCARTOONINITIALDATA_HPP_
#define BOOSTEDHEADONCARTOONINITIALDATA_HPP_

#include "BoostedBHInitialDataCartoon.hpp"
#include "Coordinates.hpp"
#include "StateVariables.hpp" //This files needs NUM_VARS - total number of components
#include "Tensor.hpp"
#include <array>

enum Lapse
{
    ONE,
    PRE_COLLAPSED,
    CHI
};

class BoostedHeadOnCartoonInitialData
{
  protected:
    amrex::Real m_dx;
    BoostedBHInitialDataCartoon bh1;
    BoostedBHInitialDataCartoon bh2;
    int m_initial_lapse;

  public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_FORCE_INLINE
    BoostedHeadOnCartoonInitialData(amrex::Real a_dx,
                        int a_initial_lapse = Lapse::PRE_COLLAPSED);
    // NOLINTEND(bugprone-easily-swappable-parameters)

    AMREX_GPU_DEVICE void
    operator()(int ix, int iy, int iz,
               const amrex::Array4<amrex::Real> &state) const;

  protected:
    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
    compute_chi(Coordinates coords) const;

    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE Tensor::Rank2
    compute_A(amrex::Real chi, Coordinates coords) const;
};

#include "BoostedHeadOnCartoonInitialData.impl.hpp"

#endif /*BOOSTEDHEADONCARTOONINITIALDATA_HPP_ */
