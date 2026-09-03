/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef BOOSTEDBHINITIALDATACARTOON_HPP_
#define BOOSTEDBHINITIALDATACARTOON_HPP_
/**
 * BOOSTED SCHWARZSCHILD BLACK HOLE
 * Baumgarte & Shapiro, pp. 73-74
 * NB: \bar{A} as defined in the book is psi^{-6} * \bar{A}_{BSSN}
 */

#include "Coordinates.hpp"
#include "Tensor.hpp"
#include <array>

class BoostedBHInitialDataCartoon
{

  public:
    struct params_t
    {
        amrex::Real mass{};
        amrex::Real center{};
        amrex::Real momentum{};
        int id{};

        AMREX_FORCE_INLINE params_t(int a_id) : id(a_id) {};

        inline static void check_params(int a_id);

        inline void fill_params();
    };

    AMREX_FORCE_INLINE BoostedBHInitialDataCartoon(int a_id);

    // conformal factor
    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
    psi_minus_one(Coordinates a_coords) const;

    // extrinsic curvature
    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE Tensor::Rank2
    Aij(Coordinates a_coords) const;

  private:
    params_t m_params;

    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
    center_dist(Coordinates a_coords) const;

    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
    psi0(amrex::Real a_r) const;

    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
    psi2(amrex::Real a_r, amrex::Real a_cos_theta) const;

    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
    psi2_0(amrex::Real a_r) const;

    [[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
    psi2_2(amrex::Real a_r) const;
};

#include "BoostedBHInitialDataCartoon.impl.hpp"

#endif /*BOOSTEDBHINITIALDATACARTOON_HPP_*/
