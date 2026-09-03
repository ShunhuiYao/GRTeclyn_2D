/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#if !defined(BOOSTEDBHINITIALDATACARTOON_HPP_)
#error "This file should only be included through BoostedBHInitialDataCartoon.hpp"
#endif

#ifndef BOOSTEDBHINITIALDATACARTOON_IMPL_HPP_
#define BOOSTEDBHINITIALDATACARTOON_IMPL_HPP_

#include "DimensionDefinitions.hpp"
#include <cmath>
#include <string>

void BoostedBHInitialDataCartoon::params_t::check_params(int a_id)
{
    GRParmParse bh_pp("bh" + std::to_string(a_id));

    amrex::Real mass;
    bh_pp.get("mass", mass);
    if (mass <= 0.0)
    {
        bh_pp.error("mass", "must be > 0");
    }

    amrex::Real momentum;
    bh_pp.get("momentum", momentum);
    if (std::sqrt(std::abs(momentum)) >= 0.3 * mass)
    {
        bh_pp.warning("momentum", "approximation used for boosted BH is only "
                                  "valid when |P| / mass is small");
    }

    GRParmParse geom_pp("geometry");
    std::array<amrex::Real, AMREX_SPACEDIM> center{};
    geom_pp.get("center", center);
    std::array<amrex::Real, AMREX_SPACEDIM> prob_extent{};
    geom_pp.get("prob_extent", prob_extent);

    amrex::Real offset{};
    bh_pp.queryAdd("offset", offset);
    amrex::Real bh_center=center[0]+offset;

    if (bh_center < 0.0 || bh_center > prob_extent[0])
    {
        bh_pp.warning("offset", "places the black hole outside the "
                                "computational domain");
    }
}

void BoostedBHInitialDataCartoon::params_t::fill_params()
{
    GRParmParse bh_pp("bh" + std::to_string(id));
    GRParmParse geom_pp("geometry");
    bh_pp.get("mass", mass);
    bh_pp.get("momentum", momentum);

    std::array<amrex::Real, AMREX_SPACEDIM> geo_center{};
    geom_pp.get("center", geo_center);
    amrex::Real offset{};
    bh_pp.get("offset", offset);
    center=geo_center[0]+offset;

}

AMREX_FORCE_INLINE BoostedBHInitialDataCartoon::BoostedBHInitialDataCartoon(int a_id)
    : m_params(a_id)
{
    m_params.fill_params();
}

[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
BoostedBHInitialDataCartoon::psi_minus_one(Coordinates coords) const
{
    const amrex::Real r         = center_dist(coords);
    const amrex::Real cos_theta = (coords.x - m_params.center) / r;
    const amrex::Real P_squared = pow(m_params.momentum,2);
    return psi0(r) +
           P_squared * psi2(r, cos_theta) / (m_params.mass * m_params.mass);
}

[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE Tensor::Rank2
BoostedBHInitialDataCartoon::Aij(Coordinates a_coords) const
{
    const amrex::Real r = center_dist(a_coords);
    const Tensor::Rank1 l{(a_coords.x - m_params.center) / r,
                           a_coords.y / r,
                           a_coords.z / r};
    const amrex::Real l_dot_p = l(0) * m_params.momentum;

    Tensor::Rank2 out;
    Tensor::Rank1 momentum_3D{};
    momentum_3D(0)=m_params.momentum;

    FOR (i, j)
    {
        const amrex::Real delta = (i == j) ? 1 : 0;
        out(i, j)               = 1.5 *
                    (momentum_3D(i) * l(j) + momentum_3D(j) * l(i) -
                     (delta - l(i) * l(j)) * l_dot_p) /
                    (r * r);
    }
    return out;
}

/* PRIVATE */

[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
BoostedBHInitialDataCartoon::center_dist(Coordinates a_coords) const
{
    amrex::Real r = std::sqrt(std::pow(a_coords.x - m_params.center, 2) +
                              std::pow(a_coords.y, 2));

    return std::max(r, 1e-6);
}

[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
BoostedBHInitialDataCartoon::psi0(amrex::Real a_r) const
{
    return m_params.mass / (2 * a_r);
}

[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
BoostedBHInitialDataCartoon::psi2(amrex::Real a_r, amrex::Real a_cos_theta) const
{
    return psi2_0(a_r) + psi2_2(a_r) * (1.5 * a_cos_theta * a_cos_theta - 0.5);
}

[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
BoostedBHInitialDataCartoon::psi2_0(amrex::Real a_r) const
{
    const amrex::Real psi0_here    = psi0(a_r);
    const amrex::Real psi0_sq_here = psi0_here * psi0_here;
    return std::pow(1 + psi0_here, -5) * (psi0_here / 8) *
           (psi0_sq_here * psi0_sq_here + 5 * psi0_here * psi0_sq_here +
            10 * psi0_sq_here + 10 * psi0_here + 5);
}

[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
BoostedBHInitialDataCartoon::psi2_2(amrex::Real a_r) const
{
    const amrex::Real psi0_here    = psi0(a_r);
    const amrex::Real psi0_sq_here = psi0_here * psi0_here;
    return 0.05 * std::pow(1 + psi0_here, -5) * psi0_sq_here *
               (84 * psi0_here * psi0_sq_here * psi0_sq_here +
                378 * psi0_sq_here * psi0_sq_here +
                658 * psi0_here * psi0_sq_here + 539 * psi0_sq_here +
                192 * psi0_here + 15) +
           4.2 * psi0_here * psi0_sq_here * log(psi0_here / (1 + psi0_here));
}

#endif /* BOOSTEDBHINITIALDATACARTOON_IMPL_HPP_ */
