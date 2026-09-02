/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef BCPARITY_HPP_
#define BCPARITY_HPP_

#include <AMReX_IntVect.H>
#include <map>

//! A scoped enumerator for the parity of a variable wrt reflective boundaries
enum class BCParity
{
    undefined,
    even,
    odd_x,
    odd_y,
    odd_z,
    odd_xy,
    odd_yz,
    odd_xz,
    odd_xyz
};

static inline const std::map<BCParity, amrex::IntVect> bc_parity_map = {
    {BCParity::even, amrex::IntVect(1)},
    {BCParity::odd_x, amrex::IntVect(AMREX_D_DECL(-1, 1, 1))},
    {BCParity::odd_y, amrex::IntVect(AMREX_D_DECL(1, -1, 1))},
    {BCParity::odd_z, amrex::IntVect(AMREX_D_DECL(1, 1, -1))},
    {BCParity::odd_xy, amrex::IntVect(AMREX_D_DECL(-1, -1, 1))},
    {BCParity::odd_yz, amrex::IntVect(AMREX_D_DECL(1, -1, -1))},
    {BCParity::odd_xz, amrex::IntVect(AMREX_D_DECL(-1, 1, -1))},
    {BCParity::odd_xyz, amrex::IntVect(AMREX_D_DECL(-1, -1, -1))},
};

#endif