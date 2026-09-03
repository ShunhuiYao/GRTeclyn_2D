/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef FOURTHORDERDERIVATIVES_HPP_
#define FOURTHORDERDERIVATIVES_HPP_

#include "DimensionDefinitions.hpp"

#include "DerivativeBase.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"
#include <AMReX_REAL.H>
#include <array>

#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
#include "CartoonDerivs.hpp"
#endif

using namespace amrex::literals;

class FourthOrderDerivatives : protected DerivativeBase
{
  public:
    AMREX_GPU_HOST_DEVICE
    FourthOrderDerivatives(amrex::Real dx) : DerivativeBase(dx){}

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    diff1(const amrex::Real *in_ptr, const int stride, const int idx = 0) const
    {
        amrex::Real weight_far  = 8.33333333333333333333e-2_rt;
        amrex::Real weight_near = 6.66666666666666666667e-1_rt;

        return (weight_far * in_ptr[idx - 2 * stride] -
                weight_near * in_ptr[idx - stride] +
                weight_near * in_ptr[idx + stride] -
                weight_far * in_ptr[idx + 2 * stride]) *
               m_one_over_dx;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Rank1
    d1_scalar(int ix, int iy, int iz,
              const amrex::Array4<const amrex::Real> &state,
              const int ivar) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        Tensor::Rank1 d1;
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        const auto strides        = get_strides(state);
        const auto *var_ptr       = get_var_ptr(ivar, state_ptr_xyz, strides);
        FORSPACEDIM (idir)
        {
            d1(idir) = diff1(var_ptr, strides[idir]);
        }
#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        // Fill cartoon derivatives
        // For a scalar, the d/dz derivative is zero
        d1(2) = 0.0;
#endif
        return d1;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Rank2
    d1_vector(int ix, int iy, int iz,
              const amrex::Array4<const amrex::Real> &state,
              const int ivar_0) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        Tensor::Rank2 d1;
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        const auto strides        = get_strides(state);

        Tensor::Rank1 V;
        FOR (icomp)
        {
            const int ivar      = ivar_0 + icomp;
            const auto *var_ptr = get_var_ptr(ivar, state_ptr_xyz, strides);
            V(icomp) = *var_ptr;
            FORSPACEDIM (idir)
            {
                d1(icomp, idir) = diff1(var_ptr, strides[idir]);
            }
        }

#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        // Fill cartoon derivatives
        const amrex::Real y = (iy + 0.5) * m_dx;
        const amrex::Real one_over_y = 1.0 / y;
        CartoonDerivs::fill_cartoon_derivs_d1_vector(V, one_over_y, d1);
#endif
        return d1;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Sym12Rank3
    d1_sym_tensor(int ix, int iy, int iz,
                  const amrex::Array4<const amrex::Real> &state,
                  const int ivar_0) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        Tensor::Sym12Rank3 d1{};
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        const auto strides        = get_strides(state);

        Tensor::Sym12Rank2 T{};
        for (int ivar = 0; ivar < NUM_SYM_IDXS; ++ivar)
        {
            const auto *var_ptr =
                get_var_ptr(ivar_0 + ivar, state_ptr_xyz, strides);
            T(ivar) = *var_ptr;
            FORSPACEDIM (idir)
            {
                d1(ivar, idir) = diff1(var_ptr, strides[idir]);
            }
        }
#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        // Fill cartoon derivatives
        const amrex::Real y = (iy + 0.5) * m_dx;
        const amrex::Real one_over_y = 1.0 / y;
        CartoonDerivs::fill_cartoon_derivs_d1_sym_tensor(T, one_over_y, d1);
#endif
        return d1;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Rank3
    d1_tensor(int ix, int iy, int iz,
              const amrex::Array4<const amrex::Real> &state,
              const int ivar_0) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        Tensor::Rank3 d1{};
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        const auto strides        = get_strides(state);

        Tensor::Rank2 T{};
        int ivar{ivar_0};
        FOR (icomp, jcomp)
        {
            const auto *var_ptr = get_var_ptr(ivar, state_ptr_xyz, strides);
            T(icomp, jcomp) = *var_ptr;
            FORSPACEDIM (idir)
            {
                d1(icomp, jcomp, idir) = diff1(var_ptr, strides[idir]);
            }
            ++ivar;
        }

#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        // Fill cartoon derivatives
        const amrex::Real y = (iy + 0.5) * m_dx;
        const amrex::Real one_over_y = 1.0 / y;
        CartoonDerivs::fill_cartoon_derivs_d1_tensor(T, one_over_y, d1);
#endif
        return d1;
    }

    // gets the derivative of a consecutive series of vars in a state
    template <int num_diff_vars>
    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        amrex::Array2D<amrex::Real, 0, num_diff_vars - 1, 0, AMREX_SPACEDIM - 1>
        diff1_state(int ix, int iy, int iz,
                    const amrex::Array4<const amrex::Real> &state,
                    int first_var = 0) const
    {
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        amrex::GpuArray<int, AMREX_SPACEDIM> strides{AMREX_D_DECL(
            1, static_cast<int>(state.stride.a[0]),
            static_cast<int>(state.stride.a[1]))};

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        amrex::Array2D<amrex::Real, 0, num_diff_vars - 1, 0, AMREX_SPACEDIM - 1>
            d1_state{};

        for (int ivar = first_var; ivar < (first_var + num_diff_vars); ++ivar)
        {
            const auto *var_ptr = state_ptr_xyz + ivar * state.stride.a[2];

            FORSPACEDIM(idir)
            {
                d1_state(ivar - first_var, idir) =
                    diff1(var_ptr, strides[idir]);
            }
        }
        return d1_state;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    diff2(const amrex::Real *in_ptr, const int stride, const int idx = 0) const
    {
        amrex::Real weight_far   = 8.33333333333333333333e-2_rt;
        amrex::Real weight_near  = 1.33333333333333333333e+0_rt;
        amrex::Real weight_local = 2.50000000000000000000e+0_rt;

        return (-weight_far * in_ptr[idx - 2 * stride] +
                weight_near * in_ptr[idx - stride] -
                weight_local * in_ptr[idx] +
                weight_near * in_ptr[idx + stride] -
                weight_far * in_ptr[idx + 2 * stride]) *
               m_one_over_dx2;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    mixed_diff2(const amrex::Real *in_ptr, const int stride1, const int stride2,
                const int idx = 0) const
    {
        amrex::Real weight_far_far   = 6.94444444444444444444e-3_rt;
        amrex::Real weight_near_far  = 5.55555555555555555556e-2_rt;
        amrex::Real weight_near_near = 4.44444444444444444444e-1_rt;

        return (weight_far_far * in_ptr[idx - 2 * stride1 - 2 * stride2] -
                weight_near_far * in_ptr[idx - 2 * stride1 - stride2] +
                weight_near_far * in_ptr[idx - 2 * stride1 + stride2] -
                weight_far_far * in_ptr[idx - 2 * stride1 + 2 * stride2]

                - weight_near_far * in_ptr[idx - stride1 - 2 * stride2] +
                weight_near_near * in_ptr[idx - stride1 - stride2] -
                weight_near_near * in_ptr[idx - stride1 + stride2] +
                weight_near_far * in_ptr[idx - stride1 + 2 * stride2]

                + weight_near_far * in_ptr[idx + stride1 - 2 * stride2] -
                weight_near_near * in_ptr[idx + stride1 - stride2] +
                weight_near_near * in_ptr[idx + stride1 + stride2] -
                weight_near_far * in_ptr[idx + stride1 + 2 * stride2]

                - weight_far_far * in_ptr[idx + 2 * stride1 - 2 * stride2] +
                weight_near_far * in_ptr[idx + 2 * stride1 - stride2] -
                weight_near_far * in_ptr[idx + 2 * stride1 + stride2] +
                weight_far_far * in_ptr[idx + 2 * stride1 + 2 * stride2]) *
               m_one_over_dx2;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Sym12Rank2
    d2_scalar(int ix, int iy, int iz,
              const amrex::Array4<const amrex::Real> &state,
              const int ivar) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        Tensor::Sym12Rank2 d2;

        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        auto strides              = get_strides(state);
        const auto *var_ptr       = get_var_ptr(ivar, state_ptr_xyz, strides);

        d2(0, 0) = diff2(var_ptr, strides[0]);
        d2(1, 1) = diff2(var_ptr, strides[1]);
        d2(0, 1) = mixed_diff2(var_ptr, strides[0], strides[1]);
#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM && AMREX_SPACEDIM == 3
        d2(2, 2) = diff2(var_ptr, strides[2]);
        d2(0, 2) = mixed_diff2(var_ptr, strides[0], strides[2]);
        d2(1, 2) = mixed_diff2(var_ptr, strides[1], strides[2]);
#elif DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        d2(0, 2) = 0.0;
        d2(1, 2) = 0.0;
        // Fill cartoon derivatives
        amrex:: Real dy_S;
        dy_S = diff1(var_ptr, strides[1]);
        const amrex::Real y = (iy + 0.5) * m_dx;
        const amrex::Real one_over_y = 1.0 / y;
        CartoonDerivs::fill_cartoon_derivs_d2_scalar( dy_S, one_over_y,d2);
#endif
        return d2;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Sym23Rank3
    d2_vector(int ix, int iy, int iz,
              const amrex::Array4<const amrex::Real> &state,
              const int ivar_0) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        Tensor::Sym23Rank3 d2;
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        auto strides              = get_strides(state);

#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM && AMREX_SPACEDIM == 3
        FOR (icomp)
        {
            const int ivar      = ivar_0 + icomp;
            const auto *var_ptr = get_var_ptr(ivar, state_ptr_xyz, strides);

            d2(icomp, 0, 0) = diff2(var_ptr, strides[0]);
            d2(icomp, 1, 1) = diff2(var_ptr, strides[1]);
            d2(icomp, 2, 2) = diff2(var_ptr, strides[2]);

            d2(icomp, 0, 1) = mixed_diff2(var_ptr, strides[0], strides[1]);
            d2(icomp, 0, 2) = mixed_diff2(var_ptr, strides[0], strides[2]);
            d2(icomp, 1, 2) = mixed_diff2(var_ptr, strides[1], strides[2]);
        }
#elif DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        Tensor::Rank1 V;
        Tensor::Rank2 d1_V;
        FOR (icomp)
        {
            const int ivar      = ivar_0 + icomp;
            const auto *var_ptr = get_var_ptr(ivar, state_ptr_xyz, strides);
            V(icomp) = *var_ptr;

            FORSPACEDIM(idir)
            {
                d1_V(icomp, idir) = diff1(var_ptr, strides[idir]);
                d2(icomp, idir, idir) = diff2(var_ptr, strides[idir]);
            }

            d2(icomp, 0, 1) = mixed_diff2(var_ptr, strides[0], strides[1]);
        }
        // Fill cartoon derivatives
        const amrex::Real y = (iy + 0.5) * m_dx;
        const amrex::Real one_over_y = 1.0 / y;
        const amrex::Real one_over_y2 = one_over_y * one_over_y;
        CartoonDerivs::fill_cartoon_derivs_d2_vector(V, d1_V, one_over_y,
                                                     one_over_y2, d2);
#endif
        return d2;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Sym34Rank4
    d2_tensor(int ix, int iy, int iz,
              const amrex::Array4<const amrex::Real> &state,
              const int ivar_0) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        Tensor::Sym34Rank4 d2;
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        auto strides              = get_strides(state);

        int ivar{ivar_0};
#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM && AMREX_SPACEDIM == 3
        FOR (icomp)
            FOR (jcomp)
            {
                const auto *var_ptr = get_var_ptr(ivar, state_ptr_xyz, strides);

                d2(icomp, jcomp, 0, 0) = diff2(var_ptr, strides[0]);
                d2(icomp, jcomp, 1, 1) = diff2(var_ptr, strides[1]);
                d2(icomp, jcomp, 2, 2) = diff2(var_ptr, strides[2]);

                d2(icomp, jcomp, 0, 1) =
                    mixed_diff2(var_ptr, strides[0], strides[1]);
                d2(icomp, jcomp, 0, 2) =
                    mixed_diff2(var_ptr, strides[0], strides[2]);
                d2(icomp, jcomp, 1, 2) =
                    mixed_diff2(var_ptr, strides[1], strides[2]);

                ++ivar;
            }
#elif DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        Tensor::Rank2 T;
        Tensor::Rank3 d1_T;
        FOR (icomp, jcomp)
        {
            const auto *var_ptr = get_var_ptr(ivar, state_ptr_xyz, strides);
            T(icomp, jcomp) = *var_ptr;

            FORSPACEDIM(idir)
            {
                d1_T(icomp, jcomp, idir) = diff1(var_ptr, strides[idir]);
                d2(icomp, jcomp, idir, idir) = diff2(var_ptr, strides[idir]);
            }

            d2(icomp, jcomp, 0, 1) =
                mixed_diff2(var_ptr, strides[0], strides[1]);

            ++ivar;
        }
        // Fill cartoon derivatives
        const amrex::Real y = (iy + 0.5) * m_dx;
        const amrex::Real one_over_y = 1.0 / y;
        const amrex::Real one_over_y2 = one_over_y * one_over_y;
        CartoonDerivs::fill_cartoon_derivs_d2_tensor(T, d1_T, one_over_y,
                                                     one_over_y2, d2);
#endif
        return d2;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Sym12Sym34Rank4
    d2_sym_tensor(int ix, int iy, int iz,
                  const amrex::Array4<const amrex::Real> &state,
                  const int ivar_0) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        Tensor::Sym12Sym34Rank4 d2;
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        auto strides              = get_strides(state);

#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM && AMREX_SPACEDIM == 3
        FOR (icomp)
            FOR (jcomp)
            {
                const int ivar      = sym_var_idx(ivar_0, icomp, jcomp);
                const auto *var_ptr = get_var_ptr(ivar, state_ptr_xyz, strides);

                d2(icomp, jcomp, 0, 0) = diff2(var_ptr, strides[0]);
                d2(icomp, jcomp, 1, 1) = diff2(var_ptr, strides[1]);
                d2(icomp, jcomp, 2, 2) = diff2(var_ptr, strides[2]);

                d2(icomp, jcomp, 0, 1) =
                    mixed_diff2(var_ptr, strides[0], strides[1]);
                d2(icomp, jcomp, 0, 2) =
                    mixed_diff2(var_ptr, strides[0], strides[2]);
                d2(icomp, jcomp, 1, 2) =
                    mixed_diff2(var_ptr, strides[1], strides[2]);
            }
#elif DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        Tensor::Sym12Rank2 T;
        Tensor::Sym12Rank3 d1_T;
        for (int icomp = 0; icomp < NUM_SYM_IDXS; ++icomp)
        {
            const int ivar      = ivar_0 + icomp;
            const auto *var_ptr = get_var_ptr(ivar, state_ptr_xyz, strides);

            T(icomp) = *var_ptr;
            FORSPACEDIM(idir)
            {
                d1_T(icomp, idir) = diff1(var_ptr, strides[idir]);
                d2(icomp, sym_var_idx(idir, idir)) =
                    diff2(var_ptr, strides[idir]);
            }
            d2(icomp, sym_var_idx(0, 1)) =
                mixed_diff2(var_ptr, strides[0], strides[1]);
        }
        // Both index pairs in d2 are stored in compressed symmetric form. The
        // sym_var_idx calls above select the compressed derivative pair while
        // the two-index accessor selects the compressed tensor component.

        // Fill cartoon derivatives
        const amrex::Real y = (iy + 0.5) * m_dx;
        const amrex::Real one_over_y = 1.0 / y;
        const amrex::Real one_over_y2 = one_over_y * one_over_y;
        CartoonDerivs::fill_cartoon_derivs_d2_sym_tensor(T, d1_T, one_over_y,
                                                         one_over_y2, d2);
#endif
        return d2;
    }

  protected:
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    advection_term(const amrex::Real *in_ptr, const amrex::Real &shift_comp,
                   const int stride, const bool shift_positive,
                   const int idx = 0) const
    {
        amrex::Real weight_0 = -2.50000000000000000000e-1_rt;
        amrex::Real weight_1 = -8.33333333333333333333e-1_rt;
        amrex::Real weight_2 = +1.50000000000000000000e+0_rt;
        amrex::Real weight_3 = -5.00000000000000000000e-1_rt;
        amrex::Real weight_4 = +8.33333333333333333333e-2_rt;

        amrex::Real upwind =
            shift_comp *
            (weight_0 * in_ptr[idx - stride] + weight_1 * in_ptr[idx] +
             weight_2 * in_ptr[idx + stride] +
             weight_3 * in_ptr[idx + 2 * stride] +
             weight_4 * in_ptr[idx + 3 * stride]) *
            m_one_over_dx;

        amrex::Real downwind =
            shift_comp *
            (-weight_4 * in_ptr[idx - 3 * stride] -
             weight_3 * in_ptr[idx - 2 * stride] -
             weight_2 * in_ptr[idx - stride] - weight_1 * in_ptr[idx] -
             weight_0 * in_ptr[idx + stride]) *
            m_one_over_dx;

        return (shift_positive) ? upwind : downwind;
    }

  public:

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    advection(int ix, int iy, int iz,
              const amrex::Array4<const amrex::Real> &state,
              const Tensor::Rank1 &shift_vector, const int ivar) const
    {
        amrex::Real advec         = 0.0;
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        const auto strides        = get_strides(state);
        const auto *var_ptr       = get_var_ptr(ivar, state_ptr_xyz, strides);

        FORSPACEDIM (idir)
        {
            const bool shift_positive = (shift_vector(idir) > 0.0);
            advec += advection_term(var_ptr, shift_vector(idir), strides[idir],
                                    shift_positive);
        }
        return advec;
    }

    // NOLINTBEGIN(readability-convert-member-functions-to-static)
    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    advec_scalar(int ix, int iy, int iz,
                 const amrex::Array4<const amrex::Real> &state,
                 const Tensor::Rank1 &shift_vector, const int ivar) const
    {
        // For Cartoon reduction, S_{,z} = 0 by symmetry
        // so we don't need to compute the z advection term for a scalar
        return advection(ix, iy, iz, state, shift_vector, ivar);
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Rank1
    advec_vector(int ix, int iy, int iz,
                 const amrex::Array4<const amrex::Real> &state,
                 const Tensor::Rank1 &shift_vector, const int ivar0) const
    {
        Tensor::Rank1 advec_vector{};

        FOR (icomp)
        {
            int ivar = ivar0 + icomp;
            advec_vector(icomp) =
                advection(ix, iy, iz, state, shift_vector, ivar); 
        }
#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        // Add z derivative for Cartoon reduction
        amrex::Real sz = shift_vector(2);
        Tensor::Rank2 d1_V = d1_vector(ix, iy, iz, state, ivar0);
        FOR(icomp)
        {
            advec_vector(icomp) += sz * d1_V(icomp, 2);
        }
#endif
        return advec_vector;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Rank2
    advec_tensor(int ix, int iy, int iz,
                 const amrex::Array4<const amrex::Real> &state,
                 const Tensor::Rank1 &shift_vector, const int ivar0) const
    {
        Tensor::Rank2 advec_tensor{};
        FOR (icomp, jcomp)
        {
            int ivar = sym_var_idx(ivar0, icomp, jcomp);
            advec_tensor(icomp, jcomp) =
                advection(ix, iy, iz, state, shift_vector, ivar);
        }
#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        // Add z derivative for Cartoon reduction
        amrex::Real sz = shift_vector(2);
        Tensor::Rank3 d1_T = d1_tensor(ix, iy, iz, state, ivar0);
        FOR(icomp, jcomp)
        {
            advec_tensor(icomp, jcomp) += sz * d1_T(icomp, jcomp, 2);
        }
#endif
        return advec_tensor;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE Tensor::Sym12Rank2
    advec_sym_tensor(int ix, int iy, int iz,
                     const amrex::Array4<const amrex::Real> &state,
                     const Tensor::Rank1 &shift_vector, const int ivar0) const
    {
        Tensor::Sym12Rank2 advec_tensor{};

        for (int i = 0; i < NUM_SYM_IDXS; ++i)
        {
            advec_tensor(i) =
                advection(ix, iy, iz, state, shift_vector, ivar0 + i);
        }
#if DEFAULT_TENSOR_DIM == AMREX_SPACEDIM + 1 && AMREX_SPACEDIM == 2
        // Add z derivative for Cartoon reduction
        amrex::Real sz = shift_vector(2);
        Tensor::Sym12Rank3 d1_T = d1_sym_tensor(ix, iy, iz, state, ivar0);
        for (int i = 0; i < NUM_SYM_IDXS; ++i)
        {
            advec_tensor(i) += sz * d1_T(i, 2);
        }
#endif
        return advec_tensor;
    }
    // NOLINTEND(readability-convert-member-functions-to-static)

    // gets the derivative of a consecutive series of vars in a state
    template <int num_diff_vars>
    [[nodiscard]] AMREX_GPU_DEVICE
        AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, num_diff_vars>
        advec_state(int ix, int iy, int iz,
                    const amrex::Array4<const amrex::Real> &state,
                    const Tensor::Rank1 &shift_vector, int first_var = 0) const

    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        amrex::GpuArray<amrex::Real, num_diff_vars> advec_state;
        for (int ivar = first_var; ivar < (first_var + num_diff_vars); ++ivar)
        {
            advec_state[ivar - first_var] =
                advection(ix, iy, iz, state, shift_vector, ivar);
        }
        return advec_state;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    dissipation_term(const amrex::Real *in_ptr, const int stride,
                     const int idx = 0) const
    {
        amrex::Real weight_vfar  = 1.56250e-2_rt;
        amrex::Real weight_far   = 9.37500e-2_rt;
        amrex::Real weight_near  = 2.34375e-1_rt;
        amrex::Real weight_local = 3.12500e-1_rt;

        return (weight_vfar * in_ptr[idx - 3 * stride] -
                weight_far * in_ptr[idx - 2 * stride] +
                weight_near * in_ptr[idx - stride] -
                weight_local * in_ptr[idx] +
                weight_near * in_ptr[idx + stride] -
                weight_far * in_ptr[idx + 2 * stride] +
                weight_vfar * in_ptr[idx + 3 * stride]) *
               m_one_over_dx;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    calculate_dissipation(int ix, int iy, int iz,
                          const amrex::Array4<const amrex::Real> &state,
                          const amrex::Real sigma_coeff, const int ivar) const
    {
        amrex::Real diss          = 0.0;
        const auto *state_ptr_xyz = state.ptr(ix, iy, iz);
        const auto strides        = get_strides(state);

        FORSPACEDIM (idir)
        {
            const auto stride  = strides[idir];
            diss              += sigma_coeff *
                    dissipation_term(get_var_ptr(ivar, state_ptr_xyz, strides),
                                     stride);
        }
        return diss;
    }

    // NOLINTBEGIN(readability-convert-member-functions-to-static)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    add_dissipation(int ix, int iy, int iz,
                    const amrex::CellData<amrex::Real> &rhs_cell_data,
                    const amrex::Array4<const amrex::Real> &state,
                    const amrex::Real sigma_coeff,
                    int num_vars = NUM_VARS) const

    {
        amrex::Real diss = 0.0;
        for (int ivar = 0; ivar < num_vars; ++ivar)
        {
            diss = calculate_dissipation(ix, iy, iz, state, sigma_coeff, ivar);
            rhs_cell_data[ivar] += diss;
        }
    }
    // NOLINTEND(readability-convert-member-functions-to-static)
    // NOLINTEND(bugprone-easily-swappable-parameters)
};

#endif /* FOURTHORDERDERIVATIVES_HPP_ */
