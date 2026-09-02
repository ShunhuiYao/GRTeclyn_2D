/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef GRUTILS_HPP_
#define GRUTILS_HPP_

// #ifndef GR_SPACEDIM
// #define GR_SPACEDIM 3
// #endif
constexpr int GR_SPACEDIM = 3;

#ifndef DEFAULT_TENSOR_DIM
//#define DEFAULT_TENSOR_DIM AMREX_SPACEDIM
#define DEFAULT_TENSOR_DIM 3
#endif

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

// Fancy 'for' loop macros to iterate through spatial tensors
// use as "FOR(i, j) { ... }"
#define FOR1(IDX) for (int(IDX) = 0; (IDX) < DEFAULT_TENSOR_DIM; ++(IDX))
#define FOR2(IDX1, IDX2)                                                       \
    FOR1 (IDX1)                                                                \
        FOR1 (IDX2)
#define FOR3(IDX1, IDX2, IDX3)                                                 \
    FOR2 (IDX1, IDX2)                                                          \
        FOR1 (IDX3)
#define FOR4(IDX1, IDX2, IDX3, IDX4)                                           \
    FOR2 (IDX1, IDX2)                                                          \
        FOR2 (IDX3, IDX4)
#define FOR5(IDX1, IDX2, IDX3, IDX4, IDX5)                                     \
    FOR4 (IDX1, IDX2, IDX3, IDX4)                                              \
        FOR1 (IDX5)
#define DUMMYFOR() // prevents warning that appeared in debug mode:
                   // 'ISO C++11 requires at least one argument for the "..." in
                   // a variadic macro'

#define FOR2_SYM(IDX1, IDX2)                                                   \
    for (int(IDX1) = 0; (IDX1) < DEFAULT_TENSOR_DIM; ++(IDX1))                 \
        for (int(IDX2) = IDX1; (IDX2) < DEFAULT_TENSOR_DIM; ++(IDX2))

#define GET_MACRO6(_1, _2, _3, _4, _5, NAME, ...) NAME
#define FOR(...)                                                               \
    GET_MACRO6(__VA_ARGS__, FOR5, FOR4, FOR3, FOR2, FOR1, DUMMYFOR)(__VA_ARGS__)
// NOLINTEND(cppcoreguidelines-macro-usage)

// FORSPACEDIM macros iterate only over actual grid directions (AMREX_SPACEDIM).
// Use these wherever AMREX stride arrays (m_in_stride, m_out_stride) are
// indexed — they have AMREX_SPACEDIM entries, not DEFAULT_TENSOR_DIM.
#define FORSPACEDIM1(IDX) for (int IDX = 0; IDX < AMREX_SPACEDIM; ++IDX)
#define FORSPACEDIM2(IDX1, IDX2) FORSPACEDIM1(IDX1) FORSPACEDIM1(IDX2)
#define FORSPACEDIM3(IDX1, IDX2, IDX3)                                         \
    FORSPACEDIM2(IDX1, IDX2) FORSPACEDIM1(IDX3)
#define FORSPACEDIM4(IDX1, IDX2, IDX3, IDX4)                                  \
    FORSPACEDIM2(IDX1, IDX2) FORSPACEDIM2(IDX3, IDX4)

#define FORSPACEDIM(...)                                                        \
    GET_MACRO6(__VA_ARGS__, DUMMYFOR, FORSPACEDIM4, FORSPACEDIM3,              \
               FORSPACEDIM2, FORSPACEDIM1, DUMMYFOR)(__VA_ARGS__)



#endif /* GRUTILS_HPP_*/
