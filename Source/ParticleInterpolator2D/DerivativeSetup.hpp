/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef DERIVATIVESETUP_HPP_
#define DERIVATIVESETUP_HPP_

#include "Derivative.hpp"

inline const Derivative Derivative::LOCAL;

inline const Derivative Derivative::dx(0);
inline const Derivative Derivative::dy(1);


inline const Derivative Derivative::dxdx(0, 0);
inline const Derivative Derivative::dydy(1, 1);



inline const Derivative Derivative::dxdy(0, 1);


#endif /* DERIVATIVESETUP_HPP_ */
