"""Physical constants.

Gravitational parameter and equatorial radius follow WGS-84; zonal harmonic
coefficients follow EGM96 (unnormalized). These are the values used throughout
the truth propagator, the analytic STMs, and the mean<->osculating maps, so a
single import site keeps every model consistent.
"""

# Earth gravitational parameter [m^3/s^2] (WGS-84)
MU_EARTH = 3.986004418e14

# Earth equatorial radius [m] (WGS-84)
R_EARTH = 6378137.0

# Unnormalized zonal harmonic coefficients (EGM96)
J2 = 1.08262668e-3
J3 = -2.53265649e-6
J4 = -1.61962159e-6

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
