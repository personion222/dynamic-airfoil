from metpy.units import units
import metpy.calc as mpcalc
import math


def dynamic_visc(T):
    mu0 = 1.716e-5 # dynamic viscosity at T0
    T0 = 273.15 # reference 0 degrees C in K
    C = 110.56 # sutherlands constant of air

    return (mu0 * (T / T0) ** (3/2) * (T0 + C)) / (C + T)

def kinematic_visc(T, p, q):
    mu = dynamic_visc(T.magnitude)

    rho = mpcalc.density(p, T, q).magnitude

    return mu / rho
