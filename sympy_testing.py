from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame
from sympy.physics.mechanics import Point, Particle, KanesMethod
q, u = dynamicsymbols('q u')
qd, ud = dynamicsymbols('q u', 1)
m, c, k = symbols('m c k')
N = ReferenceFrame('N')
P = Point('P')
P.set_vel(N, u * N.x)

kd = [qd - u]
FL = [(P, (-k * q - c * u) * N.x)]
pa = Particle('pa', P, m)
BL = [pa]

KM = KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=kd)
(fr, frstar) = KM.kanes_equations(BL, FL)
MM = KM.mass_matrix
forcing = KM.forcing
rhs = MM.inv() * forcing
print(rhs)
# Matrix([[(-c*u(t) - k*q(t))/m]])
# KM.linearize(A_and_B=True)[0]
# Matrix([
# [   0,    1],
# [-k/m, -c/m]])