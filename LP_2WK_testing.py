from gettext import dpgettext
import numpy as np
import matplotlib.pyplot as plt
"""

    This model couples the two pressures between the ventricles and SAS through the aqueduct, both compartments are modeled
    with Windkessel models.

    Solves using implicit (backward) Euler

    dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS))
    dp_ven/dt = 1/C_ven(Vv_dot + Q_VEN + G_aq(p_SAS - p_VEN))

    """

VolScale = 1/1000  # mm³ to mL

T = 24
N = 600

dt = T/600
print("dt:", dt)

t = np.linspace(0, T, N+1)
p = np.zeros((2, t.shape[0]))
Q_AQ = np.zeros_like(t)

dp = 0.5  # mmHg
C_SAS = 1/(dp*133)  # [mL/Pa]
C_VEN = 1/(dp*133)  # [mL/Pa]

C = np.array([[C_SAS], [C_VEN]])

Vv_dot = 1000/2*np.sin(2*np.pi*(1/4+t))
Vs_dot = 1000*np.sin(2*np.pi*(1/4+t))
V_dot = np.array([Vs_dot, Vv_dot])

plt.figure(1)
plt.plot(t[500:], Vv_dot[500:],label='Vv_dot')
plt.plot(t[500:], Vs_dot[500:],label='Vs_dot')
plt.legend()

L = 70.0 #[mm]
d = 3.0 #[mm]
mu_f = 6.97e-4 #Viscosity fluid 3, Pa*s
 

# Conductance
G_aq = 5/133  # mL/(mmHg*s) to mL/(Pa*s), from Ambarki2007
G_aq = np.pi*d**4/(128*L*mu_f)*VolScale # mm³/(Pa*s) to mL/(Pa*s)

print(G_aq)

b = ([[0.0], [0.0]])



A_11 = 1 + dt*G_aq/C_SAS
A_12 = -dt*G_aq/C_SAS
A_21 = -dt*G_aq/C_VEN
A_22 = 1 + dt*G_aq/C_VEN

A = np.array([[A_11, A_12], [A_21, A_22]])

x = np.linalg.solve(A, b)  # x_0 = p_SAS, x_1 = p_VEN

for i, tt in enumerate(t[1:]):

    b = p[:, i-1] + (dt/C).T*V_dot[:, i-1] * VolScale

    # x_0 = p_SAS, x_1 = p_VEN, x_2 = p_SP
    p[:, i] = np.linalg.solve(A, b.T).flatten()

    #print("Pressure for SAS: ", p[0, i])
    #print("Pressure for ventricles: ", p[1, i])

    # "Positive" direction upwards, same as baledent article
    Q_AQ[i] = G_aq*(p[0, i] - p[1, i])
    

    #print("Q_AQ[mL]:", Q_AQ[i])
    

plt.figure(2)
plt.plot(t[500:], p[0, 500:], label='p_s')
plt.plot(t[500:], p[1, 500:], label='p_v')
plt.legend()

plt.figure(3)
plt.plot(t[500:], Q_AQ[500:], label='Q_aq')
plt.xlabel("t")
plt.ylabel("mL/s")
plt.legend()
plt.show()


print("Pressure for SAS: ", x[0])
print("Pressure for VEN: ", x[1])

