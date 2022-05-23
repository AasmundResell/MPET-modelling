from gettext import dpgettext
import numpy as np
import matplotlib.pyplot as plt
"""
This model calculates a 3-pressure lumped model for the SAS, ventricles and spinal-SAS compartments

Solves using implicit (backward) Euler

Equations:
dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS) + G_fm(p_SP-p_SAS)
dp_ven/dt = 1/C_ven(Vv_dot + Q_VEn + G_aq(p_SAS - p_VEN))
dp_sp/dt = 1/C_sp(G_fm(p_SAS-p_SP))

"""

VolScale = 1/1000  # mm³ to mL

T = 12
N = 300

dt = T/N
print("dt:", dt)

t = np.linspace(0, T, N+1)
p = np.zeros((3, t.shape[0]))
Q_AQ = np.zeros_like(t)
Q_FM = np.zeros_like(t)

source_scale = 1/1173887

sourceFile = "data/Arterial_bloodflow_shifted.csv"
time_period = 1.0
data = np.loadtxt(sourceFile, delimiter = ",")
data_t = data[:,0]
source = data[:,1]
g = np.interp(t,data_t,source,period = 1.0)*source_scale
gm = np.mean(g)
print(gm)
plt.plot(t,g)
plt.show()



dp = 0.5 #mmHg
C_SAS = 1/(dp*133)  # [mL/Pa]
C_VEN = 1/(dp*133)  # [mL/Pa]
C_SP = C_SAS*10  # [mL/Pa#]

C = np.array([[C_SAS], [C_VEN], [C_SP]])


Vv_dot = -50000*(g - gm)  
Vs_dot = 100000*(g - gm)

plt.plot(t,Vs_dot)
plt.show()


#Vv_dot = 1000*np.sin(2*np.pi*(1/4+t))  
#Vs_dot = 1000*np.sin(2*np.pi*(3/4+t)) 
Vsp_dot = np.zeros_like(t)
V_dot = np.array([Vs_dot, Vv_dot, Vsp_dot])

"""
plt.figure(1)
plt.plot(t[250:], Vv_dot[250:],label='Vv_dot')
plt.plot(t[250:], Vs_dot[250:],label='Vs_dot')
plt.legend()
plt.show()
"""

# Conductance
G_aq = 5/133  # mL/mmHg to mL/Pa, from Ambarki2007
G_fm = G_aq*10  # from Ambarki2007

b = ([[0.0], [0.0], [0.0]])
A_11 = 1 + dt*G_aq/C[0,0] + dt*G_fm/C[0,0]
A_12 = -dt*G_aq/C[0,0]
A_13 = -dt*G_fm/C[0,0]
A_21 = -dt*G_aq/C[1,0]
A_22 = 1 + dt*G_aq/C[1,0]
A_23 = 0
A_31 = -dt*G_fm/C[2,0]
A_32 = 0
A_33 = 1 + dt*G_fm/C[2,0]

A = np.array([[A_11, A_12, A_13], [A_21, A_22, A_23], [A_31, A_32, A_33]])

for i, tt in enumerate(t[1:]):

    b = p[:, i-1] + (dt/C).T*V_dot[:, i-1] * VolScale

    p[:, i] = np.linalg.solve(A, b.T).flatten()  # x_0 = p_SAS, x_1 = p_VEN, x_2 = p_SP
     
    print("Pressure for SAS: ", p[0, i])
    print("Pressure for ventricles: ", p[1, i])
    print("Pressure in spinal-SAS:", p[2, i])

    # "Positive" direction upwards, same as baledent article
    Q_AQ[i] = G_aq*(p[0, i] - p[1, i])
    Q_FM[i] = G_fm*(p[2, i] - p[0, i])

    print("Q_AQ[mL]:", Q_AQ[i])
    print("Q_FM[mL]:", Q_FM[i])

plt.figure(2)
plt.plot(t[250:], p[0,250:],label='p_s')
plt.plot(t[250:], p[1,250:],label='p_v')
plt.plot(t[250:], p[2,250:],label='p_sp')  
plt.legend()

plt.figure(3)
plt.plot(t[250:], Q_AQ[250:],label='Q_aq')
plt.plot(t[250:], Q_FM[250:],label='Q_fm')
plt.legend()
plt.show()



