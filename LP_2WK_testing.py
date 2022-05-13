#from gettext import dpgettext
import numpy as np
import matplotlib.pyplot as plt

"""

    This model couples the two pressures between the ventricles and SAS through the aqueduct, both compartments are modeled
    with Windkessel models.

    Solves using implicit (backward) Euler

    dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS))
    dp_ven/dt = 1/C_ven(Vv_dot + Q_VEN + G_aq(p_SAS - p_VEN))

    """


T = 24
N = 600
p0 = 1330 #10 mmHg
t = np.linspace(0, T, N+1)


L = 70.0 #mm
d = 3.0 #mm
mu_f = 6.97e-4 #Pa*s

#Resistance
R_aq = 128*L*mu_f/(np.pi*d**4)*1000 #Poiseuille flow constant Pa/mm³ to Pa/mL

dp = 0.5  # mmHg
#C_SAS = 1/(dp*133)  # [mL/Pa]
C_ven = 1/(dp*133)  # [mL/Pa]

C = np.array([[C_SAS], [C_VEN]])

dt = T/600
PVI = 3 #mL
#C_ven = 10 # mL/Pa


p = np.zeros((2, t.shape[0]))
Q_AQ = np.zeros_like(t)

Vv_dot = 1/2*np.sin(2*np.pi*(1/4+t))
Vs_dot = 1*np.sin(2*np.pi*(1/4+t))
V_dot = np.array([Vs_dot, Vv_dot])

plt.figure(1)
plt.plot(t[500:], Vv_dot[500:],label='Vv_dot')
plt.plot(t[500:], Vs_dot[500:],label='Vs_dot')
plt.legend()

p_ven_n = 0

print("dt:", dt)



for i, tt in enumerate(t[1:]):

    p_ven = (Vv_dot[i] + C_ven/dt*p_ven_n)/(C_ven/dt+1/R_aq)
    ICP = p0*10**((Vs_dot[i] + p_ven/R_aq)*dt/PVI)

    p[:, i+1] = ICP, ICP + p_ven

    print("Pressure for SAS: ", p[0, i])
    print("Pressure for ventricles: ", p[1, i])

    # "Positive" direction upwards, same as baledent article
    Q_AQ[i] = 1/R_aq*p_ven

    p_ven_n = p_ven

    #print("Q_AQ[mL]:", Q_AQ[i])
    
ICP_v = ICP + p_ven
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


#print("Pressure for SAS: ", x[0])
#print("Pressure for VEN: ", x[1])

