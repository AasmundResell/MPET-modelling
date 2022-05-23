import numpy as np
import matplotlib.pyplot as plt


"""
p = np.linspace(-20,20)


PVI1 = 3 #ml
PVI2 = 6 #ml

dt = 0.04 #s
N = 24 #Timesteps
DV = np.linspace(0.0,10,100)

#p0 = 1596 #1mmHg
p1 = 1330 #1mmHg

ICP1 = p0*10**(DV*dt/PVI1)
ICP2 = p0*np.exp(DV*dt/(0.4343*PVI1))
#ICP3 = p0*10**(DV*dt3/PVI1)
#ICP4 = p0*10**(DV*dt/PVI4)

plt.plot(DV,ICP1,label='PVI = 3')
plt.plot(DV,ICP2,label='PVI = 6')
#plt.plot(DV,ICP3,label='PVI = 8')
#plt.plot(DV,ICP4,label='PVI = 1.5')
plt.xlabel("DeltaV[ml]")
plt.ylabel("Pressure[Pa]")
plt.legend()
plt.show()
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
#sourceFile = "data/baladont_tot_inflow_series_shifted.csv"

time_period = 1.0
data = np.loadtxt(sourceFile, delimiter = ",")
data_t = data[:,0]
source = data[:,1]
g = np.interp(t,data_t,source,period = 1.0)*source_scale
gm = np.mean(g)
print(gm)

Vv_dot = -20000*(g - gm)*VolScale  
Vs_dot = 200000*(g - gm)*VolScale

vsm = np.mean(Vs_dot)


dV = np.zeros_like(Vs_dot)

plt.figure(1)
plt.plot(t, Vv_dot,label='Vv_dot')
plt.plot(t, Vs_dot,label='Vs_dot')
plt.legend()
plt.show()

for i,vs in enumerate(Vs_dot):
    dV[i] = vs*dt
    if i != 0:
        dV[i] += dV[i-1] 
    

"""
plt.figure(2)
plt.plot(t, dV,label='dV')
plt.legend()
plt.show()
"""

PVI = 7
E = 100
ICP = 1330*10**(dV/PVI)


G_aq = 5/133  # mL/mmHg to mL/Pa, from Ambarki2007
R_aq = 1/G_aq
p_r = ICP - np.mean(ICP)
print("p_r",p_r)
p_b = 0
print("p_b",p_b)
p_ven = ((p_b - p_r)*(R_aq*Vv_dot + p_b-p_r))/((p_b-p_r) + R_aq*np.exp(-E/R_aq*(R_aq*Vv_dot+p_b-p_r)*t))
p_ven = p_ven - np.mean(p_ven)
plt.figure(1)
plt.plot(t[250:], p_ven[250:],label='p_ven')
plt.legend()
plt.show()

ICPV = ICP + p_ven
plt.figure(2)
plt.plot(t[250:], ICPV[250:],label='ICPV')
plt.plot(t[250:], ICP[250:],label='ICP')
plt.legend()
plt.show()


