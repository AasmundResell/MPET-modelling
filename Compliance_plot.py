import numpy as np
import matplotlib.pyplot as plt


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
