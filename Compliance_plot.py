import numpy as np
import matplotlib.pyplot as plt


p = np.linspace(-20,20)


PVI1 = 0.5 #ml
PVI2 = 1 #ml

dt = 0.04 #s
N = 24 #Timesteps
DV = np.linspace(-0.5,0.5,100)

p0 = 133 #1mmHg

ICP1 = p0*(10**(DV*dt/PVI1) - 1)
ICP2 = p0*(10**(DV*dt/PVI2) - 1)
#ICP3 = p0*10**(DV*dt3/PVI1)
#ICP4 = p0*10**(DV*dt/PVI4)

plt.plot(DV,ICP1,label='PVI = 0.5')
plt.plot(DV,ICP2,label='PVI = 1')
#plt.plot(DV,ICP3,label='PVI = 8')
#plt.plot(DV,ICP4,label='PVI = 1.5')
plt.xlabel("DeltaV[ml]")
plt.ylabel("Pressure[Pa]")
plt.legend()
plt.show()
