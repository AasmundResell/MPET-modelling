import numpy as np
import matplotlib.pyplot as plt


p = np.linspace(-20,20)

E_1 = 2#[1/mL]

E_2 = 0.2


p_r1 = 0 #referance pressure, mmHg
p_r2 = 0#baseline pressure, mmHg

C_1 = 1/(E_1*(p-p_r1))
C_2 = 1/(E_2*(p-p_r2))

plt.plot(p,C_1,label='C_1')
plt.plot(p,C_2,label='C_2')
plt.legend()
plt.show()
