import os
import matplotlib.pyplot as plt
import numpy as np
import time

os.add_dll_directory(r"C:\Users\dfavadf\source\repos\QWM\test")

from QWM import QWMSolver

# print(QWM.__doc__)

eta1 = 2 * np.pi * 0.1
g1 = 2 * np.pi * 2
g2 = 2 * np.pi * 2
alpha  = 0.5
domega = 2 * np.pi * 0.025
eps1 = 0
eps2 = 0

params = (eta1, g1, g2, alpha, domega, eps1, eps2)

T = 80
dt = 0.1
t_init_to_skip = 20
print(t_init_to_skip)

qwmsolver = QWMSolver(params, T, dt, t_init_to_skip)

start_time1 = time.perf_counter()
tlist, sm1, sm2 = qwmsolver.rk45_solve((1., 1.))
end_time1 = time.perf_counter()

start_time2 = time.perf_counter()
tlist2, sm12, sm22 = qwmsolver.rosenbrock_solve((1., 1.))
end_time2 = time.perf_counter()

print(f"rungekutta45:\t{(end_time1 - start_time1) * 1000} ms")
print(f"rosenbrock:\t{(end_time2 - start_time2) * 1000} ms")

fig, ax = plt.subplots(2, 1)
ax[0].plot(tlist, np.real(sm1))
ax[0].plot(tlist, np.imag(sm1))
ax[1].plot(tlist, np.real(sm2))
ax[1].plot(tlist, np.imag(sm2))

ax[0].plot(tlist2, np.real(sm12))
ax[0].plot(tlist2, np.imag(sm12))
ax[1].plot(tlist2, np.real(sm22))
ax[1].plot(tlist2, np.imag(sm22))
plt.show()