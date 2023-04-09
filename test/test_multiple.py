import os
import matplotlib.pyplot as plt
import numpy as np
import time

os.add_dll_directory(r"C:\Users\dfavadf\source\repos\QWM\test")

from QWM import QWMSolver

# print(QWM.__doc__)

eta1 = 2 * np.pi * 0.1
g1 = 2 * np.pi * 3
g2 = 2 * np.pi * 3
alpha  = 0.5
domega = 2 * np.pi * 0.1
eps1 = 0
eps2 = 0

params = (eta1, g1, g2, alpha, domega, eps1, eps2)

T = 160
dt = 0.1
t_init_to_skip = 20

qwmsolver = QWMSolver(params, T, dt, t_init_to_skip)

start_time1 = time.perf_counter()
result = qwmsolver.rosenbrock_solve_multiple(np.logspace(-1, 2, 100), np.logspace(-1, 2, 100))
end_time1 = time.perf_counter()

print(f"Execution time: {(end_time1 - start_time1) * 1000} ms")

xx, yy = np.meshgrid(W_powers, E_powers)

zz_arrays = [np.empty_like(xx) for i in range(8)]

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
# cax, kw = colorbar.make_axes(axes[:, -1], use_gridspec=True, aspect=20, pad=0.02)


axes_list = [*axes[0, :], *axes[1, :]]

plt.suptitle("Simulation")
def plot_model_at(ax, n):
	zz = np.empty_like(xx)
	for i in range(len(W_powers)):
		for j in range(len(E_powers)):
			zz[j, i] = np.abs(results[j * len(E_powers) + i][1][n])
	plot_2D(ax, xx, yy, zz, xlabel=r"$\log_{10}(\nu_{\omega_+}/\Gamma_1)$", cmap='gnuplot2', title=f"idx = {2 * n - 8 + 1}", show_colorbar=False)

for i in range(8):
	plot_model_at(axes_list[i], i)
	axes_list[i].set_aspect("equal")
	axes_list[i].set_xticks([])
	axes_list[i].set_yticks([])

# cb = plt.colorbar(axes[0, 0].get_images(), cax=cax)

plt.subplots_adjust(hspace=0.2, wspace=0.1, right=0.87, top=0.9, bottom=0.05)
plt.show()