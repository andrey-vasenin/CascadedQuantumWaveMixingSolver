import os
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import *

os.add_dll_directory(r"C:\Users\dfavadf\source\repos\QWM\test")

from QWM import qwm_parallel_solve

# print(QWM.__doc__)

eta1 = 2 * np.pi * 0.1
g1 = 2 * np.pi * 2
g2 = 2 * np.pi * 2
alpha  = 0.7
domega = 2 * np.pi * 0.1
eps1 = 0
eps2 = 0

params = (eta1, g1, g2, alpha, domega, eps1, eps2)

T = 100
dt = 0.5

print("Start")

E_powers = np.linspace(-1, 1, 101)
W_powers = np.linspace(-0.5, 1.5, 101)

t_init_to_skip = 20

start_time1 = time.perf_counter()
results = np.array(qwm_parallel_solve(params, T, dt, 10**W_powers, 10**E_powers, t_init_to_skip))
end_time1 = time.perf_counter()

print(f"Execution time: {(end_time1 - start_time1) * 1000} ms")

xx, yy = np.meshgrid(E_powers, W_powers)

sm = np.sqrt(params[2] / 2) * results[:, 1, :] + np.sqrt(params[1] * params[3]) * results[:, 0, :]
sm = np.swapaxes(sm, 0, 1)
sm = np.reshape(sm, (8, len(E_powers), len(W_powers)))

sm /= (T - t_init_to_skip) / dt

sm[3] += -1j * 10**(xx / 2)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
# cax, kw = colorbar.make_axes(axes[:, -1], use_gridspec=True, aspect=20, pad=0.02)


axes_list = [*axes[0, :], *axes[1, :]]

noise = .001

plt.suptitle("Simulation")
def plot_model_at(ax, n):
    plot_2D(ax, xx, yy, 10*np.log10(np.abs(sm[n])**2 + noise), xlabel=r"$\log_{10}(\nu_{\omega_+}/\Gamma_1)$",
            cmap='gnuplot2', title=f"idx = {2 * n - 8 + 1}", show_colorbar=False,
            vmin=10*np.log10(noise), vmax=10*np.log10(1000 * noise))
    # ax.set_box_aspect(1)

for i in range(8):
    plot_model_at(axes_list[i], i)
    axes_list[i].set_aspect("equal")
    axes_list[i].set_xticks([])
    axes_list[i].set_yticks([])

# cb = plt.colorbar(axes[0, 0].get_images(), cax=cax)

plt.subplots_adjust(hspace=0.2, wspace=0.1, right=0.87, top=0.9, bottom=0.05)
plt.show()