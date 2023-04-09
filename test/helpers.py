import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar
import scipy.optimize as opt
import os

os.add_dll_directory("C:\\Users\\dfavadf\\Documents\\Research\\CascadedWaveMixing")
from QWM import qwm_parallel_solve, QWMSolver


def open_data(date, name, print_it=True):
    with open(f"{date} - {name}/{name}_raw_data.pkl", "rb") as f:
        data = pickle.load(f)

    with open(f"{date} - {name}/{name}_context.txt", "r") as f:
        ctx = eval(f.read())

    dfreq = np.abs(ctx['mw'][0]['freq'] - ctx['mw'][1]['freq'])/2
    d_idx = int(float(ctx['sa'][0]['nop']) * dfreq / ctx['sa'][0]['span'])
    if print_it:
        print(f"dfreq = {dfreq:.0f} Hz\n"
              f"d_idx = {d_idx}")
    return dfreq, d_idx, data, ctx

def plot_2D(ax, xx, yy, zz, xlabel='', ylabel='', title='',
            xlim=None, ylim=None, vmin=None, vmax=None,
            savepath=None, grid=False, cmap="RdBu_r", show_colorbar=False):
    ax.set_title(title)

    zmax = np.max(zz) if vmax is None else vmax
    zmin = np.min(zz) if vmin is None else vmin

    step_X = xx[0, 1] - xx[0, 0]
    step_Y = yy[1, 0] - yy[0, 0]
    extent = [xx[0, 0] - 1 / 2 * step_X, xx[0, -1] + 1 / 2 * step_X,
              yy[0, 0] - 1 / 2 * step_Y, yy[-1, 0] + 1 / 2 * step_Y]
    ax_map = ax.imshow(zz, origin='lower', cmap=cmap,
                       aspect='auto', vmax=zmax, vmin=zmin, extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(grid)
    if show_colorbar:
        cax, kw = colorbar.make_axes(ax)
        plt.colorbar(ax_map, cax=cax)
    return ax_map

def plot_a_square(data, idx):
    fig, ax = plt.subplots(1, constrained_layout=True)
    XX, YY = np.meshgrid(data['Probe power, dBm'], data['Source power, dBm'])
    ZZ = np.real(data['data'][:, :, idx])
    plot_2D(ax, XX, YY, ZZ, xlabel="dBm", ylabel="dBm", cmap='inferno', show_colorbar=True)
    plt.show()

def calc_ratio(data, d_idx):
    nop = len(data['data'][0, 0])
    c_idx = int(np.floor(nop / 2))
    # positive
    idx1 = c_idx - 3 * d_idx
    idx2 = c_idx - 5 * d_idx
    ZZ1m = np.real(data['data'][:, :, idx1])
    ZZ2m = np.real(data['data'][:, :, idx2])
    tanm = np.power(10, (ZZ2m - ZZ1m) / 20)
    # negative
    idx1 = c_idx + 3 * d_idx
    idx2 = c_idx + 5 * d_idx
    ZZ1p = np.real(data['data'][:, :, idx1])
    ZZ2p = np.real(data['data'][:, :, idx2])
    tanp = np.power(10, (ZZ2p - ZZ1p) / 20)
    # joined
    zzp = np.power(10, ZZ1p / 10 + 10)
    zzm = np.power(10, ZZ1m / 10 + 10)
    wp = zzp / (zzp + zzm)
    wm = zzm / (zzp + zzm)
    tan = tanm * wm + tanp * wp
    theta = 2 * np.arctan(tan)
    ratio = np.sin(theta)
    return ratio

def calc_ratio_2(data, d_idx, maxorder=0):
    nop = len(data['data'][0, 0])
    c_idx = int(np.floor(nop / 2))

    omit = 2
    tans = np.zeros((maxorder - 1 - omit, *data['data'].shape[:2]))
    diffs = np.zeros((maxorder - 1 - omit, *data['data'].shape[:2]))

    for i in range(3 + omit, maxorder+1, 2):
        # positive indices side
        ZZ1 = np.real(data['data'][:, :, c_idx + (i - 2) * d_idx])
        ZZ2 = np.real(data['data'][:, :, c_idx + i * d_idx])
        diff = ZZ2 - ZZ1
        tan = np.power(10, diff / 20)
        idx = int((i - 3 - omit) / 2)
        tans[idx] = tan
        diffs[idx] = np.power(10, ZZ2 / 10 + 10)
        # negative indices side
        ZZ1 = np.real(data['data'][:, :, c_idx - (i - 2) * d_idx])
        ZZ2 = np.real(data['data'][:, :, c_idx - i * d_idx])
        diff = ZZ2 - ZZ1
        tan = np.power(10, diff / 20)
        idx = int((maxorder + i - 4 - 2 * omit) / 2)
        tans[idx] = tan
        diffs[idx] = np.power(10, ZZ2 / 10 + 10)
    diffs /= np.sum(diffs, axis=0)
    tan_mean = np.sum(diffs * tans, axis=0)
    theta = 2 * np.arctan(tan_mean)
    ratio = np.sin(theta)
    return ratio

def calc_product(data, d_idx, maxorder=5):
    nop = len(data['data'][0, 0])
    c_idx = int(np.floor(nop / 2))
    theta = np.arcsin(calc_ratio_2(data, d_idx, maxorder))
    tan = np.tan(theta).T
    tanhalf = np.tan(theta / 2)
    ZZ = np.real(data['data'][:, :, c_idx + d_idx]).T
    Psc1p = np.power(10, ZZ / 10)
    Vmm, Vpp = np.meshgrid(np.power(10, data['Probe power, dBm'] / 20),
        np.power(10, data['Source power, dBm'] / 20))
    k = tan**2 * (Vmm * tanhalf - Vpp)**2 / 64 / Psc1p
    return k

def plot_ratio(data, d_idx):
    fig, ax = plt.subplots(1, constrained_layout=True)
    XX, YY = np.meshgrid(data['Probe power, dBm'], data['Source power, dBm'])
    ZZ = calc_ratio(data, d_idx)
    plot_2D(ax, XX, YY, ZZ, vmin=-1, vmax=1, xlabel="dBm", ylabel="dBm", cmap='RdBu_r', show_colorbar=True)
    plt.show()

def plot_ratio_2(data, d_idx, maxorder=0):
    fig, ax = plt.subplots(1, constrained_layout=True)
    XX, YY = np.meshgrid(data['Probe power, dBm'], data['Source power, dBm'])
    ZZ = calc_ratio_2(data, d_idx, maxorder)
    plot_2D(ax, XX, YY, ZZ, vmin=-1, vmax=1, xlabel="dBm", ylabel="dBm",
        title="Ratio", cmap='RdBu_r', show_colorbar=True)
    plt.show()

def plot_product(data, d_idx, maxorder):
    fig, ax = plt.subplots(1, constrained_layout=True)
    XX, YY = np.meshgrid(data['Probe power, dBm'], data['Source power, dBm'])
    ZZ = calc_product(data, d_idx, maxorder)
    plot_2D(ax, XX, YY, ZZ, xlabel="dBm", ylabel="dBm", cmap='inferno', show_colorbar=True)
    plt.show()

def get_weights_matrix(data, d_idx, maxorder):
    nop = len(data['data'][0, 0])
    c_idx = int(np.floor(nop / 2))
    shape = data['data'][:, :, 0].shape
    weights = np.zeros(shape)
    for i in range(3, maxorder+1, 2):
        weights += np.real(data['data'][:, :, c_idx + i * d_idx])
        weights += np.real(data['data'][:, :, c_idx - i * d_idx])
    # weights = np.power(10, weights / 10 + 10)
    pmin = np.min(weights)
    N = shape[0] * shape[1]
    wz = np.sum(weights) - N * pmin
    weights = (weights - pmin) / wz
    return weights

def get_weights_matrix_for_order(data, d_idx, order):
    nop = len(data['data'][0, 0])
    c_idx = int(np.floor(nop / 2))
    shape = data['data'][:, :, 0].shape
    weights = np.real(data['data'][:, :, c_idx + order * d_idx])
    weights = np.power(10, weights / 10 + 10)
    pmin = np.min(weights)
    N = shape[0] * shape[1]
    wz = np.sum(weights) - N * pmin
    weights = (weights - pmin) / wz
    return weights

def plot_weights_matrix(data, d_idx, maxorder):
    x_label = 'Probe power, dBm'
    y_label = 'Source power, dBm'
    fig, ax = plt.subplots(1, constrained_layout=True)
    XX, YY = np.meshgrid(data[x_label], data[y_label])
    ZZ = get_weights_matrix(data, d_idx, maxorder)
    plot_2D(ax, XX, YY, ZZ, xlabel=x_label, ylabel=y_label, title="Weights matrix",
            cmap='inferno', show_colorbar=True)
    plt.show()

def plot_from_formula(xp, xm, G2_over_G1, det, noise_floor):
    xpp_log, xmm_log = np.meshgrid(xp, xm)
    xpp = 10**(xpp_log/10)
    xmm = 10**(xmm_log/10)
    noise = 10**(noise_floor/10)

    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(10, 5))
    cax, kw = colorbar.make_axes(axes[:, -1], use_gridspec=True, aspect=20, pad=0.02)

    Ps = model(xpp, xmm, G2_over_G1, det, maxorder=2*cols-1) + noise
    Ps = 10 * np.log10(Ps) -70

    plt.suptitle("From the equation")
    def plot_model_at(ax, Varr, title):
        plot_2D(ax, xpp_log, xmm_log, Varr,
                xlabel=r"$\log_{10}(\nu_{\omega_+}/\Gamma_1)$",
                #ylabel=r"$\log_{10}(\nu_{\omega_-}/\Gamma_1)$",
                cmap='gnuplot2', vmin=-125, vmax=-102,title=title, show_colorbar=False)
    for i in range(cols):
        plot_model_at(axes[0, i], Ps[cols-i-1], r"$\omega_{-%s}$"%(str(2*i+1)))
        axes[0, i].set_aspect("equal")
        plot_model_at(axes[1, i], Ps[cols+i], r"$\omega_{+%s}$"%(str(2*i+1)))
        axes[1, i].set_aspect("equal")
        axes[0, i].set_xticks([])
        axes[0, i].set_xlabel(None)
        if i > 0:
            axes[0, i].set_yticks([])
 #           axes[0, i].set_ylabel(None)
            axes[1, i].set_yticks([])
#            axes[1, i].set_ylabel(None)
    axes[0, 0].set_ylabel(r"$\log_{10}(\nu_{\omega_-}/\Gamma_1)$")
    axes[1, 0].set_ylabel(r"$\log_{10}(\nu_{\omega_-}/\Gamma_1)$")
    cb = plt.colorbar(axes[0, 0].get_images()[0], cax=cax)
    plt.subplots_adjust(hspace=0.15, wspace=0.1, right=0.87, top=0.97)
    plt.show()

def plot_data_grid(data, d_idx):
    nop = len(data['data'][0, 0])
    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(9, 5), tight_layout=True)
    x_label = 'Probe power, dBm'
    y_label = 'Source power, dBm'
    XX, YY = np.meshgrid(data[x_label], data[y_label])
    plt.suptitle("From data")
    def plot_data_at(ax, Varr, title):
        plot_2D(ax, XX, YY, Varr,
                xlabel=r"$P_{\omega_+}$, dBm",
                ylabel=r"$P_{\omega_-}$, dBm",
                cmap='gnuplot2', title=title, show_colorbar=True)
    for i in range(cols):
        p = 2 * i + 1
        idx = int(nop / 2 - d_idx * p)
        plot_data_at(axes[0, i], np.real(data['data'][:, :, idx]), f"peak -{2*i+1}")
        axes[0, i].set_aspect("equal")
        idx = int(nop / 2 + d_idx * p)
        plot_data_at(axes[1, i], np.real(data['data'][:, :, idx]), f"peak +{2*i+1}")
        axes[1, i].set_aspect("equal")
    plt.show()

def plot_corrected_data_grid(data, d_idx, xp0, ym0):
    nop = len(data['data'][0, 0])
    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(10, 5))
    cax, kw = colorbar.make_axes(axes[:, -1], use_gridspec=True, aspect=20, pad=0.1)
    y_label = 'Probe power, dBm'
    x_label = 'Source power, dBm'
    Xpp, Ymm = np.meshgrid(data[x_label], data[y_label])
    Xpp -= xp0
    Ymm -= ym0
    plt.suptitle("From data")
    def plot_data_at(ax, Varr, title):
        plot_2D(ax, Xpp, Ymm, Varr,
                xlabel=r"$\log_{10}(\nu_{\omega_+}/\Gamma_1)$",
#                ylabel=r"$\log_{10}(\nu_{\omega_-}/\Gamma_1)$",
                cmap='gnuplot2', title=title, vmin=-125, vmax=-102, show_colorbar=False)
    for i in range(cols):
        p = 2 * i + 1
        idx = int(nop / 2 - d_idx * p)
        plot_data_at(axes[0, i], np.real(data['data'][:, :, idx]), r"$\omega_{-%s}$"%(str(p)))
        idx = int(nop / 2 + d_idx * p)
        plot_data_at(axes[1, i], np.real(data['data'][:, :, idx]), r"$\omega_{+%s}$"%(str(p)))
        axes[0, i].set_aspect("equal")
        axes[1, i].set_aspect("equal")
        axes[0, i].set_xticks([])
        axes[0, i].set_xlabel(None)
        if i > 0:
            axes[0, i].set_yticks([])
 #           axes[0, i].set_ylabel(None)
            axes[1, i].set_yticks([])
#            axes[1, i].set_ylabel(None)
        idx = int(nop / 2 + d_idx * p)
    axes[0, 0].set_ylabel(r"$\log_{10}(\nu_{\omega_-}/\Gamma_1)$")
    axes[1, 0].set_ylabel(r"$\log_{10}(\nu_{\omega_-}/\Gamma_1)$")
    cb = plt.colorbar(axes[0, 0].get_images()[0], cax=cax)
    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=0.87, top=0.95)
    plt.show()


# fitting data
def ratio_func(xpp, xmm, xp0, xm0, g):
    return 2 * np.sqrt(xpp / xp0 * xmm / xm0) / (g / 2 + xpp / xp0 + xmm / xm0)

def residual_func(x0, **kwargs):
    xpp = kwargs['xpp']
    xmm = kwargs['xmm']
    ratio_y = kwargs['ratio']
    weights = kwargs['weights']
    ratio_fit = ratio_func(xpp, xmm, *x0)
    residuals = weights * (ratio_fit - ratio_y)
    return residuals.flatten()

def fitting_procedure(data, d_idx, maxorder):
    XX, YY = np.meshgrid(data['Probe power, dBm'], data['Source power, dBm'])
    xpp = np.power(10, XX/10)
    xmm = np.power(10, YY/10)
    ratio = calc_ratio_2(data, d_idx, maxorder)
    weights = get_weights_matrix(data, d_idx, maxorder)
    initial_guess = np.array([1, 1, 1])
    res = opt.least_squares(residual_func, initial_guess,
                            bounds=(np.array([1e-20, 1e-20, 0.5]),
                                   np.array([np.inf, np.inf, 10])),
                            kwargs={
                                "xpp": xpp,
                                "xmm": xmm,
                                "ratio": ratio,
                                "weights": weights,
                            })
    return res

def fitting_procedure(data, d_idx, maxorder):
    PPp_log, PPm_log = np.meshgrid(data['Probe power, dBm'], data['Source power, dBm'])
    PPp = np.power(10, PPp_log/10)
    PPm = np.power(10, PPm_log/10)

    nop = len(data['data'][0, 0])
    c_idx = int(np.floor(nop/2))
    idc = np.linspace(c_idx - maxorder * d_idx,
        c_idx + maxorder * d_idx,
        maxorder + 1, dtype=int)
    peaks_log = np.transpose(data['data'][:, :, idc], axes=[2, 0, 1])
    peaks = np.power(10, peaks_log / 10)

    weights = np.empty_like(peaks)
    p_end = int((maxorder - 1) / 2) + 1
    yp1_idx = p_end
    for p in range(0, p_end):
        weights[yp1_idx + p] = get_weights_matrix_for_order(data, d_idx, 2 * p + 1)
        weights[yp1_idx - p - 1] = get_weights_matrix_for_order(data, d_idx, -(2 * p + 1))

    def residuals(args):
        Lp = args[0]
        Lm = args[1]
        G = args[2]
        Noise = args[3]
        g = args[4]
        d = args[5]
        f_vals = fit_model(PPp, PPm, g, d,
            Lp, Lm, G, Noise, maxorder)
        R = (f_vals - peaks) * weights
        return R.ravel()

    initial_guess = np.array([10, 10, 1000, 1e-4, 1.0, 0.])
    res = opt.least_squares(residuals, initial_guess,
                            bounds=(np.array([1e-20, 1e-20, 1e-20, 0., 0.5, -10.]),
                                    np.array([np.inf, np.inf, np.inf, np.inf, 5., 10.]))
                            )
    return res

def model(xp, xm, g, d, maxorder=7, skip_first=False):
    A = (g**2 + d**2) / 2 / g
    xps = np.sqrt(xp)
    xms = np.sqrt(xm)
    Theta = np.arcsin(2 * xps * xms / (A + xp + xm))
    tanT = np.tan(Theta)
    tanT2 = np.tan(Theta / 2)
    den = 32 * g * xp * xm
    yp1 = (A * (tanT * (xms * tanT2 - xps))**2) / den
    ym1 = (A * (tanT * (xps * tanT2 - xms))**2) / den
    tanT2_2 = tanT2**2
    p_start = 1 if skip_first else 0
    p_end = int((maxorder + 1) / 2)
    sz = maxorder - 1 if skip_first else maxorder + 1
    Ys = np.empty((sz, *xp.shape))
    yp1_idx = p_end - 1 if skip_first else p_end
    for p in range(p_start, p_end):
        k = tanT2_2**p
        if skip_first:
            Ys[yp1_idx + p - 1] = yp1 * k
            Ys[yp1_idx - p] = ym1 * k
        else:
            Ys[yp1_idx + p] = yp1 * k
            Ys[yp1_idx - p - 1] = ym1 * k
    return Ys

def fit_model(PPp, PPm, g, d, Lp, Lm, G, Noise, maxorder, skip_first=False):
    xp = PPp / Lp
    xm = PPm / Lm
    res = G * (model(xp, xm, g, d, maxorder, skip_first) + Noise)
    return res

def fit_model_log(PPp_l, PPm_l, g, d, Lp_l, Lm_l, G_l, Noise_l, maxorder, skip_first=False):
    xp = np.power(10, (PPp_l - Lp_l) / 10)
    xm = np.power(10, (PPm_l - Lm_l) / 10)
    Noise = 10**(Noise_l / 10)
    # print(Noise)
    res = G_l + 10 * np.log10(model(xp, xm, g, d, maxorder, skip_first) + Noise)
    # print(res)
    return res

def fitting_procedure_log(data, d_idx, maxorder, skip_first=False):
    PPp_log, PPm_log = np.meshgrid(data['Source power, dBm'], data['Probe power, dBm'])

    nop = len(data['data'][0, 0])
    c_idx = int(np.floor(nop/2))
    if skip_first:
        idc = np.array([c_idx - i * d_idx for i in range(maxorder, 1, -2)] +
            [c_idx + i * d_idx for i in range(3, maxorder + 1, 2)], dtype=int)
    else:
        idc = np.linspace(c_idx - maxorder * d_idx,
            c_idx + maxorder * d_idx,
            maxorder + 1, dtype=int)
    peaks_log = np.transpose(data['data'][:, :, idc], axes=[2, 0, 1])

    weights = np.empty_like(peaks_log)
    p_start = 1 if skip_first else 0
    p_end = int((maxorder + 1) / 2)
    yp1_idx = p_end - 1 if skip_first else p_end
    for p in range(p_start, p_end):
        if skip_first:
            weights[yp1_idx + p - 1] = get_weights_matrix_for_order(data, d_idx, 2 * p + 1)
            weights[yp1_idx - p] = get_weights_matrix_for_order(data, d_idx, -(2 * p + 1))

        else:
            weights[yp1_idx + p] = get_weights_matrix_for_order(data, d_idx, 2 * p + 1)
            weights[yp1_idx - p - 1] = get_weights_matrix_for_order(data, d_idx, -(2 * p + 1))

    def residuals(args):
        Lp = args[0]
        Lm = args[1]
        G = args[2]
        Noise = args[3]
        g = args[4]
        d = args[5]
        f_vals = fit_model_log(PPp_log, PPm_log, g, d,
            Lp, Lm, G, Noise, maxorder, skip_first)
        R = (f_vals - peaks_log) * weights
        return R.ravel()

    initial_guess = np.array([0, 0, 0, 0, 1.0, 0.])
    res = opt.least_squares(residuals, initial_guess,
                            bounds=(np.array([-200, -200, -100, -300, 0.5, -10.]),
                                    np.array([200, 200, 200, 0, 5., 10.]))
                            )
    return res
