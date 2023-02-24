#from cmd2 import style
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.ticker import (AutoLocator, AutoMinorLocator,
                               FormatStrFormatter, MultipleLocator)
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition, inset_axes,
                                                   mark_inset)
from scipy.interpolate import interp1d


class Plotters():
    '''
    Utilities class to handle all plots.
    '''
    def __init__(self):
        pass

    def plot_property_model(self, property: str):

        model = pickle.load(open(f'../../models/{property}.pkl', 'rb'))

        model_data = pickle.load(
            open(f'../../models/{property}_data.pkl', 'rb'))

        fig1 = plt.figure(figsize=(6, 5))
        axs1 = fig1.add_subplot(1, 1, 1)

        X_train = model_data[0]
        X_test = model_data[1]
        y_train = model_data[2]
        y_test = model_data[3]

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        r_score = model.score(X_test, y_test)

        axs1.scatter(y_pred_train,
                     y_train,
                     s=50,
                     marker='s',
                     c='b',
                     label=r'$Training$')
        axs1.scatter(y_pred_test,
                     y_test,
                     s=50,
                     marker='s',
                     c='r',
                     label=r'$Testing$')
        axs1.plot(
            [min(min(y_train), min(y_test)),
             max(max(y_train), max(y_test))],
            [min(min(y_train), min(y_test)),
             max(max(y_train), max(y_test))], 'k')

        axs1.annotate(f'$R^{2} = {"{:.2f}".format(r_score)}$', (20, 10),
                      fontfamily='serif',
                      fontweight='bold')

        axs1.set_xlabel(r'$\kappa_{measured} (\rm mS\ cm^{-1})$',
                        labelpad=5,
                        fontsize='x-large')
        axs1.set_ylabel(r'$\kappa_{predicted} (\rm mS\ cm^{-1})$',
                        labelpad=5,
                        fontsize='x-large')

        axs1.yaxis.set_major_locator(AutoLocator())
        axs1.yaxis.set_minor_locator(AutoMinorLocator())
        axs1.xaxis.set_major_locator(AutoLocator())
        axs1.xaxis.set_minor_locator(AutoMinorLocator())
        axs1.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs1.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs1.spines[axis].set_linewidth(1.0)

        axs1.set_ylim(min(min(y_train), min(y_test)),
                      max(max(y_train), max(y_test)))
        axs1.set_xlim(min(min(y_train), min(y_test)),
                      max(max(y_train), max(y_test)))
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        plt.legend(loc='best',
                   ncol=1,
                   fontsize='medium',
                   fancybox=False,
                   frameon=False)
        plt.tight_layout()
        plt.savefig(f'../../reports/figures/edi_{property}.pdf')
        plt.savefig(f'../../reports/figures/edi_{property}.png',
                    transparent=True)
        plt.show()

        pass

    def plot_cd_edi(self, time, j_exp, j_model):

        fig0 = plt.figure(figsize=(6, 5))
        axs0 = fig0.add_subplot(1, 1, 1)

        axs0.plot(time, j_model, marker='*', color='r', label='Model')
        axs0.plot(time, j_exp, marker='*', color='b', label='Experiment')

        axs0.set_xlabel(r'$Time\ (\rm minutes)$',
                        labelpad=5,
                        fontsize='x-large')
        axs0.set_ylabel(r'$j\ (\rm A\ m^{-2})$',
                        labelpad=5,
                        fontsize='x-large')

        axs0.yaxis.set_major_locator(AutoLocator())
        axs0.yaxis.set_minor_locator(AutoMinorLocator())
        axs0.xaxis.set_major_locator(AutoLocator())
        axs0.xaxis.set_minor_locator(AutoMinorLocator())
        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        for tick in axs0.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs0.get_yticklabels():
            tick.set_fontname("Times New Roman")

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        plt.legend(loc='best',
                   ncol=1,
                   fontsize='medium',
                   fancybox=False,
                   frameon=False)
        plt.tight_layout()
        plt.savefig(f'../../reports/figures/edi_cd.pdf')
        plt.savefig(f'../../reports/figures/edi_cd.png', transparent=True)
        plt.show()

        return None

    def plot_Emem_edi(self, time, Emem):

        fig0 = plt.figure(figsize=(6, 5))
        axs0 = fig0.add_subplot(1, 1, 1)

        axs0.plot(time, Emem, marker='*', color='r', label='Model')

        axs0.set_xlabel(r'$Time\ (\rm minutes)$',
                        labelpad=5,
                        fontsize='x-large')
        axs0.set_ylabel(r'$E_{mem}\ (\rm V)$', labelpad=5, fontsize='x-large')

        axs0.yaxis.set_major_locator(AutoLocator())
        axs0.yaxis.set_minor_locator(AutoMinorLocator())
        axs0.xaxis.set_major_locator(AutoLocator())
        axs0.xaxis.set_minor_locator(AutoMinorLocator())
        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        for tick in axs0.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs0.get_yticklabels():
            tick.set_fontname("Times New Roman")

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        plt.legend(loc='best',
                   ncol=1,
                   fontsize='medium',
                   fancybox=False,
                   frameon=False)
        plt.tight_layout()
        plt.savefig(f'../../reports/figures/edi_Emem.pdf')
        plt.savefig(f'../../reports/figures/edi_Emem.png', transparent=True)
        plt.show()

        return None

    def plot_concentration(self, data_exp, Cc_model, Cd_model):
        fig0 = plt.figure(figsize=(6, 5))
        axs0 = fig0.add_subplot(1, 1, 1)

        axs0.scatter(data_exp[:, 0], Cc_model, marker='s', color='r')
        axs0.plot(data_exp[:, 0], data_exp[:, 0], 'k')
        axs0.set_xlabel(r'$C_{measured}\ (\rm ppm)$',
                        labelpad=5,
                        fontsize='x-large')
        axs0.set_ylabel(r'$C_{predicted}\ (\rm ppm)$',
                        labelpad=5,
                        fontsize='x-large')

        axs0.yaxis.set_major_locator(AutoLocator())
        axs0.yaxis.set_minor_locator(AutoMinorLocator())
        axs0.xaxis.set_major_locator(AutoLocator())
        axs0.xaxis.set_minor_locator(AutoMinorLocator())
        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        #plt.legend(labels, loc='best', ncol = 1, fontsize = 'medium', fancybox=False, frameon = False)
        plt.tight_layout()
        plt.show()

        fig1 = plt.figure(figsize=(6, 5))
        axs1 = fig1.add_subplot(1, 1, 1)
        axs1.scatter(data_exp[:, 1], Cd_model, marker='s', color='b')
        axs1.plot(data_exp[:, 1], data_exp[:, 1], 'k')

        axs1.set_xlabel(r'$C_{measured}\ (\rm ppm)$',
                        labelpad=5,
                        fontsize='x-large')
        axs1.set_ylabel(r'$C_{predicted}\ (\rm ppm)$',
                        labelpad=5,
                        fontsize='x-large')

        axs1.yaxis.set_major_locator(AutoLocator())
        axs1.yaxis.set_minor_locator(AutoMinorLocator())
        axs1.xaxis.set_major_locator(AutoLocator())
        axs1.xaxis.set_minor_locator(AutoMinorLocator())
        axs1.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs1.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs1.spines[axis].set_linewidth(1.0)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        #plt.legend(labels, loc='best', ncol = 1, fontsize = 'medium', fancybox=False, frameon = False)
        plt.tight_layout()

        plt.show()

        return None

    def plot_mass_balance(self, data_exp, wsol, t):
        fig0 = plt.figure(figsize=(6, 5))
        axs0 = fig0.add_subplot(1, 1, 1)

        labels = [
            r'$C_{conc}$', r'$C_{dil}$', r'$C_{dil}^{exp}$',
            r'$C_{conc}^{exp}$'
        ]

        time = data_exp[:, 0]

        cdil_smooth = interp1d(np.asanyarray(t) / 60, wsol[:, 0])
        cconc_smooth = interp1d(np.asanyarray(t) / 60, wsol[:, 1])

        cdil = []
        cconc = []

        for t_i in time:
            cdil.append(cdil_smooth(t_i).item())
            cconc.append(cconc_smooth(t_i).item())

        plt.scatter(time, data_exp[:, 1], marker='s', color='b')
        plt.scatter(time, data_exp[:, 2], marker='s', color='g')
        plt.plot(time, np.asarray(cdil) * 0.058, marker='*', color='b')
        plt.plot(time, np.asarray(cconc) * 0.058, marker='*', color='g')

        axs0.set_xlabel(r'$Time\ (\rm h)$', labelpad=5, fontsize='x-large')
        axs0.set_ylabel(r'$Concentration\ (\rm 1000\ ppm\ Cl^{-})$',
                        labelpad=5,
                        fontsize='x-large')

        axs0.yaxis.set_major_locator(AutoLocator())
        axs0.yaxis.set_minor_locator(AutoMinorLocator())
        axs0.xaxis.set_major_locator(AutoLocator())
        axs0.xaxis.set_minor_locator(AutoMinorLocator())
        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        for tick in axs0.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs0.get_yticklabels():
            tick.set_fontname("Times New Roman")

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        plt.legend(labels,
                   loc='best',
                   ncol=1,
                   fontsize='medium',
                   fancybox=False,
                   frameon=False)
        plt.tight_layout()
        plt.savefig(f'../../reports/figures/edi_mass_balance.pdf')
        plt.savefig(f'../../reports/figures/edi_mass_balance.png',
                    transparent=True)
        plt.show()

        return None

    def plot_multi_mass_balance(self, wsol_list, t):
        fig0 = plt.figure(figsize=(6, 5))
        axs0 = fig0.add_subplot(1, 1, 1)

        labels = [
            r'$Mixed\ resin\ with\ PE\ binder$',
            r'$Mixed\ resin\ with\ CEI$',
            r'$AER\ with\ CEI\ binder$',
        ]

        colors = ['g', 'k', 'purple']

        for i, sol in enumerate(wsol_list):
            axs0.plot(np.asarray(t) / 60, sol[:, 0] * 0.058, color=colors[i])
            axs0.plot(np.asarray(t) / 60, sol[:, 1] * 0.058, color=colors[i])

        axs0.set_xlabel(r'$Time\ (\rm min)$', labelpad=5, fontsize='x-large')
        axs0.set_ylabel(r'$Concentration\ (\rm 1000\ ppm\ Cl^{-})$',
                        labelpad=5,
                        fontsize='x-large')

        axs0.yaxis.set_major_locator(AutoLocator())
        axs0.yaxis.set_minor_locator(AutoMinorLocator())
        axs0.xaxis.set_major_locator(AutoLocator())
        axs0.xaxis.set_minor_locator(AutoMinorLocator())
        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        for tick in axs0.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs0.get_yticklabels():
            tick.set_fontname("Times New Roman")

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        plt.legend(labels,
                   loc='best',
                   ncol=1,
                   fontsize='medium',
                   fancybox=False,
                   frameon=False)
        plt.tight_layout()
        plt.savefig(f'../../reports/figures/edi_multi_mass_balance.pdf')
        plt.savefig(f'../../reports/figures/edi_multi_mass_balance.png',
                    transparent=True)
        plt.show()

    def plot_current_coupled(self, data_exp, j_model, t):
        fig1 = plt.figure(figsize=(6, 5))
        axs1 = fig1.add_subplot(1, 1, 1)

        labels = [r'$j_{model}$', r'$j_{experimental}$']

        plt.scatter(np.asanyarray(t) / 3600,
                    j_model / 10,
                    marker='s',
                    s=20,
                    color='b')
        plt.plot(data_exp[:, 0] / 3600, data_exp[:, 3] / 10, 'b')

        axs1.set_xlabel(r'$Time\ (\rm h)$', labelpad=5, fontsize='x-large')
        axs1.set_ylabel(r'$Current\ density\ (\rm mA\ cm^{-2})$',
                        labelpad=5,
                        fontsize='x-large')

        axs1.yaxis.set_major_locator(AutoLocator())
        axs1.yaxis.set_minor_locator(AutoMinorLocator())
        axs1.xaxis.set_major_locator(AutoLocator())
        axs1.xaxis.set_minor_locator(AutoMinorLocator())
        axs1.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs1.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs1.spines[axis].set_linewidth(1.0)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        plt.legend(labels,
                   loc='best',
                   ncol=1,
                   fontsize='medium',
                   fancybox=False,
                   frameon=False)
        plt.tight_layout()

        plt.show()

        return None
