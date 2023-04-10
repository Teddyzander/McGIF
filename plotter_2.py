import numpy as np
import equations.prob_funcs as funcs
import equations.fairness_optimiser as fairness
import sys
import matplotlib.pyplot as plt
import warnings
T0white = 4.0
T1white = 42.5
pwhite = 0.10
T0black = 22.5
T1black = 35.0
pblack = 0.78
width = 3

xwhite=np.linspace(T0white, T1white, 100)
ywhite = np.linspace(T0white, T1white, 100)
for i in range(0, len(ywhite)):
    ywhite[i] = funcs.phi_smooth(T0white, T1white, pwhite, xwhite[i])
    ywhite[i] = pwhite

xblack=np.linspace(T0black, T1black, 100)
yblack = np.linspace(T0black, T1black, 100)
for i in range(0, len(ywhite)):
    yblack[i] = funcs.phi_smooth(T0black, T1black, pblack, xblack[i])
    yblack[i] = pblack

warnings.filterwarnings("ignore")

data = np.genfromtxt('data/transrisk_performance_by_race_ssa.csv', delimiter=',')[1:, :]
cdf = np.genfromtxt('data/transrisk_cdf_by_race_ssa.csv', delimiter=',')[1:, :]
totals = np.genfromtxt('data/totals.csv', delimiter=',')[1:, 1:]
total = np.sum(totals)
colors = ['darkblue', 'red', 'green', 'purple'] #plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure(figsize=(10, 2), dpi=150)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
"""plt.plot(data[:, 0], 1 - data[:, 2]/100, color=colors[0],  linestyle='dashed',
         label=r'$\mathbb{P}\{Y=1\vert A=black, R=r\}$')
plt.plot(data[:, 0], 1 - data[:, 1]/100, color=colors[1], linestyle='dashed',
         label=r'$\mathbb{P}\{Y=1\vert A=white, R=r\}$')"""
plt.fill_between(data[:, 0], 1 - data[:, 2]/100, alpha=0.2, color=colors[0], label=r'repayment probability for black')
plt.fill_between(data[:, 0], 1 - data[:, 1]/100, alpha=0.2, color=colors[1], label=r'repayment probability for white')
plt.vlines(T0black, -1, 2, linestyle='dotted', color=colors[0], label=r'thresholds for black', linewidth=width)
plt.vlines(T1black, -1, 2, linestyle='dotted', color=colors[0], linewidth=width)
plt.vlines(T0white, -1, 2, linestyle='dotted', color=colors[1], label=r'thresholds for white', linewidth=width)
plt.vlines(T1white, -1, 2, linestyle='dotted', color=colors[1], linewidth=width)
# plt.hlines(pblack, T0black, T1black, linestyle='-', color=colors[0], label=r'approval probability for black', alpha=1, linewidth=width)
plt.plot(xblack, yblack, linestyle='-', color=colors[0], label=r'approval probability for black', alpha=1, linewidth=width)
plt.hlines(0, 0, T0black, linestyle='-', color=colors[0], alpha=1, linewidth=width)
plt.hlines(1, T1black, 100, linestyle='-', color=colors[0], alpha=1, linewidth=width)
# plt.hlines(pwhite, T0white, T1white, linestyle='-', color=colors[1], label=r'approval probability for white', alpha=1, linewidth=width/1.5)
plt.plot(xwhite, ywhite, linestyle='-', color=colors[1], label=r'approval probability for white', alpha=1, linewidth=width/1.5)
plt.hlines(0, 0, T0white, linestyle='-', color=colors[1], alpha=1, linewidth=width/1.5)
plt.hlines(1, T1white, 100, linestyle='-', color=colors[1], alpha=1, linewidth=width/1.5)
plt.xlim([0, 100])
plt.ylim([-0.005, 1.01])

plt.legend(loc='lower right', fancybox=True, framealpha=0.2)

plt.show()
