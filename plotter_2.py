import numpy as np
import equations.prob_funcs as funcs
import equations.fairness_optimiser as fairness
import sys
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data = np.genfromtxt('data/transrisk_performance_by_race_ssa.csv', delimiter=',')[1:, :]
cdf = np.genfromtxt('data/transrisk_cdf_by_race_ssa.csv', delimiter=',')[1:, :]
totals = np.genfromtxt('data/totals.csv', delimiter=',')[1:, 1:]
total = np.sum(totals)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.fill_between(data[:, 0], data[:, 2]/100, alpha=0.5, color=colors[0], label='Default Prob for Black Population')
plt.fill_between(data[:, 0], data[:, 1]/100, alpha=0.25, color=colors[1], label='Default Prob for White Population')
plt.vlines(22.5, -1, 2, linestyle='--', color=colors[0], label='Thresholds for Black Population')
plt.vlines(36.5, -1, 2, linestyle='--', color=colors[0])
plt.vlines(37.5, -1, 2, linestyle='--', color=colors[1], label='Thresholds for White Population')
plt.vlines(94.5, -1, 2, linestyle='--', color=colors[1])
plt.hlines(0.36, 22.5, 36.5, linestyle='-', color=colors[0], label='Probability for Black Population')
plt.hlines(0, 0, 22.5, linestyle='-', color=colors[0])
plt.hlines(1, 36.5, 100, linestyle='-', color=colors[0])
plt.hlines(0.82, 37.5, 94.5, linestyle='-', color=colors[1], label='Probability for White Population')
plt.hlines(0, 0, 37.5, linestyle='-', color=colors[1])
plt.hlines(1, 94.5, 100, linestyle='-', color=colors[1])
plt.xlim([0, 100])
plt.ylim([0, 1])
plt.xlabel(r'Credit Score $R$')
plt.ylabel('Probability')
plt.legend(loc='center right')
plt.title('Probability of Recieving a Loan and Probability of Defaulting')
plt.show()