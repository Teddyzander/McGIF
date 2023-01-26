import numpy as np
import equations.prob_funcs as funcs
import equations.fairness_optimiser as fairness
import sys
import matplotlib.pyplot as plt
import warnings

T0s = [48.5, 45.5, 43.5, 44.5, 45.0]
T1s = [49.0, 51.5, 54.0, 52.5, 53.0]
probs = [0.74, 0.49, 0.52, 0.49, 0.558]
curves = [0, funcs.phi, funcs.phi_quad, funcs.phi_cube, funcs.phi_smooth]
scores = np.linspace(0, 100, 201)
labels = ['Step', 'Continuous', 'Boundary-smooth', 'Piecewise-smooth', 'Smooth']
nrow = 5
ncol = 1;
fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(0, len(T0s)):
    ytilde = np.zeros(len(scores))
    if i == 0:
        axs[i].vlines(T0s[i], 0, 1, color='k', linestyles='--', linewidth=0.5)
        axs[i].vlines(T1s[i], 0, 1, color='k', linestyles='--', linewidth=0.5)
        axs[i].hlines(0, 0, T0s[i], color='k', linestyles='-', linewidth=0.5)
        axs[i].hlines(probs[i], T0s[i], T1s[i], color='k', linestyles='-', label=labels[i], linewidth=0.5)
        axs[i].hlines(1, T1s[1], 1, color='k', linestyles='-', linewidth=0.5)
    else:
        for j in range(0, len(scores)):
            ytilde[j] = curves[i](T0s[i], T1s[i], probs[i], scores[j])
        axs[i].vlines(T0s[i], 0, 1, color='k', linestyles='--', linewidth=0.5)
        axs[i].vlines(T1s[i], 0, 1, color='k', linestyles='--', linewidth=0.5)
        axs[i].plot(scores, ytilde, color='k', label=labels[i], linewidth=0.5)
        diffs = np.abs(np.diff(ytilde))
        print('max diff: {}'.format(np.max(diffs)*100))
    axs[i].legend(loc='lower right')

plt.xlim([0, 100])
plt.ylim([0, 1.00001])
plt.xlabel(r'Scores $R$')
fig.text(0.04, 0.5, r'$\mathbb{P}\{\tilde{Y}=1|A=Asian\}$', va='center', rotation='vertical')
fig.text(0.25, 0.9, 'Asian Class Probability Curves for Equalised Odds', va='center')
plt.show()
