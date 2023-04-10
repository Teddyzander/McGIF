import numpy as np
import equations.prob_funcs as funcs
import equations.fairness_optimiser as fairness
import sys
import matplotlib.pyplot as plt
import warnings

T0s = [22.5, 17.0, 15.0, 17.0, 16.0]
T1s = [35.0, 31.5, 35.5, 33.0, 36.0]
probs = [0.78, 0.44, 0.51, 0.49, 0.54]
curves = [0, funcs.phi, funcs.phi_quad, funcs.phi_cube, funcs.phi_smooth]
scores = np.linspace(0, 100, 201)
labels = ['Fixed', 'Linear', 'Quadratic', 'Cubic', r'$4^{th}$ Order']
nrow = 5
ncol = 1;
fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=True, figsize=(4/1.4, 6/1.4), dpi=180)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(0, len(T0s)):
    ytilde = np.zeros(len(scores))
    if i == 0:
        axs[i].vlines(T0s[i], 0, 1, color='r', linestyles='--', linewidth=0.5)
        axs[i].vlines(T1s[i], 0, 1, color='r', linestyles='--', linewidth=0.5)
        axs[i].hlines(0, 0, T0s[i], color='k', linestyles='-', linewidth=0.5)
        axs[i].hlines(probs[i], T0s[i], T1s[i], color='k', linestyles='-', label=labels[i], linewidth=0.5)
        axs[i].hlines(1, T1s[i], 100, color='k', linestyles='-', linewidth=0.5)
    else:
        for j in range(0, len(scores)):
            ytilde[j] = curves[i](T0s[i], T1s[i], probs[i], scores[j])
        axs[i].vlines(T0s[i], 0, 1, color='r', linestyles='--', linewidth=0.5)
        axs[i].vlines(T1s[i], 0, 1, color='r', linestyles='--', linewidth=0.5)
        axs[i].plot(scores, ytilde, color='k', label=labels[i], linewidth=0.5)
        diffs = np.abs(np.diff(ytilde))
        print('max diff: {}'.format(np.max(diffs)*100))
    axs[i].legend(loc='lower right', fancybox=True, framealpha=0.2)
    axs[i].yaxis.set_tick_params(labelleft=False)
    axs[i].spines[['right', 'top']].set_visible(False)
    axs[i].set_yticks([])

plt.xlim([0, 100])
plt.ylim([0, 1.00001])
plt.xlabel(r'Credit Scores $r$')
fig.text(0.04, 0.5, r'$\mathbb{P}\{f(X)=1|A=black, R=r\}$', va='center', rotation='vertical')
plt.show()
