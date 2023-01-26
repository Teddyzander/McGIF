import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import equations.prob_funcs as funcs


# get data
data = np.genfromtxt('data/transrisk_performance_by_race_ssa.csv', delimiter=',')[1:, :]
cdf = np.genfromtxt('data/transrisk_cdf_by_race_ssa.csv', delimiter=',')[1:, :]
totals = np.genfromtxt('data/totals.csv', delimiter=',')[1:, 1:]

# get raw number of individuals back from data set
raw_data = np.copy(data)
for i in range(0, len(data[:, 1])):
    if i == 0:
        raw_data[i, 1:] = np.round(np.multiply(cdf[i, 1:] / 100, totals[0, :]))

    else:
        raw_data[i, 1:] = np.round(np.multiply((cdf[i, 1:] - cdf[i - 1, 1:]) / 100, totals[0, :]))

# convert back to raw credit scores
lower = 336
upper = 843
data[:, 0] = ((data[:, 0] / 100) * (upper - lower)) + lower

# save probabilities and thresholds
p = [0.21, 0.19, 1, 0.98]
# equalised odds thresholds
T_0 = [521, 508, 603, 649]
T_1 = [672, 588, 603, 683]
"""# max profit thresholds
T_0 = [520, 564, 550.25, 509]
T_1 = [520, 564, 550.25, 509]
# global thresholds
T_0 = [620, 620, 620, 620]
T_1 = [620, 620, 620, 620]"""

# calculate equalised odds
pos = np.zeros(4)
neg = np.zeros(4)
fp = np.zeros(4)
tp = np.zeros(4)
fn = np.zeros(4)
tn = np.zeros(4)
phi_fp = np.zeros(4)
phi_tp = np.zeros(4)
phi_fn = np.zeros(4)
phi_tn = np.zeros(4)

# errors
err = np.zeros(4)
phi_err = np.zeros(4)

# forgive how horrendous this is
for j in range(1, 5):
    for i in range(0, len(data[:, 0])):
        pos_num = np.abs(raw_data[i, j] * (1 - (data[i, j] / 100)))
        neg_num = np.abs(raw_data[i, j] * data[i, j] / 100)
        pos[j - 1] += pos_num
        neg[j - 1] += neg_num
        if data[i, 0] < T_0[j - 1]:
            fn[j - 1] += pos_num
            tn[j - 1] += neg_num
            phi_fn[j - 1] += pos_num
            phi_tn[j - 1] += neg_num
            err[j - 1] += pos_num
            phi_err[j - 1] += pos_num
        if T_0[j - 1] <= data[i, 0] < T_1[j - 1]:
            # fixed p values
            tp[j - 1] += pos_num * p[j - 1]
            fp[j - 1] += neg_num * p[j - 1]
            fn[j - 1] += pos_num * (1 - p[j - 1])
            tn[j - 1] += neg_num * (1 - p[j - 1])

            # phi values
            phi_p = funcs.phi_quad2(T_0[j - 1], T_1[j - 1], p[j - 1], data[i, 0])
            phi_tp[j - 1] += pos_num * phi_p
            phi_fp[j - 1] += neg_num * phi_p
            phi_fn[j - 1] += pos_num * (1 - phi_p)
            phi_tn[j - 1] += neg_num * (1 - phi_p)

            # errors
            err[j - 1] += neg_num * p[j - 1] + pos_num * (1 - p[j - 1])
            phi_err[j - 1] += neg_num * phi_p + pos_num * (1 - phi_p)
        if data[i, 0] >= T_1[j - 1]:
            # fixed p values
            tp[j - 1] += pos_num
            fp[j - 1] += neg_num

            # phi values
            phi_tp[j - 1] += pos_num
            phi_fp[j - 1] += neg_num

            # errors
            err[j - 1] += neg_num
            phi_err[j - 1] += neg_num

# calculate FP and TP rate
print("Order of races: White, Black, Hispanic, Asian")
print("True positive rates for fixed p: {}".format(np.divide(tp, tp + fn)))
print("True positive rates for phi: {}".format(np.divide(phi_tp, phi_tp + phi_fn)))
print("False positive rates for fixed p: {}".format(np.divide(fp, fp + tn)))
print("False positive rates for phi: {}".format(np.divide(phi_fp, phi_fp + phi_tn)))
print("Error for fixed p: {}".format(err / totals))
print("Error for phi: {}".format(phi_err / totals))
print("Total Error rate: p: {} \t phi: {}".format(np.sum(err) / np.sum(totals), np.sum(phi_err) / np.sum(totals)))

print("\nImprovement (false positives are 82/18 times as expensive as true positives are profitable):")
print("Profit from fixed p per group: {} \t total: {}".format(tp - 82 / 18 * fp, np.sum(tp - 82 / 18 * fp)))
print("Profit from phi per group: {} \t total: {}".format(phi_tp - 82 / 18 * phi_fp, np.sum(phi_tp - 82 / 18 * phi_fp)))
print("percent of max profit: {} \t Average Group Fairness loss: {}".
      format((np.sum(phi_tp - 82 / 18 * phi_fp) / np.sum(tp - 82 / 18 * fp)) * 80.2,
             (np.max([np.mean(np.divide(tp, tp + fn) -
                              np.divide(phi_tp, phi_tp + phi_fn)),
                      np.mean(np.divide(fp, fp + tn) -
                              np.divide(phi_fp, phi_fp + phi_tn))]))))

# measure individual fairness by generating an extended data set using distributions from collected data
score_probs = np.zeros((len(data[:, 0]), 4))
total = np.sum(totals)
race_probs = (np.array(totals / total)[0])
score_probs[0, :] = cdf[0, 1:] / 100
for i in range(1, len(data[:, 0])):
    score_probs[i, :] = (cdf[i, 1:] - cdf[i - 1, 1:]) / 100

# generate data of size N and create K perturbations
N = 1000000
K = 100
synth_data = np.zeros((N, 2))
synth_data[:, 0] = np.random.choice([0, 1, 2, 3], p=race_probs, size=len(synth_data[:, 0]))
i_fairness = np.zeros(4)
phi_i_fairness = np.zeros(4)
white_count = 0
black_count = 0
hisp_count = 0
asian_count = 0
for i in range(0, len(synth_data[:, 0])):
    race = int(synth_data[i, 0])
    synth_data[i, 1] = np.random.choice(data[:, 0], p=score_probs[:, race])
    if synth_data[i, 1] < T_0[race]:
        dec = 0
        phi_dec = 0
    elif T_0[race] <= synth_data[i, 1] < T_1[race]:
        dec = np.random.choice([0, 1], p=[1 - p[race], p[race]])
        phi_p = funcs.phi_quad2(T_0[race], T_1[race], p[race], synth_data[i, 1])
        phi_dec = np.random.choice([0, 1], p=[1 - phi_p, phi_p])
    else:
        dec = 1
        phi_dec = 1

    for j in range(0, K):
        pert = np.random.uniform(-13.75, 13.75, 1)[0]
        if race == 0:
            white_count += 1
        elif race == 1:
            black_count += 1
        elif race == 2:
            hisp_count += 1
        elif race == 3:
            asian_count += 1

        if synth_data[i, 1] + pert < T_0[race]:
            dec_pert = 0
            phi_dec_pert = 0
        elif T_0[race] <= synth_data[i, 1] + pert < T_1[race]:
            dec_pert = np.random.choice([0, 1], p=[1-p[race], p[race]])
            phi_p = funcs.phi_quad2(T_0[race], T_1[race], p[race], synth_data[i, 1] + pert)
            if phi_p < 0:
                print('error')
            phi_dec_pert = np.random.choice([0, 1], p=[1-phi_p, phi_p])
        else:
            dec_pert = 1
            phi_dec_pert = 1

            i_fairness[race] += np.abs(dec - dec_pert)
            phi_i_fairness[race] += np.abs(phi_dec - phi_dec_pert)

counts = np.array([white_count, black_count, hisp_count, asian_count])
print("\nGlobal Individual fairness for fixed p: {}".format(np.sum(i_fairness) / (N * K)))
print("Global Individual fairness for phi: {}".format(np.sum(phi_i_fairness) / (N * K)))
print("Group Specific Individual fairness for fixed p: {}".format(np.divide(i_fairness, counts)))
print("Group Specific Individual fairness for phi: {}".format(np.divide(phi_i_fairness, counts)))


# plotting
titles = ['White', 'Black', 'Hispanic', 'Asian']
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(9, 6))
index = 0

# calculate probabilities of having a certain FICO score, give race
cdf_2 = np.copy(cdf)
cdf_2[1:, 1:] = cdf_2[1:, 1:] - cdf[:-1, 1:]
lim = np.amax(cdf_2[1:, 1:]) / 100
for i in range(0, 2):
    for j in range(0, 2):
        axs[i, j].plot(data[:, 0], (cdf_2[:, 1 + index]) / 100, color='lightcoral', linewidth=0.5)
        ax2 = axs[i, j].twinx()
        axs[i, j].fill_between(data[:, 0], (cdf_2[:, 1 + index]) / 100, step="pre", alpha=0.4, color='lightcoral')
        axs[i, j].vlines(T_0[index], 0, 100, color='black', linewidth=0.5, label=r'$T_{y}$', linestyles='--')
        axs[i, j].vlines(T_1[index], 0, 100, color='black', linewidth=0.5, linestyles='--')
        ax2.hlines(p[index], T_0[index], T_1[index], color='dodgerblue', linewidth=0.5,
                         label=r'$p$')
        axs[i, j].set_xlim([lower, upper])
        axs[i, j].set_ylim([0, lim])
        ax2.set_ylim([0, 1])
        axs[i, j].title.set_text(titles[index])
        if T_0[index] - T_1[index] != 0:
            x_1 = np.linspace(T_0[index], T_0[index] + (T_1[index] - T_0[index]) * (1 - p[index]), 100000)
            x_2 = np.linspace(T_0[index] + (T_1[index] - T_0[index]) * (1 - p[index]), T_1[index], 100000)
            """phi_1 = (p[index] / ((T_1[index] - T_0[index]) * (1 - p[index]))) * x_1 - \
                    (p[index] * T_0[index]) / ((T_1[index] - T_0[index]) * (1 - p[index]))
            phi_2 = ((1 - p[index]) / ((T_1[index] - T_0[index]) * p[index])) * x_2 + \
                    (1 - ((1 - p[index]) * T_1[index]) / ((T_1[index] - T_0[index]) * p[index]))"""

            phi_vec = np.vectorize(funcs.phi_quad2)
            phi_1 = phi_vec(T_0[index], T_1[index], p[index], x_1)
            phi_2 = phi_vec(T_0[index], T_1[index], p[index], x_2)

            ax2.plot(x_1, phi_1, label=r'$\phi$', linewidth=0.5, color='fuchsia')
            ax2.plot(x_2, phi_2, linewidth=0.5, color='fuchsia')

        # axs[i, j].legend(loc="upper left")
        if index == 0 or index == 2:
            axs[i, j].set_ylabel('Probability of Score')

        if index == 2 or index == 3:
            axs[i, j].set_xlabel('Credit score')

        if index == 1 or index == 3:
            ax2.set_ylabel(r'Probability of qualifying for loan')
        else:
            ax2.set_yticklabels([])

        index += 1

plt.legend()
plt.savefig('graphs/domainsmooth_probs.png')
plt.show()
