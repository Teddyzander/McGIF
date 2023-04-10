import numpy as np
import equations.prob_funcs as funcs
import equations.fairness_optimiser as fairness
import sys
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class Tee:
    def write(self, *args, **kwargs):
        self.out1.write(*args, **kwargs)
        self.out2.write(*args, **kwargs)

    def __init__(self, out1, out2):
        self.out1 = out1
        self.out2 = out2

    def flush(self):
        pass


# class to equalise
A = 0
# density of solution space
grid = 21
# brute force best solution?
brute = True
# individual fairness constraints
delta = 0.5  # greatest change in score
epsilon = 0.05  # greatest change in prob
force = True  # enforce individual fairness constraints
# save plots ?
save = True
# show plots ?
show = False
# compas or credit risk?

# save print statements to log file
filename = "TEST" + str(A) + "_with_max_acc_and_ifair"
sys.stdout = Tee(open("logs/" + filename + ".txt", "w"), sys.stdout)

# get data
data = np.genfromtxt('data/transrisk_performance_by_race_ssa.csv', delimiter=',')[1:, :]
cdf = np.genfromtxt('data/transrisk_cdf_by_race_ssa.csv', delimiter=',')[1:, :]
totals = np.genfromtxt('data/totals.csv', delimiter=',')[1:, 1:]
total = np.sum(totals)

# get raw number of individuals back from data set
raw_data = np.copy(data)
for i in range(0, len(data[:, 1])):
    if i == 0:
        raw_data[i, 1:] = cdf[i, 1:] / 100

    else:
        raw_data[i, 1:] = (cdf[i, 1:] - cdf[i - 1, 1:]) / 100

lower = 0
upper = 100
data[:, 0] = ((data[:, 0] / 100) * (upper - lower)) + lower

# calculate ROC curves
TPR = np.zeros((len(data[:, 0]), 4))
FPR = np.zeros((len(data[:, 0]), 4))
TNR = np.zeros((len(data[:, 0]), 4))
FNR = np.zeros((len(data[:, 0]), 4))
distance = np.zeros((len(data[:, 0]), 4))

for T in range(0, len(data[:, 0])):
    for j in range(1, 5):
        FP = 0
        TP = 0
        FN = 0
        TN = 0
        for i in range(0, len(data[:, 0])):
            if data[i, 0] < data[T, 0]:
                FN += ((100 - data[i, j]) / 100) * (raw_data[i, j])
                TN += ((data[i, j]) / 100) * (raw_data[i, j])
            else:
                TP += ((100 - data[i, j]) / 100) * (raw_data[i, j])
                FP += ((data[i, j]) / 100) * (raw_data[i, j])
        TPR[T, j - 1] = TP / (TP + FN)
        FPR[T, j - 1] = FP / (FP + TN)
        TNR[T, j - 1] = TN / (TN + FP)
        FNR[T, j - 1] = FN / (FN + TP)
        distance[T, j - 1] = np.sqrt((0-FPR[T, j - 1])**2 + (1-TPR[T, j - 1])**2)

dist = 0
for i in range(0,4):
    j = np.argmin(distance[:, i])
    testf = FPR[j, i]
    testt = TPR[j, i]
    if np.sqrt((0-testf)**2 + (1-testt)**2) > dist:
        dist = np.sqrt((0-testf)**2 + (1-testt)**2)
        FP_con = testf
        TP_con = testt

style = ["dashed", "dotted", "dashdot", "solid"]
groups = ["white", "black", "Hispanic", "Asian"]
colours = ['darkblue', 'red', 'green', 'purple']
x = np.linspace(0, 1, len(FPR[:, 0]))
y = np.linspace(1, 0, len(FPR[:, 0]))
plt.figure(figsize=(5, 5), dpi=150)
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
for a in range(0, 4):
    plt.plot(FPR[:, a], TPR[:, a], linewidth=1, linestyle=style[a], label=r"$A={}$".format(groups[a]), color=colours[a])
    plt.fill_between(FPR[:, a], TPR[:, a], alpha=0.15, color=colours[a])
    plt.fill_between(x,x,color="white",alpha=1)

plt.plot(x, x, linestyle='--', color='black', linewidth=0.5)
# plt.plot(x, y, linestyle='--', color='black', linewidth=0.5)
#plt.xlabel(r"$\mathbb{P}\{\hat{Y}=1|Y=0, A=a\}$")
#plt.ylabel(r"$\mathbb{P}\{\hat{Y}=1|Y=1, A=a\}$")
plt.plot([FP_con], [TP_con], marker="+", markersize=15, markeredgecolor="black",
         markerfacecolor="black",
         label="Optimal EO", mew=3, )
plt.xlim([FP_con-0.05, FP_con+0.05])
plt.ylim([TP_con-0.05, TP_con+0.05])
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.legend(loc='lower center', fancybox=True, framealpha=0.2)
if save:
    plt.savefig("graphs/ROCcurvefull")
plt.show()
# find maximum profit threshold for each group
max_profit_T = np.ones(4) * 1000
max_profit_T_ind = np.ones(4)
min_rate = np.ones(4) * 1000

for i in range(0, len(data[:, 0])):
    for j in range(1, 5):
        diff = np.abs(18 - data[i, j])
        if diff < min_rate[j - 1]:
            max_profit_T[j - 1] = data[i, 0]
            max_profit_T_ind[j - 1] = i
            min_rate[j - 1] = diff

# calculate confusion matrix for groups at max profit
TPs = np.zeros(4)
FPs = np.zeros(4)
TNs = np.zeros(4)
FNs = np.zeros(4)
accs = np.zeros(4)
profits = np.zeros(4)

for j in range(1, 5):
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    profit_temp = 0
    for i in range(0, len(data[:, 0])):
        if data[i, 0] < max_profit_T[j - 1]:
            FN += ((100 - data[i, j]) / 100) * (raw_data[i, j])
            TN += ((data[i, j]) / 100) * (raw_data[i, j])
        else:
            TP += ((100 - data[i, j]) / 100) * (raw_data[i, j])
            FP += ((data[i, j]) / 100) * (raw_data[i, j])
            profit_temp += TP - 82 / 18 * FP
    TPs[j - 1] = TP / (TP + FN)
    FPs[j - 1] = FP / (FP + TN)
    TNs[j - 1] = TN / (TN + FP)
    FNs[j - 1] = FN / (FN + TP)
    accs[j - 1] = (TP + TN) / (FP + TP + FN + TN)
    profits[j - 1] = profit_temp

# find the most profitable group
# find the best performing group (unweighted by cost sensitivity)


# fix threshold for best performing group, and find mix of thresholds and probabilities to minimise equalised odds
# we do this under maximal profits conditions
FP_con = np.max(FPs)
TP_con = np.max(TPs)
if brute:
    FP_con = 0.1826192
    TP_con = 0.817381

print(" ----------------------------------------- \n"
      "+                                         +\n"
      "+       EQUALISED ODDS FOR CLASS {}        +\n"
      "+_________________________________________+\n".format(A))
print("Solution space size: {}".format(grid))
print("False positive: {}".format(np.round(FPs[A], 8)))
print("True positives: {}".format(np.round(TPs[A], 8)))
print("Target False positive: {}".format(np.round(FP_con, 8)))
print("Target True positives: {}".format(np.round(TP_con, 8)))
print("Max profit thresholds: {}".format(max_profit_T[A]))
print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(FPs[A] - FP_con), 8),
                                                   np.round(np.abs(TPs[A] - TP_con), 8),
                                                   np.round(np.max(np.abs([FPs[A] - FP_con, TPs[A] - TP_con])), 8)))
print("Accuracy: {}%".format(np.round(100 * accs[A], 4)))
print("Percent of max profit: {}".format(100))
print("------------------------------------------\n")

# fixed random parameter
probs = np.linspace(0, 1, num=grid)
best_TP = np.copy(TPs)
best_FP = np.copy(FPs)
best_prob = np.zeros(4)

eps = 1*10**(-10)

if brute:
    best_T0, best_T1, best_prob, best_FP, \
    best_TP, best_FN, best_TN, sols0, ind, \
    acc, profit = fairness.eo_optimiser(probs, data[:, A + 1] / 100, data[:, 0],
                                        raw_data[:, A + 1], FP_con, TP_con,
                                        eps=epsilon,
                                        delta=delta)
    print("Fixed probability (Hardt et al)")
    if best_prob > epsilon or 1 - best_prob > epsilon:
        print("Violates individual fairness constraints")
    print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                                  np.round(best_TP, 8),
                                                                  best_T0, best_T1,
                                                                  np.round(best_prob, 8)))
    print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                       np.round(np.abs(best_TP - TP_con), 8),
                                                       np.round(sols0[ind], 8)))
    print("Accuracy: {}%".format(np.round(100 * acc, 4)))
    lips1 = best_prob / eps
    lips2 = (1 - best_prob) / eps
    lips = np.max([lips1, lips2])
    print("Lipschitz Constant: {} (theoretically infinite)".format(lips))
    """print("Percent of max profit: {}%".format(np.round(100 * profit /
                                                       (profits[A]), 8)))"""
    print("------------------------------------------")

"""best_T, best_prob, best_FP, best_TP, \
best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser_tmax(int(max_profit_T_ind[A]), probs,
                                                                       data[:, A + 1] / 100, data[:, 0],
                                                                       raw_data[:, A + 1], FP_con, TP_con)

fairness.plot_solution_space(sols0, max_profit_T[A], "graphs/stepspace" + filename, save=save, show=show)

print("Fixed probability (Hardt et al) and max profit top threshold")
if best_prob > epsilon or 1 - best_prob > epsilon:
    print("Violates individual fairness constraints")
print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                              np.round(best_TP, 8),
                                                              best_T, np.round(max_profit_T[A], 8),
                                                              np.round(best_prob, 8)))
print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                   np.round(np.abs(best_TP - TP_con), 8),
                                                   np.round(sols0[ind], 8)))
print("Accuracy: {}%".format(np.round(100 * acc, 4)))
# print("Percent of max profit: {}%".format(np.round(100 * profit /
#                                                    (profits[A]), 8)))
print("------------------------------------------")"""

if brute:
    best_T0, best_T1, best_prob, best_FP, \
    best_TP, best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser(probs, data[:, A + 1] / 100, data[:, 0],
                                                                               raw_data[:, A + 1], FP_con, TP_con,
                                                                               funcs.phi,
                                                                               eps=epsilon, delta=delta, force=force)
    print("Linear function")
    print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                                  np.round(best_TP, 8),
                                                                  best_T0, best_T1,
                                                                  np.round(best_prob, 8)))
    print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                       np.round(np.abs(best_TP - TP_con), 8),
                                                       np.round(sols0[ind], 8)))

    lip1 = np.abs(
        funcs.phi(best_T0, best_T1, best_prob, best_T0) - funcs.phi(best_T0, best_T1, best_prob, best_T0 + eps)) / eps
    lip2 = np.abs(
        funcs.phi(best_T0, best_T1, best_prob, best_T1) - funcs.phi(best_T0, best_T1, best_prob, best_T1 - eps)) / eps

    lip = np.max([lip1, lip2])
    print("Lipschitz Constant: {}".format(np.round(lip, 6)))
    print("Accuracy: {}%".format(np.round(100 * acc, 4)))
    """print("Percent of max profit: {}%".format(np.round(100 * profit /
                                                       (profits[A]), 8)))"""
    print("------------------------------------------")

"""best_T, best_prob, best_FP, best_TP, \
best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser_tmax(int(max_profit_T_ind[A]), probs,
                                                                       data[:, A + 1] / 100,
                                                                       data[:, 0], raw_data[:, A + 1], FP_con, TP_con,
                                                                       funcs.phi)

fairness.plot_solution_space(sols0, max_profit_T[A], "graphs/linspace" + filename, save=save, show=show)

print("linear function with max profit top threshold")
print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                              np.round(best_TP, 8),
                                                              best_T, np.round(max_profit_T[A], 8),
                                                              np.round(best_prob, 8)))
print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                   np.round(np.abs(best_TP - TP_con), 8),
                                                   np.round(sols0[ind], 8)))
print("Accuracy: {}%".format(np.round(100 * acc, 4)))
 #print("Percent of max profit: {}%".format(np.round(100 * profit /
 #                                                   (profits[A]), 8)))
print("------------------------------------------")"""

if brute:
    best_T0, best_T1, best_prob, best_FP, \
    best_TP, best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser(probs, data[:, A + 1] / 100, data[:, 0],
                                                                               raw_data[:, A + 1], FP_con, TP_con,
                                                                               funcs.phi_quad,
                                                                               eps=epsilon, delta=delta, force=force)
    print("Smooth boundary function")
    print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                                  np.round(best_TP, 8),
                                                                  best_T0, best_T1,
                                                                  np.round(best_prob, 8)))
    print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                       np.round(np.abs(best_TP - TP_con), 8),
                                                       np.round(sols0[ind], 8)))
    tau = best_T0 + (1 - best_prob) * (best_T1 - best_T0)
    lip1 = np.abs(
        funcs.phi_quad(best_T0, best_T1, best_prob, tau) - funcs.phi_quad(best_T0, best_T1, best_prob, tau + eps)) / eps
    lip2 = np.abs(
        funcs.phi_quad(best_T0, best_T1, best_prob, tau) - funcs.phi_quad(best_T0, best_T1, best_prob, tau - eps)) / eps

    lip = np.max([lip1, lip2])
    print("Lipschitz Constant: {}".format(np.round(lip, 6)))
    print("Accuracy: {}%".format(np.round(100 * acc, 4)))
    """print("Percent of max profit: {}%".format(np.round(100 * profit /
                                                       (profits[A]), 8)))"""
    print("------------------------------------------")

"""best_T, best_prob, best_FP, best_TP, \
best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser_tmax(int(max_profit_T_ind[A]), probs,
                                                                       data[:, A + 1] / 100, data[:, 0],
                                                                       raw_data[:, A + 1], FP_con, TP_con,
                                                                       funcs.phi_quad)

fairness.plot_solution_space(sols0, max_profit_T[A], "graphs/quadpspace" + filename, save=save, show=show)

print("smooth boundary function with max profit top threshold")
print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                              np.round(best_TP, 8),
                                                              best_T, np.round(max_profit_T[A], 8),
                                                              np.round(best_prob, 8)))
print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                   np.round(np.abs(best_TP - TP_con), 8),
                                                   np.round(sols0[ind], 8)))
print("Accuracy: {}%".format(np.round(100 * acc, 4)))
# print("Percent of max profit: {}%".format(np.round(100 * profit /
#                                                    (profits[A]), 8)))
print("------------------------------------------")"""

if brute:
    best_T0, best_T1, best_prob, best_FP, \
    best_TP, best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser(probs, data[:, A + 1] / 100, data[:, 0],
                                                                               raw_data[:, A + 1], FP_con, TP_con,
                                                                               funcs.phi_cube,
                                                                               eps=epsilon, delta=delta, force=force)
    print("Pieceswise smooth function")
    print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                                  np.round(best_TP, 8),
                                                                  best_T0, best_T1,
                                                                  np.round(best_prob, 8)))
    print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                       np.round(np.abs(best_TP - TP_con), 8),
                                                       np.round(sols0[ind], 8)))
    tau = best_T0 + (1 - best_prob) * (best_T1 - best_T0)
    tau1 = best_T0 + (tau - best_T0) / 2
    tau2 = tau + (best_T1 - tau) / 2
    lip1 = np.abs(
        funcs.phi_cube(best_T0, best_T1, best_prob, tau1) - funcs.phi_cube(best_T0, best_T1, best_prob, tau1 + eps)) / eps
    lip2 = np.abs(
        funcs.phi_cube(best_T0, best_T1, best_prob, tau2) - funcs.phi_cube(best_T0, best_T1, best_prob, tau2 - eps)) / eps

    lip = np.max([lip1, lip2])
    print("Lipschitz Constant: {}".format(np.round(lip, 6)))
    print("Accuracy: {}%".format(np.round(100 * acc, 4)))
    """print("Percent of max profit: {}%".format(np.round(100 * profit /
                                                       (profits[A]), 8)))"""
    print("------------------------------------------")

best_T, best_prob, best_FP, best_TP, \
best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser_tmax(int(max_profit_T_ind[A]), probs,
                                                                       data[:, A + 1] / 100,
                                                                       data[:, 0], raw_data[:, A + 1], FP_con, TP_con,
                                                                       funcs.phi_cube)

"""fairness.plot_solution_space(sols0, max_profit_T[A], "graphs/cubespace" + filename, save=save, show=show)

print("piecewise smooth function with max profit top threshold")
print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                              np.round(best_TP, 8),
                                                              best_T, np.round(max_profit_T[A], 8),
                                                              np.round(best_prob, 8)))
print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                   np.round(np.abs(best_TP - TP_con), 8),
                                                   np.round(sols0[ind], 8)))
print("Accuracy: {}%".format(np.round(100 * acc, 4)))
# print("Percent of max profit: {}%".format(np.round(100 * profit /
#                                                   (profits[A]), 8)))
print("------------------------------------------")"""

probs = np.linspace(2 / 5, 3 / 5, num=grid)
best_TP = TPs[A]
best_FP = FPs[A]
best_prob = 0
best_T0 = max_profit_T[A]
best_T1 = max_profit_T[A]

if brute:
    best_T0, best_T1, best_prob, best_FP, \
    best_TP, best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser(probs, data[:, A + 1] / 100, data[:, 0],
                                                                               raw_data[:, A + 1], FP_con, TP_con,
                                                                               funcs.phi_smooth, eps=epsilon,
                                                                               delta=delta, force=force)
    print("Smooth function p->[0.4, 0.6]")
    print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                                  np.round(best_TP, 8),
                                                                  best_T0, best_T1,
                                                                  np.round(best_prob, 8)))
    print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                       np.round(np.abs(best_TP - TP_con), 8),
                                                       np.round(sols0[ind], 8)))
    print("Accuracy: {}%".format(np.round(100 * acc, 4)))
    """print("Percent of max profit: {}%".format(np.round(100 * profit /
                                                       (profits[A]), 8)))"""
    inflex = (-(np.sqrt(75 * best_prob**2 - 75*best_prob + 19) - 15*best_prob + 7) /
                              (30*best_prob - 15)) * (best_T1 - best_T0) + best_T0
    lip1 = np.abs(
        funcs.phi_smooth(best_T0, best_T1, best_prob, inflex) - funcs.phi_smooth(best_T0, best_T1, best_prob,
                                                                           inflex + eps)) / eps
    lip2 = np.abs(
        funcs.phi_smooth(best_T0, best_T1, best_prob, inflex) - funcs.phi_smooth(best_T0, best_T1, best_prob,
                                                                           inflex - eps)) / eps

    lip = np.max([lip1, lip2])
    print("Lipschitz Constant: {}".format(np.round(lip, 6)))
    print("------------------------------------------")

"""probs = np.linspace(2 / 5, 3 / 5, num=grid)
best_T, best_prob, best_FP, best_TP, \
best_FN, best_TN, sols0, ind, acc, profit = fairness.eo_optimiser_tmax(int(max_profit_T_ind[A]), probs,
                                                                       data[:, A + 1] / 100,
                                                                       data[:, 0], raw_data[:, A + 1], FP_con, TP_con,
                                                                       funcs.phi_smooth)

fairness.plot_solution_space(sols0, max_profit_T[A], "graphs/smoothspace" + filename, save=save, show=show)

print("smooth function p->[0.4,0.6] with max profit top threshold")
print("FP: {} TP: {} Lower T: {} Upper T: {} prob: {}".format(np.round(best_FP, 8),
                                                              np.round(best_TP, 8),
                                                              best_T, np.round(max_profit_T[A], 8),
                                                              np.round(best_prob, 8)))
print("Differences: FP ({}) TP ({}) EO({})".format(np.round(np.abs(best_FP - FP_con), 8),
                                                   np.round(np.abs(best_TP - TP_con), 8),
                                                   np.round(sols0[ind], 8)))
print("Accuracy: {}%".format(np.round(100 * acc, 4)))
# print("Percent of max profit: {}%".format(np.round(100 * profit /
#                                                    (profits[A]), 8)))
print("------------------------------------------")"""

print("stop")
