import numpy as np
import matplotlib.pyplot as plt


def prob(A, B, p, S):
    return p


def eo_optimiser_tmax(max_T_ind, probs, default_rate, scores, prob_mass, FP_con, TP_con, prob_func=prob):
    sols = np.ones((len(probs), int(max_T_ind))) * 10
    best_FP = np.ones((len(probs), int(max_T_ind)))
    best_TP = np.ones((len(probs), int(max_T_ind)))
    best_TN = np.ones((len(probs), int(max_T_ind)))
    best_FN = np.ones((len(probs), int(max_T_ind)))
    acc = np.ones((len(probs), int(max_T_ind)))
    profit = np.ones((len(probs), int(max_T_ind)))

    # for all thresholds below profit maximiser
    for T in range(0, max_T_ind):

        # for all probabilities
        for p in range(0, len(probs)):
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            profit_temp = 0

            # tally each probability of Y=1 at each score level
            for i in range(0, len(scores)):
                if scores[i] >= scores[max_T_ind]:
                    TP += (1 - default_rate[i]) * prob_mass[i]
                    FP += default_rate[i] * prob_mass[i]

                elif scores[T] <= scores[i] < scores[max_T_ind]:
                    probability = prob_func(scores[T], scores[max_T_ind], probs[p], scores[i])
                    TP += probability * (1 - default_rate[i]) * prob_mass[i]
                    FP += probability * default_rate[i] * prob_mass[i]
                    TN += (1 - probability) * default_rate[i] * prob_mass[i]
                    FN += (1 - probability) * (1 - default_rate[i]) * prob_mass[i]

                else:
                    TN += default_rate[i] * prob_mass[i]
                    FN += (1 - default_rate[i]) * prob_mass[i]

                profit_temp += (TP - (82 / 18 * FP))

            # check EO rates
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            TNR = TN / (TN + FP)
            FNR = FN / (FN + TP)
            acc[p, T] = (TP + TN) / (FP + TP + FN + TN)
            best_FP[p, T] = FPR
            best_TP[p, T] = TPR
            best_FN[p, T] = FNR
            best_TN[p, T] = TNR
            sols[p, T] = np.max([np.abs(TPR - TP_con), np.abs(FPR - FP_con)])
            profit[p, T] = profit_temp

    ind = np.unravel_index(np.argmin(sols, axis=None), sols.shape)
    best_FP = best_FP[ind]
    best_TP = best_TP[ind]
    best_prob = probs[ind[0]]
    best_T = scores[ind[1]]
    acc = acc[ind]
    profit = profit[ind]

    return best_T, best_prob, best_FP, best_TP, best_FN, best_TN, sols, ind, acc, profit


def eo_optimiser(probs, default_rate, scores, prob_mass, FP_con, TP_con, prob_func=prob, eps=0.05, delta=1,
                 force=False):
    best_TP = np.zeros((len(probs), len(scores), len(scores)))
    best_FP = np.zeros((len(probs), len(scores), len(scores)))
    best_TN = np.zeros((len(probs), len(scores), len(scores)))
    best_FN = np.zeros((len(probs), len(scores), len(scores)))
    best_prob = 0
    best_T0 = 0
    best_T1 = 100
    sols = np.ones((len(probs), len(scores), len(scores))) * 10
    acc = np.zeros((len(probs), len(scores), len(scores)))
    profit = np.ones((len(probs), len(scores), len(scores)))

    for T1 in range(0, len(scores)):
        for T0 in range(0, T1):
            # for all probabilities
            for p in range(0, len(probs)):
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                profit_temp = 0

                # tally each probability of Y=1 at each score level
                for i in range(0, len(scores)):
                    if scores[i] >= scores[T1]:
                        TP += (1 - default_rate[i]) * prob_mass[i]
                        FP += default_rate[i] * prob_mass[i]

                    elif scores[T0] <= scores[i] < scores[T1]:
                        probability = prob_func(scores[T0], scores[T1], probs[p], scores[i])
                        TP += probability * (1 - default_rate[i]) * prob_mass[i]
                        FP += probability * default_rate[i] * prob_mass[i]
                        TN += (1 - probability) * default_rate[i] * prob_mass[i]
                        FN += (1 - probability) * (1 - default_rate[i]) * prob_mass[i]

                    else:
                        TN += default_rate[i] * prob_mass[i]
                        FN += (1 - default_rate[i]) * prob_mass[i]

                    profit_temp += (TP - (82/18 * FP))

                # check EO rates
                TPR = TP / (TP + FN)
                FPR = FP / (FP + TN)
                TNR = TN / (TN + FP)
                FNR = FN / (FN + TP)
                if force:
                    change = scores[T0] + (1 - probs[p]) * (scores[T1] - scores[T0])
                    inflex = (-(np.sqrt(75 * probs[p]**2 - 75*probs[p] + 19) - 15*probs[p] + 7) /
                              (30*probs[p] - 15)) * (scores[T1] - scores[T0]) + scores[T0]
                    max_slope1 = scores[T0] + (change - scores[T0]) / 2
                    max_slope2 = scores[T1] - (scores[T1] - change) / 2
                    LB_diff1 = prob_func(scores[T0], scores[T1], probs[p], scores[T0] + delta)
                    UB_diff1 = 1 - prob_func(scores[T0], scores[T1], probs[p], scores[T1] - delta)
                    LB_diff2 = np.abs(prob_func(scores[T0], scores[T1], probs[p], change) -
                                      prob_func(scores[T0], scores[T1], probs[p], change + delta))
                    UB_diff2 = np.abs(prob_func(scores[T0], scores[T1], probs[p], change) -
                                      prob_func(scores[T0], scores[T1], probs[p], change - delta))
                    LB_diff3 = np.abs(prob_func(scores[T0], scores[T1], probs[p], max_slope1) -
                                      prob_func(scores[T0], scores[T1], probs[p], max_slope1 + delta))
                    UB_diff3 = np.abs(prob_func(scores[T0], scores[T1], probs[p], max_slope2) -
                                      prob_func(scores[T0], scores[T1], probs[p], max_slope2 - delta))
                    LB_diff4 = np.abs(prob_func(scores[T0], scores[T1], probs[p], inflex) -
                                      prob_func(scores[T0], scores[T1], probs[p], inflex + delta))
                    UB_diff4 = np.abs(prob_func(scores[T0], scores[T1], probs[p], inflex) -
                                      prob_func(scores[T0], scores[T1], probs[p], inflex - delta))
                    if (LB_diff1 < eps and UB_diff1 < eps and LB_diff2 < eps and UB_diff2 < eps and
                        LB_diff3 < eps and UB_diff3 < eps and LB_diff4 < eps and UB_diff4 < eps):
                        sols[p, T0, T1] = np.abs(FPR - FP_con) + np.abs(TPR - TP_con)
                        best_FP[p, T0, T1] = FPR
                        best_TP[p, T0, T1] = TPR
                        best_FN[p, T0, T1] = FNR
                        best_TN[p, T0, T1] = TNR
                        acc[p, T0, T1] = (TP + TN) / (FP + TP + FN + TN)
                        profit[p, T0, T1] = profit_temp
                else:
                    sols[p, T0, T1] = np.max([np.abs(FPR - FP_con), np.abs(TPR - TP_con)])
                    best_FP[p, T0, T1] = FPR
                    best_TP[p, T0, T1] = TPR
                    best_FN[p, T0, T1] = FNR
                    best_TN[p, T0, T1] = TNR
                    acc[p, T0, T1] = (TP + TN) / (FP + TP + FN + TN)
                    profit[p, T0, T1] = profit_temp

    ind = np.unravel_index(np.argmin(sols, axis=None), sols.shape)
    ind_test = np.argwhere(sols <= 0.005)
    temp = -np.infty
    index = 0
    for i in ind_test:
        if profit[i[0], i[1], i[2]] > temp:
            index = (i[0], i[1], i[2])
            temp = acc[index]
    ind = ind
    best_FP = best_FP[ind]
    best_TP = best_TP[ind]
    best_FN = best_FN[ind]
    best_TN = best_TN[ind]
    best_prob = probs[ind[0]]
    best_T1 = scores[ind[2]]
    best_T0 = scores[ind[1]]
    acc = acc[ind]
    profit = profit[ind]

    if np.min(sols) > 1:
        print("No viable solutions for individual fairness constraints")

    return best_T0, best_T1, best_prob, best_FP, best_TP, best_FN, best_TN, sols, ind, acc, profit


def plot_solution_space(sols0, lim, filename, save=False, show=False):
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    plt.imshow(sols0, aspect='auto', extent=[0, 1, lim, 0])
    plt.xlabel('Probability')
    plt.ylabel('Lower Thresholds')
    ax.set_title('Step Solution Space')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    if save:
        plt.savefig(filename)
    if show:
        plt.show()
