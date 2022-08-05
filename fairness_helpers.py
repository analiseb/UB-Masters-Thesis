from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def fair_metrics(dataset, y_pred):
    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred
        
    attr = dataset_pred.protected_attribute_names[0]
    
    idx = dataset_pred.protected_attribute_names.index(attr)
    privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
    unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 

    classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    result = {'statistical_parity_difference': metric_pred.statistical_parity_difference(),
             'disparate_impact': metric_pred.disparate_impact(),
             'equal_opportunity_difference': classified_metric.equal_opportunity_difference()}
        
    return result

#Validate model on given dataset and find threshold for best balanced accuracy
def validate_visualize(dataset, y_pred_proba):
    thresh_arr = np.linspace(0.01, 0.7, 50)

    bal_acc_arr = []
    disp_imp_arr = []
    avg_odds_diff_arr = []
    stat_par_diff = []
    eq_opp_diff = []
    theil_ind = []

    for thresh in tqdm(thresh_arr):
        y_validate_pred = (y_pred_proba[:,1] > thresh).astype(np.double)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_validate_pred

        classified_metric = ClassificationMetric(dataset, 
                                                        dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)

        TPR = classified_metric.true_positive_rate()
        TNR = classified_metric.true_negative_rate()
        bal_acc = 0.5*(TPR+TNR)

        acc = accuracy_score(y_true=dataset.labels,
                             y_pred=dataset_pred.labels)
        bal_acc_arr.append(bal_acc)
        avg_odds_diff_arr.append(classified_metric.average_odds_difference())
        disp_imp_arr.append(metric_pred.disparate_impact())
        stat_par_diff.append(classified_metric.statistical_parity_difference())
        eq_opp_diff.append(classified_metric.equal_opportunity_difference())
        theil_ind.append(classified_metric.theil_index())


    thresh_arr_best_ind = np.where(bal_acc_arr == np.max(bal_acc_arr))[0][0]
    thresh_arr_best = np.array(thresh_arr)[thresh_arr_best_ind]

    best_bal_acc = bal_acc_arr[thresh_arr_best_ind]
    disp_imp_at_best_bal_acc = np.abs(1.0-np.array(disp_imp_arr))[thresh_arr_best_ind]

    avg_odds_diff_at_best_bal_acc = avg_odds_diff_arr[thresh_arr_best_ind]

    stat_par_diff_at_best_bal_acc = stat_par_diff[thresh_arr_best_ind]
    eq_opp_diff_at_best_bal_acc = eq_opp_diff[thresh_arr_best_ind]
    theil_ind_at_best_bal_acc = theil_ind[thresh_arr_best_ind]
    
    #Plot balanced accuracy, abs(1-disparate impact)
    % matplotlib inline

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(thresh_arr, bal_acc_arr)
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)


    ax2 = ax1.twinx()
    ax2.plot(thresh_arr, np.abs(1.0-np.array(disp_imp_arr)), color='r')
    ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')

    ax2.axvline(np.array(thresh_arr)[thresh_arr_best_ind], 
                color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(thresh_arr, bal_acc_arr)
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)


    ax2 = ax1.twinx()
    ax2.plot(thresh_arr, avg_odds_diff_arr, color='r')
    ax2.set_ylabel('avg. odds diff.', color='r', fontsize=16, fontweight='bold')

    ax2.axvline(np.array(thresh_arr)[thresh_arr_best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)

    print("Threshold corresponding to Best balanced accuracy: %6.4f" % thresh_arr_best)
    print("Best balanced accuracy: %6.4f" % best_bal_acc)
    print("Corresponding abs(1-disparate impact) value: %6.4f" % disp_imp_at_best_bal_acc)
    print("Corresponding average odds difference value: %6.4f" % avg_odds_diff_at_best_bal_acc)
    print("Corresponding statistical parity difference value: %6.4f" % stat_par_diff_at_best_bal_acc)
    print("Corresponding equal opportunity difference value: %6.4f" % eq_opp_diff_at_best_bal_acc)
    print("Corresponding Theil index value: %6.4f" % theil_ind_at_best_bal_acc)
    return thresh_arr_best

#Evaluate performance of a given model with a given threshold on a given dataset

def validate_test(dataset, y_pred_proba, threshold):
    y_pred = (y_pred_proba[:,1] > threshold).astype(np.double)

    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred

    classified_metric = ClassificationMetric(dataset, 
                                            dataset_pred,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
    metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)

    TPR = classified_metric.true_positive_rate()
    TNR = classified_metric.true_negative_rate()
    bal_acc = 0.5*(TPR+TNR)

    acc = accuracy_score(y_true=dataset.labels,
                        y_pred=dataset_pred.labels)

    #get results
    best_bal_acc = bal_acc
    disp_imp_at_best_bal_acc = np.abs(1.0-metric_pred.disparate_impact())
    avg_odds_diff_at_best_bal_acc = classified_metric.average_odds_difference()
    stat_par_diff_at_best_bal_acc = classified_metric.statistical_parity_difference()
    eq_opp_diff_at_best_bal_acc = classified_metric.equal_opportunity_difference()
    theil_ind_at_best_bal_acc = classified_metric.theil_index()
    
    print("Threshold corresponding to Best balanced accuracy: %6.4f" % threshold)
    print("Best balanced accuracy: %6.4f" % best_bal_acc)
    print("Corresponding abs(1-disparate impact) value: %6.4f" % disp_imp_at_best_bal_acc)
    print("Corresponding average odds difference value: %6.4f" % avg_odds_diff_at_best_bal_acc)
    print("Corresponding statistical parity difference value: %6.4f" % stat_par_diff_at_best_bal_acc)
    print("Corresponding equal opportunity difference value: %6.4f" % eq_opp_diff_at_best_bal_acc)
    print("Corresponding Theil index value: %6.4f" % theil_ind_at_best_bal_acc)
    
    return {"best_bal_acc": best_bal_acc,
            "disp_imp": disp_imp_at_best_bal_acc,
            "avg_odds_diff": avg_odds_diff_at_best_bal_acc,
            "stat_par_diff": stat_par_diff_at_best_bal_acc,
            "eq_opp_diff": eq_opp_diff_at_best_bal_acc,
            "theil_ind": theil_ind_at_best_bal_acc}