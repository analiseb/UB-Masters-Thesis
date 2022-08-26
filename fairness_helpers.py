from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
from aif360.sklearn import metrics as mt
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler
from aif360.sklearn import metrics as mt
import global_variables as gv
import utilities
from IPython.display import Markdown, display

import keras
import tensorflow as tf
from tensorflow.keras.optimizers  import Adam, Adagrad, SGD, RMSprop

algo_names = ['baseline', 'preprocessing', 'inprocessing', 'postprocessing']
protected_attribute_names = ['sex-binary', 'race-binary', 'race-grouped', 'age-binary']

thresh_arr = np.linspace(0.01, 0.5, 100)

model = keras.models.load_model('saved_models/mlp_binary_1.h5')
model.compile(loss='categorical_hinge',
              optimizer=SGD(learning_rate=0.0005),
              metrics=['acc',tf.keras.metrics.AUC(), tf.keras.metrics.Recall()])


def get_aif360_data(colab=False):
    """
    configure datasets according to aif360 library
        step 1.  Binarize the sensitive attribute: 1 set to privileged group, 0 to unprivileged group
        step 2. Binarize the label columns: 1 is the positive outcome and 0 else
        step 3. Set the sensitive attribute as index
    """

    if colab:
        df = pd.read_csv('/content/gdrive/MyDrive/UB-Masters-Thesis/data/binary_full.csv')
    else:
        df = pd.read_csv('data/binary_full.csv')
    pd.set_option('display.max_columns', None)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df['sex-binary']=df['sex'].map(gv.binary_sex)
    df['race-grouped']=df['race'].map(gv.alternate_race_groupings).map(gv.race_groupings_encoded)

    protected_attribute_names = ['sex-binary', 'race-binary', 'race-grouped', 'age-binary']

    input_cols = df.iloc[:,:61].columns.to_list()
    X1 = df.loc[:,input_cols+['CVD']+['sex-binary', 'race-binary', 'race-grouped']]

    X1['age-binary'] = np.where((df['age']>=50)&(df['age']<70),1,0)

    X = X1.set_index(protected_attribute_names, drop=False)

    X_transformed = utilities.transform_features(X, 'CVD', norm_method=RobustScaler()) # excluding protected attribute cols
    X_transformed = X_transformed.reset_index(drop=True)

    return X, X_transformed

def confusion_eval2(predictions, X_test, y_test, threshold=0.5):
    """
    Takes in model, testing data, and decision threshold and outputs prediction performance (confusion matrix,
    sensitivity, specificity, and accuracy)
    """
    y_predicted = (predictions >= threshold)

    conf_mat = confusion_matrix(y_test, y_predicted)
    print(conf_mat)
    total = sum(sum(conf_mat))
    sensitivity = conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[1, 0])
    specificity = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1])/total
    bal_accuracy = balanced_accuracy_score(y_test, y_predicted)
    print('specificity : ', specificity)
    print('sensitivity : ', sensitivity)
    print('accuracy : ', accuracy)
    print('balanced accuracy: ', bal_accuracy)
    return conf_mat, sensitivity, specificity, accuracy, bal_accuracy

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

def get_fairness(df, protected_attribute, threshold=0.5, report=True):

    _, _, X_test, _, _, y_test = utilities.process_features(df, 'CVD', RobustScaler(), one_hot=True)
    
    y_prob = model.predict(X_test)
    y_pred = np.where(y_prob > threshold, 1,0)
    
    if report:
        baseline_fairness = fairness_report(y_test, y_pred, prot_attr=protected_attribute)
        return baseline_fairness
    else:
        return y_pred

def fairness_report(y_test, y_pred, prot_attr, return_df=True):
    fairness_report_dict = {
        prot_attr: {"disparate_impact_ratio": 
                        mt.disparate_impact_ratio(y_test, y_pred, prot_attr=prot_attr),
                    "statistical_parity_difference": 
                        mt.statistical_parity_difference(y_test, y_pred, prot_attr=prot_attr),
                    "equal_opportunity_difference": 
                        mt.equal_opportunity_difference(y_test, y_pred, prot_attr=prot_attr),
                    "average_odds_difference": 
                        mt.average_odds_difference(y_test, y_pred, prot_attr=prot_attr),


                   }
    }
    if return_df:
        return pd.DataFrame(fairness_report_dict)
    else:
        return fairness_report_dict

def get_att_privilege_groups(protected_attribute):
    privileged_group = [{protected_attribute: 1}]
    unprivileged_group = [{protected_attribute: 0}]
    return privileged_group, unprivileged_group

def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics

# def test_lr_model(y_data_pred_prob, dataset, thresh_arr, protected_attribute):
#     y_pred = (y_data_pred_prob[:,1] > thresh_arr).astype(np.double)
#     dataset_pred = dataset.copy()
#     dataset_pred.labels = y_pred

#     privileged_group, unprivileged_group = get_att_privilege_groups(protected_attribute)

#     classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_group, privileged_group)
#     metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_group, privileged_group)
#     return dataset_pred.labels, classified_metric, metric_pred, fontweight='bold')

#     ax2.axvline(np.array(thresh_arr)[thresh_arr_best_ind], 
#                 color='k', linestyle=':')
#     ax2.yaxis.set_tick_params(labelsize=14)
#     ax2.grid(True)

#     fig, ax1 = plt.subplots(figsize=(8,4))
#     ax1.plot(thresh_arr, bal_acc_arr)#Validate model on given dataset and find threshold for best balanced accuracy

def validate_visualize(dataset, y_pred_proba, attribute, metric='balanced_accuracy'):
    thresh_arr = np.linspace(0.01, 0.9, 50) # check with other

    privileged_group, unprivileged_group =  get_att_privilege_groups(attribute)
   
    TPR_arr = []
    TNR_arr = []
    bal_acc_arr = []
    disp_imp_arr = []
    avg_odds_diff_arr = []
    stat_par_diff = []
    eq_opp_diff = []
    theil_ind = []
    
    full_results = pd.DataFrame(columns = ['threshold', 'sensitivity', 'specificity', 'Balanced Accuracy', 'Avg Odds Difference', 'Disparate Impact', 'Statistical Parity Difference', 'Equal Opportunity Difference', 'Theil Index'])
    dataset = StandardDataset(dataset, 
                          label_name='CVD', 
                          favorable_classes=[1], 
                          protected_attribute_names=[attribute],  #attribute = 'sex-binary'
                          privileged_classes=[[1]])

    for thresh in thresh_arr:
        y_validate_pred = (y_pred_proba> thresh).astype(np.double)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_validate_pred

        classified_metric = ClassificationMetric(dataset, 
                                                 dataset_pred,
                                                 unprivileged_groups=unprivileged_group,
                                                 privileged_groups=privileged_group)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                               unprivileged_groups=unprivileged_group,
                                               privileged_groups=privileged_group)

        TPR = classified_metric.true_positive_rate()
        TNR = classified_metric.true_negative_rate()
        bal_acc = 0.5*(TPR+TNR)

        acc = accuracy_score(y_true=dataset.labels,
                             y_pred=dataset_pred.labels)
        bal_acc_arr.append(bal_acc)
        TPR_arr.append(TPR)
        TNR_arr.append(TNR)
        avg_odds_diff_arr.append(classified_metric.average_odds_difference())
        disp_imp_arr.append(metric_pred.disparate_impact())
        stat_par_diff.append(classified_metric.statistical_parity_difference())
        eq_opp_diff.append(classified_metric.equal_opportunity_difference())
        theil_ind.append(classified_metric.theil_index())

        df_new = pd.DataFrame({'threshold':thresh, 'sensitivity':TPR, 'specificity':TNR, 'Balanced Accuracy': bal_acc, 'Avg Odds Difference':classified_metric.average_odds_difference(), 'Disparate Impact':metric_pred.disparate_impact(), 'Statistical Parity Difference':classified_metric.statistical_parity_difference(), 'Equal Opportunity Difference':classified_metric.equal_opportunity_difference(), 'Theil Index':classified_metric.theil_index()}, index=[0])
        full_results = pd.concat([full_results,df_new])

    thresh_arr_best_ind = np.where(bal_acc_arr == np.max(bal_acc_arr))[0][0]
    thresh_arr_best = np.array(thresh_arr)[thresh_arr_best_ind]

    best_bal_acc = bal_acc_arr[thresh_arr_best_ind]
    disp_imp_at_best_bal_acc = np.abs(1.0-np.array(disp_imp_arr))[thresh_arr_best_ind]

    avg_odds_diff_at_best_bal_acc = avg_odds_diff_arr[thresh_arr_best_ind]

    stat_par_diff_at_best_bal_acc = stat_par_diff[thresh_arr_best_ind]
    eq_opp_diff_at_best_bal_acc = eq_opp_diff[thresh_arr_best_ind]
    theil_ind_at_best_bal_acc = theil_ind[thresh_arr_best_ind]
    
    sns.set_style("darkgrid")

    if metric=='balanced_accuracy':
        # abs(1-disparate impact)
        fig, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(thresh_arr, bal_acc_arr)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)


        ax2 = ax1.twinx()
        ax2.plot(thresh_arr, np.abs(1.0-np.array(disp_imp_arr)), color='r')
        ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        plt.axhspan(0, 0.2, facecolor='0.2', alpha=0.15)


        # average odds difference
        fig, ax3 = plt.subplots(figsize=(8,4))
        ax3.plot(thresh_arr, bal_acc_arr)
        ax3.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax3.xaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)


        ax4 = ax3.twinx()
        ax4.plot(thresh_arr, avg_odds_diff_arr, color='g')
        ax4.set_ylabel('avg odds difference', color='g', fontsize=16)
        ax3.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax3.xaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)

        # statistical parity difference
        fig, ax5 = plt.subplots(figsize=(8,4))
        ax5.plot(thresh_arr, bal_acc_arr)
        ax5.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax5.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax5.xaxis.set_tick_params(labelsize=14)
        ax5.yaxis.set_tick_params(labelsize=14)


        ax6 = ax5.twinx()
        ax6.plot(thresh_arr, stat_par_diff, color='magenta')
        ax6.set_ylabel('Statistical Parity Difference', color='magenta', fontsize=16)
        ax5.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax5.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax5.xaxis.set_tick_params(labelsize=14)
        ax5.yaxis.set_tick_params(labelsize=14)

        # equal opportunity difference
        fig, ax7 = plt.subplots(figsize=(8,4))
        ax7.plot(thresh_arr, bal_acc_arr)
        ax7.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax7.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax7.xaxis.set_tick_params(labelsize=14)
        ax7.yaxis.set_tick_params(labelsize=14)

        ax8 = ax7.twinx()
        ax8.plot(thresh_arr, eq_opp_diff, color='darkorange')
        ax8.set_ylabel('Equal Opp Difference', color='darkorange', fontsize=16)
        ax7.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax7.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax7.xaxis.set_tick_params(labelsize=14)
        ax7.yaxis.set_tick_params(labelsize=14)

        # theil index
        fig, ax9 = plt.subplots(figsize=(8,4))
        ax9.plot(thresh_arr, bal_acc_arr)
        ax9.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax9.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax9.xaxis.set_tick_params(labelsize=14)
        ax9.yaxis.set_tick_params(labelsize=14)


        ax10 = ax9.twinx()
        ax10.plot(thresh_arr, theil_ind, color='black')
        ax10.set_ylabel('Theil Index', color='black', fontsize=16)
        ax9.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax9.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax9.xaxis.set_tick_params(labelsize=14)
        ax9.yaxis.set_tick_params(labelsize=14)
    

    elif metric=='sensitivity':
        # abs(1-disparate impact)
        fig, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(thresh_arr, TPR_arr)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)


        ax2 = ax1.twinx()
        ax2.plot(thresh_arr, np.abs(1.0-np.array(disp_imp_arr)), color='r')
        ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        plt.axhspan(0, 0.2, facecolor='0.2', alpha=0.15)


        # average odds difference
        fig, ax3 = plt.subplots(figsize=(8,4))
        ax3.plot(thresh_arr, TPR_arr)
        ax3.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax3.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax3.xaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)


        ax4 = ax3.twinx()
        ax4.plot(thresh_arr, avg_odds_diff_arr, color='g')
        ax4.set_ylabel('avg odds difference', color='g', fontsize=16)
        ax3.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax3.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax3.xaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)

        # statistical parity difference
        fig, ax5 = plt.subplots(figsize=(8,4))
        ax5.plot(thresh_arr, TPR_arr)
        ax5.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax5.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax5.xaxis.set_tick_params(labelsize=14)

        ax6 = ax5.twinx()
        ax6.plot(thresh_arr, stat_par_diff, color='magenta')
        ax6.set_ylabel('Statistical Parity Difference', color='magenta', fontsize=16)
        ax5.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax5.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax5.xaxis.set_tick_params(labelsize=14)
        ax5.yaxis.set_tick_params(labelsize=14)

        # equal opportunity difference
        fig, ax7 = plt.subplots(figsize=(8,4))
        ax7.plot(thresh_arr, TPR_arr)
        ax7.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax7.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax7.xaxis.set_tick_params(labelsize=14)
        ax7.yaxis.set_tick_params(labelsize=14)

        ax8 = ax7.twinx()
        ax8.plot(thresh_arr, eq_opp_diff, color='darkorange')
        ax8.set_ylabel('Equal Opp Difference', color='darkorange', fontsize=16)
        ax7.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax7.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax7.xaxis.set_tick_params(labelsize=14)
        ax7.yaxis.set_tick_params(labelsize=14)

        # theil index
        fig, ax9 = plt.subplots(figsize=(8,4))
        ax9.plot(thresh_arr, TPR_arr)
        ax9.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax9.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax9.xaxis.set_tick_params(labelsize=14)
        ax9.yaxis.set_tick_params(labelsize=14)


        ax10 = ax9.twinx()
        ax10.plot(thresh_arr, theil_ind, color='black')
        ax10.set_ylabel('Theil Index', color='black', fontsize=16)
        ax9.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax9.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
        ax9.xaxis.set_tick_params(labelsize=14)
        ax9.yaxis.set_tick_params(labelsize=14)

    elif metric=='specificity':
        # abs(1-disparate impact)
        fig, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(thresh_arr, TNR_arr)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)


        ax2 = ax1.twinx()
        ax2.plot(thresh_arr, np.abs(1.0-np.array(disp_imp_arr)), color='r')
        ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        plt.axhspan(0, 0.2, facecolor='0.2', alpha=0.15)


        # average odds difference
        fig, ax3 = plt.subplots(figsize=(8,4))
        ax3.plot(thresh_arr, TNR_arr)
        ax3.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax3.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax3.xaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)


        ax4 = ax3.twinx()
        ax4.plot(thresh_arr, avg_odds_diff_arr, color='g')
        ax4.set_ylabel('avg odds difference', color='g', fontsize=16)
        ax3.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax3.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax3.xaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)

        # statistical parity difference
        fig, ax5 = plt.subplots(figsize=(8,4))
        ax5.plot(thresh_arr, TNR_arr)
        ax5.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax5.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax5.xaxis.set_tick_params(labelsize=14)
        ax5.yaxis.set_tick_params(labelsize=14)


        ax6 = ax5.twinx()
        ax6.plot(thresh_arr, stat_par_diff, color='magenta')
        ax6.set_ylabel('Statistical Parity Difference', color='magenta', fontsize=16)
        ax5.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax5.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax5.xaxis.set_tick_params(labelsize=14)
        ax5.yaxis.set_tick_params(labelsize=14)

        # equal opportunity difference
        fig, ax7 = plt.subplots(figsize=(8,4))
        ax7.plot(thresh_arr, TNR_arr)
        ax7.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax7.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax7.xaxis.set_tick_params(labelsize=14)
        ax7.yaxis.set_tick_params(labelsize=14)

        ax8 = ax7.twinx()
        ax8.plot(thresh_arr, eq_opp_diff, color='darkorange')
        ax8.set_ylabel('Equal Opp Difference', color='darkorange', fontsize=16)
        ax7.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax7.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax7.xaxis.set_tick_params(labelsize=14)
        ax7.yaxis.set_tick_params(labelsize=14)

        # theil index
        fig, ax9 = plt.subplots(figsize=(8,4))
        ax9.plot(thresh_arr, TNR_arr)
        ax9.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax9.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax9.xaxis.set_tick_params(labelsize=14)
        ax9.yaxis.set_tick_params(labelsize=14)


        ax10 = ax9.twinx()
        ax10.plot(thresh_arr, theil_ind, color='black')
        ax10.set_ylabel('Theil Index', color='black', fontsize=16)
        ax9.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax9.set_ylabel('True Negative Rate', color='b', fontsize=16, fontweight='bold')
        ax9.xaxis.set_tick_params(labelsize=14)
        ax9.yaxis.set_tick_params(labelsize=14)
    # # TPR DI
    # fig, ax11 = plt.subplots(figsize=(8,4))
    # ax11.plot(thresh_arr, TPR_arr)
    # ax11.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    # ax11.set_ylabel('True Postive Rate', color='b', fontsize=16, fontweight='bold')
    # ax11.xaxis.set_tick_params(labelsize=14)
    # ax11.yaxis.set_tick_params(labelsize=14)


    # ax12 = ax11.twinx()
    # ax12.plot(thresh_arr, np.abs(1.0-np.array(disp_imp_arr)), color='#ff9221')
    # ax12.set_ylabel('abs(1-disparate impact)', color='#ff9221', fontsize=16)
    # ax11.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    # ax11.set_ylabel('True Positive Rate', color='b', fontsize=16, fontweight='bold')
    # ax11.xaxis.set_tick_params(labelsize=14)
    # ax11.yaxis.set_tick_params(labelsize=14)

    print("Threshold corresponding to Best balanced accuracy: %6.4f" % thresh_arr_best)
    print("Best balanced accuracy: %6.4f" % best_bal_acc)
    print("Corresponding abs(1-disparate impact) value: %6.4f" % disp_imp_at_best_bal_acc)
    print("Corresponding average odds difference value: %6.4f" % avg_odds_diff_at_best_bal_acc)
    print("Corresponding statistical parity difference value: %6.4f" % stat_par_diff_at_best_bal_acc)
    print("Corresponding equal opportunity difference value: %6.4f" % eq_opp_diff_at_best_bal_acc)
    print("Corresponding Theil index value: %6.4f" % theil_ind_at_best_bal_acc)

    return thresh_arr_best, best_bal_acc, disp_imp_at_best_bal_acc, avg_odds_diff_at_best_bal_acc, stat_par_diff_at_best_bal_acc, eq_opp_diff_at_best_bal_acc, theil_ind_at_best_bal_acc, full_results



#Evaluate performance of a given model with a given threshold on a given dataset

def validate_test(dataset, y_pred_proba, threshold):
    y_pred = (y_pred_proba > threshold).astype(np.double)

    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred

    classified_metric = ClassificationMetric(dataset, 
                                            dataset_pred,
                                            unprivileged_groups=gv.unprivileged_groups,
                                            privileged_groups=gv.privileged_groups)
    metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                            unprivileged_groups=gv.unprivileged_groups,
                                            privileged_groups=gv.privileged_groups)

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

def plot_compare_intervention():
    y_pred = postproc.predict_proba(X_test)[:, 1]
    y_lr = postproc.estimator_.predict_proba(X_test)[:, 1]
    br = postproc.postprocessor_.base_rates_
    i = X_test.index.get_level_values('sex') == 1

    plt.plot([0, br[0]], [0, 1-br[0]], '-b', label='All calibrated classifiers (Females)')
    plt.plot([0, br[1]], [0, 1-br[1]], '-r', label='All calibrated classifiers (Males)')

    plt.scatter(mt.generalized_fpr(y_test[~i], y_lr[~i]),
                mt.generalized_fnr(y_test[~i], y_lr[~i]),
                300, c='b', marker='.', label='Original classifier (Females)')
    plt.scatter(mt.generalized_fpr(y_test[i], y_lr[i]),
                mt.generalized_fnr(y_test[i], y_lr[i]),
                300, c='r', marker='.', label='Original classifier (Males)')
                                                                            
    plt.scatter(mt.generalized_fpr(y_test[~i], y_pred[~i]),
                mt.generalized_fnr(y_test[~i], y_pred[~i]),
                100, c='b', marker='d', label='Post-processed classifier (Females)')
    plt.scatter(mt.generalized_fpr(y_test[i], y_pred[i]),
                mt.generalized_fnr(y_test[i], y_pred[i]),
                100, c='r', marker='d', label='Post-processed classifier (Males)')

    plt.plot([0, 1], [mt.generalized_fnr(y_test, y_pred)]*2, '--', c='0.5')

    plt.axis('square')
    plt.xlim([0.0, 0.4])
    plt.ylim([0.3, 0.7])
    plt.xlabel('generalized fpr');
    plt.ylabel('generalized fnr');
    plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')

def get_disparity_index(di):
    return 1 - np.minimum(di, 1 / di)


def get_bal_acc(classified_metric):
    return 0.5 * (classified_metric.true_positive_rate() + classified_metric.true_negative_rate())

def get_best_bal_acc_cutoff(y_pred_prob, dataset):
    y_validate_pred_prob = y_pred_prob
    bal_acc_arr = []
    disp_imp_arr = []

    for thresh in tqdm(thresh_arr):
        y_validate_pred = (y_validate_pred_prob[:,1] > thresh).astype(np.double)
        dataset_pred = dataset.copy()
        dataset_pred.labels = y_validate_pred

        # Calculate accuracy for each threshold value
        classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_group, privileged_group)
        bal_acc = get_bal_acc(classified_metric)
        bal_acc_arr.append(bal_acc)

        # Calculate fairness for each threshold value
        metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_group, privileged_group)
        disp_imp_arr.append(metric_pred.disparate_impact())

    # Find threshold for best accuracy
    thresh_arr_best_ind = np.where(bal_acc_arr == np.max(bal_acc_arr))[0][0]
    thresh_arr_best = np.array(thresh_arr)[thresh_arr_best_ind]

    # Calculate accuracy and fairness at this threshold
    best_bal_acc = bal_acc_arr[thresh_arr_best_ind]
    disp_imp_at_best_bal_acc = disp_imp_arr[thresh_arr_best_ind]

    # Output metrics
    acc_metrics = pd.DataFrame({'thresh_arr_best_ind' : thresh_arr_best_ind, \
    'thresh_arr_best' : thresh_arr_best, \
    'best_bal_acc' : best_bal_acc, \
    'disp_imp_at_best_bal_acc' : disp_imp_at_best_bal_acc}, index=[0]).transpose()
    return acc_metrics, bal_acc_arr, disp_imp_arr, dataset_pred.labels

# post-processing
def plot_acc_vs_fairness(metric, metric_name, bal_acc_arr, thresh_arr_best_ind):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(thresh_arr, bal_acc_arr, color='b')
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14, labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(thresh_arr, metric, color='r')
    ax2.set_ylabel(metric_name, color='r', fontsize=16, fontweight='bold')
    ax2.axvline(np.array(thresh_arr)[thresh_arr_best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14, labelcolor='r')
    ax2.grid(True)

def fp_vs_fn(dataset, gamma_list, iters):
    fp_auditor = Auditor(dataset, 'FP')
    fn_auditor = Auditor(dataset, 'FN')
    fp_violations = []
    fn_violations = []
    for g in gamma_list:
        print('gamma: {} '.format(g), end =" ")
        fair_model = GerryFairClassifier(C=100, printflag=False, gamma=g, max_iters=iters)
        fair_model.gamma=g
        fair_model.fit(dataset)
        predictions = array_to_tuple((fair_model.predict(dataset)).labels)
        _, fp_diff = fp_auditor.audit(predictions)
        _, fn_diff = fn_auditor.audit(predictions)
        fp_violations.append(fp_diff)
        fn_violations.append(fn_diff)

    plt.plot(fp_violations, fn_violations, label='adult')
    plt.xlabel('False Positive Disparity')
    plt.ylabel('False Negative Disparity')
    plt.legend()
    plt.title('FP vs FN Unfairness')
    plt.savefig('gerryfair_fp_fn.png')
    plt.close()

    gamma_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]
    fp_vs_fn(dataset, gamma_list, pareto_iters)
    Image(filename='gerryfair_fp_fn.png')

def plot_calibrated_eq_odds(dataset_orig_valid_pred1, dataset_orig_test_pred1, dataset_transf_valid_pred11, dataset_transf_test_pred11, u1, p1 ):
  # Thresholds
  all_thresh = np.linspace(0.01, 0.99, 25)
  display(Markdown("#### Classification thresholds used for validation and parameter selection"))

  bef_avg_odds_diff_test = []
  bef_avg_odds_diff_valid = []
  aft_avg_odds_diff_test = []
  aft_avg_odds_diff_valid = []
  bef_bal_acc_valid = []
  bef_bal_acc_test = []
  aft_bal_acc_valid = []
  aft_bal_acc_test = []
  for thresh in tqdm(all_thresh):
      
      dataset_orig_valid_pred_thresh1 = dataset_orig_valid_pred1.copy(deepcopy=True)
      dataset_orig_test_pred_thresh1 = dataset_orig_test_pred1.copy(deepcopy=True)
      dataset_transf_valid_pred_thresh1 = dataset_transf_valid_pred11.copy(deepcopy=True)
      dataset_transf_test_pred_thresh1 = dataset_transf_test_pred11.copy(deepcopy=True)
      
      # Labels for the datasets from scores
      y_temp = np.zeros_like(dataset_orig_valid_pred_thresh1.labels)
      y_temp[dataset_orig_valid_pred_thresh1.scores >= thresh] = dataset_orig_valid_pred_thresh1.favorable_label
      y_temp[~(dataset_orig_valid_pred_thresh1.scores >= thresh)] = dataset_orig_valid_pred_thresh1.unfavorable_label
      dataset_orig_valid_pred_thresh1.labels = y_temp

      y_temp = np.zeros_like(dataset_orig_test_pred_thresh1.labels)
      y_temp[dataset_orig_test_pred_thresh1.scores >= thresh] = dataset_orig_test_pred_thresh1.favorable_label
      y_temp[~(dataset_orig_test_pred_thresh1.scores >= thresh)] = dataset_orig_test_pred_thresh1.unfavorable_label
      dataset_orig_test_pred_thresh1.labels = y_temp
      
      y_temp = np.zeros_like(dataset_transf_valid_pred_thresh1.labels)
      y_temp[dataset_transf_valid_pred_thresh1.scores >= thresh] = dataset_transf_valid_pred_thresh1.favorable_label
      y_temp[~(dataset_transf_valid_pred_thresh1.scores >= thresh)] = dataset_transf_valid_pred_thresh1.unfavorable_label
      dataset_transf_valid_pred_thresh1.labels = y_temp
      
      y_temp = np.zeros_like(dataset_transf_test_pred_thresh1.labels)
      y_temp[dataset_transf_test_pred_thresh1.scores >= thresh] = dataset_transf_test_pred_thresh1.favorable_label
      y_temp[~(dataset_transf_test_pred_thresh1.scores >= thresh)] = dataset_transf_test_pred_thresh1.unfavorable_label
      dataset_transf_test_pred_thresh1.labels = y_temp
      
      # Metrics for original validation data
      classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid1,
                                                  dataset_orig_valid_pred_thresh1,
                                                  unprivileged_groups=u1,
                                                  privileged_groups=p1)
      bef_avg_odds_diff_valid.append(classified_metric_orig_valid.equal_opportunity_difference())

      bef_bal_acc_valid.append(0.5*(classified_metric_orig_valid.true_positive_rate()+
                                classified_metric_orig_valid.true_negative_rate()))

      classified_metric_orig_test = ClassificationMetric(dataset_orig_test1,
                                                  dataset_orig_test_pred_thresh1,
                                                  unprivileged_groups=u1,
                                                  privileged_groups=p1)
      bef_avg_odds_diff_test.append(classified_metric_orig_test.equal_opportunity_difference())
      bef_bal_acc_test.append(0.5*(classified_metric_orig_test.true_positive_rate()+
                                classified_metric_orig_test.true_negative_rate()))

      # Metrics for transf validation data
      classified_metric_transf_valid = ClassificationMetric(
                                      dataset_orig_valid1, 
                                      dataset_transf_valid_pred_thresh1,
                                      unprivileged_groups=u1,
                                      privileged_groups=p1)
      aft_avg_odds_diff_valid.append(classified_metric_transf_valid.equal_opportunity_difference())
      aft_bal_acc_valid.append(0.5*(classified_metric_transf_valid.true_positive_rate()+
                                classified_metric_transf_valid.true_negative_rate()))

      # Metrics for transf testing data
      classified_metric_transf_test = ClassificationMetric(dataset_orig_test1,
                                                  dataset_transf_test_pred_thresh1,
                                                  unprivileged_groups=u1,
                                                  privileged_groups=p1)
      aft_avg_odds_diff_test.append(classified_metric_transf_test.equal_opportunity_difference())
      aft_bal_acc_test.append(0.5*(classified_metric_transf_test.true_positive_rate()+
                                    classified_metric_transf_test.true_negative_rate()))
      return classified_metric_orig_test, classified_metric_transf_test