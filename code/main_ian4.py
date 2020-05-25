################################
# Main.py                      #
# SYSC 5405 - Term Project     #
# Group 7: Decision Trees      #
# By: Mo, Jason, Ian and Ben   #
################################
# Imports
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import preprocessing as prc
import feature_selection as fs
import random
from random import seed
from random import randint
from sklearn.model_selection import *
from sklearn.tree import * 
from sklearn.metrics import * 
from sklearn.feature_selection import *
from sklearn.utils import *
from sklearn.base import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from datetime import datetime
################################
startTime = datetime.now()
bootstrap_test_count = 100
random.seed(30)
rand_state = randint(0, bootstrap_test_count)

plot_on = True
test_mode = False
################################

# Report will plot a PR curve and return the test stat
def report(name, y_true, y_pred, y_prob, verbose=False):

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    PrecisionAtRe50_DT = np.max(precision[recall>0.5])
    
    if verbose:
        print("===== " + name +"=====")
        print("TN = " + str(cm[0][0]))
        print("FP = " + str(cm[0][1]))
        print("FN = " + str(cm[1][0]))
        print("TP = " + str(cm[1][1]))
        print("Pr = " + str(cm[1][1]/(cm[1][1] + cm[0][1])))
        print("Re = " + str(cm[1][1]/(cm[1][1] + cm[1][0])))
        print("Confusion Matrix: \n" + str(cm))
        print('Pr@Re50 = ', PrecisionAtRe50_DT)
        print()
        plt.figure()
        plt.plot(recall, precision,label=name + ", Pr@Re>50 = {0:.5f}".format(PrecisionAtRe50_DT))
        plt.legend(loc=1)
        plt.title(name + " Precision Recall Curve")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
   
    return PrecisionAtRe50_DT

# plot bootstrapping holdout and apparent errors
def plot_report(bs_stats, apparent_stat, hold_out_stat):
    bs_mean = np.mean(bs_stats)
    bs_mean = np.mean(bs_stats)
    bs_std  = np.std(bs_stats)
    bs_632 = 0.632*bs_mean + 0.368*apparent_stat
    
    plt.figure()
    plt.hist(bs_stats, 20, label='Bootstrap Precision', edgecolor='black')
    ylim = plt.ylim()
    
    plt.plot(2 * [apparent_stat], ylim, ':g', linewidth=3, label="Apparent Pr@Re50 = {0:.4f}".format(apparent_stat))
    plt.plot(2 * [hold_out_stat], ylim, '-.r', linewidth=3, label="Holdout Pr@Re50 = {0:.4f}".format(hold_out_stat))
    plt.plot(2 * [bs_632], ylim, '-m', linewidth=2, label="632 Bootstrap Pr@Re50 = {0:.4f}".format(bs_632))
    plt.plot(2 * [bs_mean], ylim, '-b', linewidth=2, label="Bootstrap Pr@Re50 = {0:.4f}".format(bs_mean))
    plt.plot(2 * [bs_mean-bs_std], ylim, '--b', linewidth=2)
    plt.plot(2 * [bs_mean+bs_std], ylim, '--b', linewidth=2)
    
    plt.ylim(ylim)
    plt.legend(loc=1)
    plt.title("Precision Comparison")
    plt.xlabel('Precision')
    plt.show()
    
def plot_permutation(model, test_data, test_class):    
    cv = StratifiedKFold(2)
    score, permutation_scores, pvalue = permutation_test_score(model, test_data, test_class, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
    
    print("Classification score %s (pvalue : %s)" % (score, pvalue))
    
    # #############################################################################
    # View histogram of permutation scores
    plt.figure()
    plt.hist(permutation_scores, 20, label='Permutation scores', edgecolor='green')
    ylim = plt.ylim()
    
    #plt.plot(2 * [score], ylim, '--b', linewidth=1, label="Classification Score = {0:.4f}".format(score))
    plt.plot(2 * [1. / 2], ylim, '--k', linewidth=3, label='Luck')
    
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.show()    

# This function trains and tests a model
# Returns the predictions and their probabilty 
def test_model(model, X_train, X_test, y_train):
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[::,1]
	return y_pred, y_prob, model

def run_model(model, X_train, X_test, y_train):
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[::,1]
	return y_pred, y_prob

def eval_bs(bs_stats, apparent_stat, hold_out_stat, verbose = False):
    mean = np.mean(bs_stats)
    std  = np.std(bs_stats)
    bs_632 = 0.632*mean + 0.368*apparent_stat
    if verbose:
        print("===== Bootstrapping Results =====")
        print('Bootstrapping Estimate = ', mean)
        print('              With Std = ', std)
        print('Apparent Estimate = ', apparent_stat)
        print('Holdout Estimate = ', hold_out_stat)
        print('0.632 Bootstrapping Estimate = ', bs_632)

    return bs_632, mean, std

def bootstrap_test(name, model, X, y):
    test_stat = list()
    n_size = int(X.count()[0] * 0.5)
    all_index = [j for j in range(len(X))]
    for i in range(bootstrap_test_count):
        # Wipe out training from last itteration:
        model = clone(model)
        # Create bootstrap sub-sample
        X_b_train, y_b_train, b_idx = resample(X, y, all_index, n_samples=n_size, stratify=y, random_state = rand_state+i)
        test_idx = np.array([x for x in all_index if x not in b_idx])
        
        X_b_test = X.values[test_idx,:];
        y_b_test = y.values[test_idx];
        # Fit a model and send it to 
        pred, prob, _ = test_model(model, X_b_train, X_b_test, y_b_train)
        test_stat.append(report(name + ", bs sample " + str(i), y_b_test, pred, prob))
    return test_stat

# Main function
# Inputs: model - will be trained and validated using k-fold
# [optional] clean_data or unsupervise_fs
def main(df, df_holdout, name, model, unsupervise_fs = False, bs_estimate = False, verbose=False):
    # Don't train on ID, BEN!
    if 'id' in df:
        df = df.drop(['id'], axis=1)
        df_holdout = df_holdout.drop(['id'], axis=1)
    # Split X and y 
    X = df.drop(['class'], axis=1)
    y = df['class']
    X_holdout = df_holdout.drop(['class'], axis=1)
    y_holdout = df_holdout['class']

    if bs_estimate:
        # Take out a holdout sample for the entire test
        X = X.append(X_holdout)
        y = y.append(y_holdout)
        X, X_holdout, y, y_holdout = train_test_split(X, y, test_size = 0.2, random_state = rand_state, stratify = y)

        # Run a simple test to get optimistic apparent score
        pred, prob, trained_model = test_model(model, X, X, y)
        apparent_stat = report(name + ", Apparent Stat", y, pred, prob, verbose=plot_on)

        if test_mode:
            print(test_mode)
            df = pd.read_csv('Files/optimal_features_fake_test.csv', sep=',') 
            df = df.drop(['id'], axis=1)
            if 'class' in df:
                df = df.drop(['class'], axis=1)

            pred, prob = run_model(trained_model, df, df, df)
            test_mode_stat = report(name + ", Test Mode", y_holdout, pred, prob, verbose=True)
            df_out = pd.DataFrame(pred,columns=['Predicted Class'])
            df_out.to_csv('Files/predicted_class.csv', sep=',', mode = 'w', index=False)   
 
        # Run bootstrapping
        test_stat = bootstrap_test(name, model, X, y)

        # Train on all X, y and test against the holdout test. 
        pred, prob = run_model(trained_model, X, X_holdout, y)
        holdout_stat = report(name + ", holdout", y_holdout, pred, prob)

        # plot permutation test and precision results
        if plot_on:
            plot_permutation(trained_model, X_holdout, y_holdout)
            plot_report(test_stat, apparent_stat, holdout_stat)

        return eval_bs(test_stat, apparent_stat, holdout_stat, verbose=verbose)

    else:
        y_pred = []
        y_prob = []
        y_true = []
        kf = StratifiedKFold(n_splits=5, shuffle = True, random_state = rand_state)
        for train_index, test_index in kf.split(X, y):
            # Split train and test set
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Train and test model  
            pred, prob, _ = test_model(model, X_train, X_test, y_train)
            y_pred.extend(pred)
            y_prob.extend(prob)
            y_true.extend(y_test)

    return report(name, y_true, y_pred, y_prob, verbose=verbose)

# Evaluates the depth of the tree
def test_tree_depth(data, class_weight=None):
    test_stats = [0,0]
    for i in range(2, 16): # 2 to 15
        dt = DecisionTreeClassifier(max_depth = i, class_weight=class_weight)
        test_stats.append(main(df=data, name="DT with depth = "+str(i), model=dt))
    return test_stats


# Run the tree depth test
def run_depth_test():
    df = pd.read_csv('Files/optimal_features_pca_cos.csv', sep=',') 
    df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
    df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df).dropna(thresh=20))
    df = prc.standarize(df) # or normalize
    rslt = test_tree_depth(df)

    print("Run Time: " + str(datetime.now() - startTime))

    # Print PR Curves from test
    plt.legend(loc=1)
    plt.title("Precision Recall Curve")
    plt.show()

    # Print out the distribution of curves 
    plt.plot(list(range(2, len(rslt))), rslt[2:])
    plt.ylabel("Depth of Tree")
    plt.xlabel("Pr@Re>50")
    plt.title("Testing Decision Tree Depth")
    plt.xticks(list(range(2, len(rslt))))
    plt.show()

def run_bs_dt():
    df = pd.read_csv('Files/optimal_features_pca_cos.csv', sep=',') 
    df = df.drop(['id'], axis=1)
 
    df_holdout = pd.read_csv('Files/optimal_features_calibration.csv', sep=',') 
    df_holdout = df_holdout.drop(['id'], axis=1)


    dt = DecisionTreeClassifier(max_depth = 4, class_weight = {1: 20, 0:1})
    print(main(df, df_holdout, "Decision Tree", dt, bs_estimate = True, verbose=True))


def run_bs_adaboost():
    df = pd.read_csv('Files/optimal_features_pca_cos.csv', sep=',') 
    df = df.drop(['id'], axis=1)
 
    df_holdout = pd.read_csv('Files/optimal_features_calibration.csv', sep=',') 
    df_holdout = df_holdout.drop(['id'], axis=1)
   
    dt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, class_weight = {1: 20, 0:1}), n_estimators=21)
    print(main(df, df_holdout, "AdaBoost", dt, bs_estimate = True, verbose=True))

# run_depth_test()
#run_bs_dt()
run_bs_adaboost()


# Test meta learning example
#abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
#main(df=df, name = "AdaBoost Decision Stumps", model=abc)
# Print PR Curves from test
#plt.legend(loc=1)
#plt.title("Precision Recall Curve")
#plt.show()