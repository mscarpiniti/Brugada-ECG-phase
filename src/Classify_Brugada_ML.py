# -*- coding: utf-8 -*-
"""
Main script for running the 7-fold CV of the traditional ML-based classifiers
used in [1] for comparing the identification performance on the Brugada syndrome.

Data are available at: https://datadryad.org/stash/dataset/doi:10.5061/dryad.s1rn8pkd9


[1] M. Scarpiniti and A. Uncini, "Exploiting phase information for the
identification of Brugada syndrome: A preliminary study", in Italian Workshop
on Neural Networks (WIRN 2024), Vietri sul Mare (SA), Italy, June 05-07, 2024.


Created on Fri Jun 28 14:25:32 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""




import numpy as np
import utils as ut

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# Set main hyper-parameters
N_L = 9    # Number of ECG leads
K_f = 7    # Number of CV folds


# Set all folders
data_folder = './Data/'
save_folder = './Saved_Models/'
result_folder = './Results/'


# Load the dataset
X, y = ut.load_data_DNN(data_folder, N_L)


# Set selected classifiers
clf1 = DecisionTreeClassifier(random_state=42)
clf2 = KNeighborsClassifier(n_neighbors=11)
clf3 = clf = LogisticRegression(C=1, max_iter=1000, random_state=42, n_jobs=-1)
clf4 = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
clf5 = SVC(C=10, random_state=42, probability=True)

classifiers = [clf1, clf2, clf3, clf4, clf5]
clf_names = ['DT', 'kNN', 'LR', 'RF', 'SVM']




# %% K-fold Cross Validation model evaluation

for clf, c_name in zip(classifiers, clf_names):

    ACC = []
    PRE = []
    REC = []
    AUC = []
    F1  = []
    CM  = []


    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=K_f, shuffle=True, random_state=42)

    fold_no = 1
    model_name = c_name + str(N_L)
    res_file = result_folder + 'Results_CV_' + model_name + '.txt'


    for train, test in kfold.split(X, y):

        # Train the selected classifier
        clf.fit(X[train], y[train])

        # Evaluate the model
        y_prob = clf.predict(X[test])
        y_pred = np.round(y_prob)


        # Compute metrics
        acc = accuracy_score(y[test], y_pred)
        pre = precision_score(y[test], y_pred, average='binary')
        rec = recall_score(y[test], y_pred, average='binary')
        f1  = f1_score(y[test], y_pred, average='binary')
        auc = roc_auc_score(y[test], y_prob)
        cm  = confusion_matrix(y[test], y_pred)

        # Append results
        ACC.append(acc)
        PRE.append(pre)
        REC.append(rec)
        AUC.append(auc)
        F1.append(f1)
        CM.append(cm)


        # Write results of each fold to a text file
        with open(res_file, 'a') as results:  # save the results in a .txt file
              results.write('-------------------------------------------------------\n')
              results.write('Model: %s\n' % model_name)
              results.write('Fold: %s\n' % fold_no)
              results.write('Acc: %s\n' % round(100*acc,2))
              results.write('Pre: %s\n' % round(pre,3))
              results.write('Rec: %s\n' % round(rec,3))
              results.write('F1: %s\n' % round(f1,3))
              results.write('AUC: %s\n\n' % round(auc,3))
              results.write(classification_report(y[test], y_pred, digits=3))
              results.write('\n\n')


        # Increase fold number
        print("End of fold {}/{}".format(fold_no,K_f))
        fold_no += 1



    # Print mean and std values for each metrics
    print('\nModel:', model_name)
    print("Accuracy. Mean: {}, Std: {}".format(round(np.mean(ACC),4), round(np.std(ACC),4)))
    print("Precision. Mean: {}, Std: {}".format(round(np.mean(PRE),4), round(np.std(PRE),4)))
    print("Recall. Mean: {}, Std: {}".format(round(np.mean(REC),4), round(np.std(REC),4)))
    print("F1-score. Mean: {}, Std: {}".format(round(np.mean(F1),4), round(np.std(F1),4)))
    print("AUC. Mean: {}, Std: {}\n\n".format(round(np.mean(AUC),4), round(np.std(AUC),4)))


    # Write averaged results to the text file
    with open(res_file, 'a') as results:  # save the results in a .txt file
          results.write('-------------------------------------------------------\n')
          results.write('Model: %s\n' % (model_name))
          results.write('Averaged results \n')
          results.write('Accuracy. Mean: %s, Std: %s\n' % (round(np.mean(ACC),4), round(np.std(ACC),4)))
          results.write('Precision. Mean: %s, Std: %s\n' % (round(np.mean(PRE),4), round(np.std(PRE),4)))
          results.write('Recall. Mean: %s, Std: %s\n' % (round(np.mean(REC),4), round(np.std(REC),4)))
          results.write('F1-score. Mean: %s, Std: %s\n' % (round(np.mean(F1),4), round(np.std(F1),4)))
          results.write('AUC. Mean: %s, Std: %s\n\n' % (round(np.mean(AUC),4), round(np.std(AUC),4)))
          results.write('\n\n')
