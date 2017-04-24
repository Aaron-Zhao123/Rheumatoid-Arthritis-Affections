import sklearn
from sklearn import metrics
from svmPlot import svmPlot
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

def random_forest_learning(X, y, x_train, y_train, x_test, y_test):
    RANDOM_STATE = 0
    ESTIMATOR_TEST = 0
    clf = RandomForestClassifier(   warm_start=True, oob_score=True,
                                    max_features = 'sqrt',
                                    random_state = RANDOM_STATE)
    # Range of `n_estimators` values to explore.
    if (ESTIMATOR_TEST == 1):
        min_estimators = 3
        max_estimators = 120
        error_rate = []
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators = i)
            clf.fit(x_train, y_train)
            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate.append(oob_error)
        print(error_rate)
        sys.exit()
    clf.set_params(n_estimators = 20)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)
    y_score = clf.predict_proba(x_train)
    print("Train Accuracy:", round(metrics.accuracy_score(y_train, y_pred_train),2))
    print("Test Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))


def nn_learning(X, y, x_train, y_train, x_test, y_test):
    batch_size = 1
    num_classes = 3
    epochs = 100
    input_shape = (np.shape(X)[1],)
    print(input_shape)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = Sequential()
    model.add(Dense(300, activation = 'relu',input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=batch_size,
                epochs=epochs,
                verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print(hist.history)
    # print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def svm_learning(X, y, x_train, y_train, x_test, y_test):
    PLOT_FIG = 0

    '''Linear Kernal'''
    # C = 1e-8  # SVM regularization parameter
    C = 1e-18  # SVM regularization parameter
    # C = 0.1  # SVM regularization parameter
    degree = 3 #
    gamma = 1e-2
    gamma = 0.01
    LINEAR = 0
    POLY = 1
    RBF = 0
    BEST_COMB = 0
    SVM_linear = svm.SVC(kernel='linear', C=C, probability=True).fit(x_train, y_train)
    SVM_poly = svm.SVC(kernel='poly', degree = degree, C=C, gamma = gamma, probability=True).fit(x_train, y_train)
    SVM_rbf = svm.SVC(kernel='rbf', C=C, gamma = gamma,probability=True).fit(x_train, y_train)
    if (LINEAR == 1):
        y_pred = SVM_linear.predict(x_test)
        y_pred_train = SVM_linear.predict(x_train)
        '''Compute some thresholds'''
        y_score = SVM_linear.predict_proba(x_test)
        y_score_train = SVM_linear.predict_proba(x_train)
    elif(POLY == 1):
        y_pred = SVM_poly.predict(x_test)
        y_pred_train = SVM_poly.predict(x_train)
        y_score = SVM_poly.predict_proba(x_test)
        y_score_train = SVM_poly.predict_proba(x_train)
    elif(BEST_COMB == 1):
        d_range = np.arange(1,4,step = 1)
        C_range = 10. ** np.arange(-20,0,step = 2)
        g_range = 10. ** np.arange(-10,10,step = 2)
        kernel = ['rbf','linear','poly']
        parameters = [{'gamma': g_range, 'degree': d_range, 'C': C_range, 'kernel': kernel}]
        grid = GridSearchCV(svm.SVC(), parameters, n_jobs=4)
        grid.fit(x_train , y_train)
        bestg = grid.best_params_['gamma']
        bestd = grid.best_params_['degree']
        bestC = grid.best_params_['C']
        bestk = grid.best_params_['kernel']
        print ("The best parameters are: gamma=", bestg, " and Cost=", bestC,
                "degree = ", bestd, "kernel = ", bestk)
        sys.exit()
    elif(RBF == 1):
        y_pred = SVM_rbf.predict(x_test)
        y_pred_train = SVM_rbf.predict(x_train)
        '''Compute some thresholds'''
        y_score = SVM_rbf.predict_proba(x_test)
        y_score_train = SVM_rbf.predict_proba(x_train)

    # print(metrics.classification_report(y_test, y_pred))
    print("Train Accuracy:", round(metrics.accuracy_score(y_train, y_pred_train),2))
    print("Test Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))

    # g_range = [0.001,0.01,0.05,0.1, 0.5, 1, 5, 10, 50]
    # print(g_range)
    # C_range = [0.001,0.01,0.05,0.1, 0.5, 1, 5, 10, 50]
    # g_range = 2. ** np.arange(-20, -10, step=1)
    # print(g_range)
    # C_range = 2. ** np.arange(-20, -10, step=1)
    # parameters = [{'gamma': g_range, 'C': C_range, 'kernel': ['rbf']}]
    # grid = GridSearchCV(svm.SVC(), parameters, n_jobs=4)
    # grid.fit(x_train , y_train)
    # bestG = grid.best_params_['gamma']
    # bestC = grid.best_params_['C']
    # bestk = grid.best_params_['kernel']
    # print ("The best parameters are: gamma=", bestG, " and Cost=", bestC, "kernel is ", bestk)


    if (PLOT_FIG):
        fpr = dict()
        fpr_train = dict()
        tpr = dict()
        tpr_train = dict()
        roc_auc = dict()
        roc_auc_train = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            fpr_train[i], tpr_train[i], _ = roc_curve(y_train[:, i], y_score_train[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])
        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic testing')
        plt.legend(loc="lower right")
        plt.figure()
        lw = 2
        plt.plot(fpr_train[0], tpr_train[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic testing')
        plt.legend(loc="lower right")
        plt.show()
