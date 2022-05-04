import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# Function for encoding categorical features with Label Encoder
# Parameter : df -> Missing value handling, feature selection completed dataset
def labelEnc(df):
    label = LabelEncoder()
    # select only columns with values are in object type and make new dataframe
    categorical_df = df.select_dtypes(include='object')
    numerical_df = df.select_dtypes(exclude='object')
    # encode all categorical columns with Label Encoder
    for i in range(0, len(categorical_df.columns)):
        df[categorical_df.columns[i]] = label.fit_transform(categorical_df.iloc[:, [i]].values)

    return df

def OneHotEnc(df):
    df = pd.get_dummies(df)
    return df

# Function for showing heatmap of correlation between each attributes
# Parameter : df -> Missing value handling, feature selection completed dataset
def showHeatmap(df):
    df = labelEnc(df)
    heatmap_data = df
    colormap = plt.cm.PuBu
    plt.figure(figsize=(15, 15))
    plt.title("Correlation of Features", y=1.05, size=15)
    sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
                    annot=True, annot_kws={"size": 8})
    plt.show()


# Function for trying diverse of scaling, algorithms to train and test and shows the result of each methods.
# Parameter : df -> Missing value handling, feature selection completed dataset
#             target -> target column of dataset
#             test_size -> size of test set in percentage
#             k -> number of subsets to split
#             encoders -> list of encoders
#             sclaers  -> list of scalers
def diverseTrainingResult(df, target, test_size, k, encoders, scalers):

    # parameter : scaler -> which scaler will you use(String type)
    #             X_train -> splitted train set to scale
    #             X_test -> splitted test set to scale
    def scaling(scaler, X_train, X_test):

        # Make scaler with the parameter inputted with below condition
        Scaler = scaler
        if scaler == 'MinMaxScaler':
            Scaler = MinMaxScaler()
        elif scaler == 'MaxAbsScaler':
            Scaler = MaxAbsScaler()
        elif scaler == 'StandardScaler':
            Scaler = StandardScaler()
        elif scaler == 'RobustScaler':
            Scaler = RobustScaler()

        # Scale the datasets with each scaler chosen with condition above
        X_train_scale = Scaler.fit_transform(X_train)
        X_test_scale = Scaler.transform(X_test)


        return X_train_scale, X_test_scale

    # Function for visulization of ROC curve
    # Parameter : y_test -> test set target column
    #             pred_proba -> test set predicted probabilty
    #             label -> algorithm name used to draw the roc curve
    def rocvis(y_test, pred_proba, label):
        # FPR, TPR values are returned according to the threshold
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
        # Draw roc curve with plot
        plt.plot(fpr, tpr, label=label)
        # Draw diagonal straight line
        plt.plot([0, 1], [0, 1], linestyle='--')


    # Function of executing scaled data with Random Forest Classifer and predict accuracy
    # Parameters : X_train_scale -> Splitted train attriubutes with scaled
    #              y_train -> Splitted train target feature
    #              X_test_scale -> Splitted test attriubtes with scaled
    #              y_test -> Splitted test target feature
    def randomForest(X_train_scale, y_train, X_test_scale, y_test):
        print('========================= Random Forest ==========================')
        # Set Grid search parameter
        # parameter : 설명바람
        parameters = {'max_depth': [10, 30, 50, 100],
                      'n_estimators': [100, 500, 1000],
                      'criterion': ["gini", "entropy"]}

        # Make the Random Forest Classifier model
        rf = RandomForestClassifier(random_state=0)
        # Make grid search model using Random Forest classifier algorithm and parameters defined above and cross validate defined
        rf_model = GridSearchCV(rf, parameters, cv=kfold)
        # Fit model
        rf_model.fit(X_train_scale, y_train)

        print('\nBest parameter : ', rf_model.best_params_)
        print('Best score : ', round(rf_model.best_score_, 6))

        rf_best = rf_model.best_estimator_
        rf_score = round(rf_model.best_score_, 6)
        rf_parameter = rf_model.best_params_

        # predict y
        rf_y_pred = rf_best.predict(X_test_scale)
        # predict proba y
        rf_y_pred_proba = rf_best.predict_proba(X_test_scale)[:,1]

        # Make confusion matrix
        rf_cf = confusion_matrix(y_test, rf_y_pred)
        rf_total = np.sum(rf_cf, axis=1)
        rf_cf = rf_cf / rf_total[:, None]
        rf_cf = pd.DataFrame(rf_cf, index=["TN", "FN"], columns=["FP", "TP"])

        # visualization
        plt.figure(figsize=(10, 7))
        plt.title("Confusion Matrix with Random Forest")
        sns.heatmap(rf_cf, annot=True, annot_kws={"size": 20})
        plt.show()

        # precision, recall, f1 score, roc_auc
        rf_p = round(precision_score(y_test, rf_y_pred), 6)
        print("precision score :", rf_p)
        rf_r = round(recall_score(y_test, rf_y_pred), 6)
        print("recall score :", rf_r)
        rf_f = round(f1_score(y_test, rf_y_pred), 6)
        print("F1 score :", rf_f)
        rf_roc_auc = roc_auc_score(y_test, rf_y_pred_proba)
        print("AUC :", rf_roc_auc)

        return 'Random Forest Clasifier', rf_best,rf_score, rf_parameter, rf_p, rf_r, rf_f, rf_y_pred_proba, rf_roc_auc

    # Function of executing scaled data with Logistic Regression and predict accuracy
    # Parameters : X_train_scale -> Splitted train attriubutes with scaled
    #              y_train -> Splitted train target feature
    #              X_test_scale -> Splitted test attriubtes with scaled
    #              y_test -> Splitted test target feature
    def logistRegression(X_train_scale, y_train, X_test_scale, y_test):
        print('======================= Logistic Regression =======================')
        # various parameter
        parameters = {'C': [0.01, 0.1, 1.0],
                      'solver': ["liblinear", "lbfgs", "sag"],
                      'max_iter': [10, 50, 100, 300, 500, 1000]}

        logisticRegr = LogisticRegression()
        lr_model = GridSearchCV(logisticRegr, parameters, cv=kfold)
        lr_model.fit(X_train_scale, y_train)

        print('\nBest parameter : ', lr_model.best_params_)
        print('Best score : ', round(lr_model.best_score_, 6))

        lr_best = lr_model.best_estimator_
        lr_score = round(lr_model.best_score_, 6)
        lr_parameter = lr_model.best_params_

        # predict y
        lr_y_pred = lr_best.predict(X_test_scale)
        # predict proba y
        lr_y_pred_proba = lr_best.predict_proba(X_test_scale)[:,1]

        # Make confusion matrix
        lr_cf = confusion_matrix(y_test, lr_y_pred)
        lr_total = np.sum(lr_cf, axis=1)
        lr_cf = lr_cf / lr_total[:, None]
        lr_cf = pd.DataFrame(lr_cf, index=["TN", "FN"], columns=["FP", "TP"])

        # visualization
        plt.figure(figsize=(10, 7))
        plt.title("Confusion Matrix with Logistic Regression")
        sns.heatmap(lr_cf, annot=True, annot_kws={"size": 20})
        plt.show()

        # precision, recall, f1 score
        lr_p = round(precision_score(y_test, lr_y_pred), 6)
        print("precision score :", lr_p)
        lr_r = round(recall_score(y_test, lr_y_pred), 6)
        print("recall score :", lr_r)
        lr_f = round(f1_score(y_test, lr_y_pred), 6)
        print("F1 score :", lr_f)
        lr_roc_auc = roc_auc_score(y_test, lr_y_pred_proba)
        print("AUC :", lr_roc_auc)

        return 'Logistic Regression', lr_best, lr_score, lr_parameter, lr_p, lr_r, lr_f, lr_y_pred_proba, lr_roc_auc

    # Function of executing scaled data with K-Nearnest Neighbor and predict accuracy
    # Parameters : X_train_scale -> Splitted train attriubutes with scaled
    #              y_train -> Splitted train target feature
    #              X_test_scale -> Splitted test attriubtes with scaled
    #              y_test -> Splitted test target feature
    def KNN(X_train_scale, y_train, X_test_scale, y_test):
        print('=============================== KNN ================================')
        parameters = {'weights': ['uniform', 'distance'], 'n_neighbors': range(1, 20)}
        knn = KNeighborsClassifier()
        knn_model = GridSearchCV(knn, parameters, cv=kfold)
        knn_model.fit(X_train_scale, y_train)

        print('\nBest parameter : ', knn_model.best_params_)
        print('Best score : ', round(knn_model.best_score_, 6))

        knn_best = knn_model.best_estimator_
        knn_score = round(knn_model.best_score_, 6)
        knn_parameter = knn_model.best_params_

        # predict y
        knn_y_pred = knn_best.predict(X_test_scale)
        # predict proba y
        knn_y_pred_proba = knn_best.predict_proba(X_test_scale)[:,1]

        # Make confusion matrix
        knn_cf = confusion_matrix(y_test, knn_y_pred)
        knn_total = np.sum(knn_cf, axis=1)
        knn_cf = knn_cf / knn_total[:, None]
        knn_cf = pd.DataFrame(knn_cf, index=["TN", "FN"], columns=["FP", "TP"])

        # visualization
        plt.figure(figsize=(10, 7))
        plt.title("Confusion Matrix with KNN")
        sns.heatmap(knn_cf, annot=True, annot_kws={"size": 20})
        plt.show()

        # precision, recall, f1 score
        knn_p = round(precision_score(y_test, knn_y_pred), 6)
        print("precision score :", knn_p)
        knn_r = round(recall_score(y_test, knn_y_pred), 6)
        print("recall score :", knn_r)
        knn_f = round(f1_score(y_test, knn_y_pred), 6)
        print("F1 score :", knn_f)
        knn_roc_auc = roc_auc_score(y_test, knn_y_pred_proba)
        print("AUC :", knn_roc_auc)

        return 'K Nearest Neighbors', knn_best, knn_score, knn_parameter, knn_p, knn_r, knn_f, knn_y_pred_proba, knn_roc_auc

    # Function of executing scaled data with XGBoosting and predict accuracy
    # Parameters : X_train_scale -> Splitted train attriubutes with scaled
    #              y_train -> Splitted train target feature
    #              X_test_scale -> Splitted test attriubtes with scaled
    #              y_test -> Splitted test target feature
    def XGB(X_train_scale, y_train, X_test_scale, y_test):
        parameters = {'booster': ['gbtree'],
                      "n_estimators": [100, 500, 1000],
                      'max_depth': [5, 10, 30, 50],
                      "learning_rate": [0.05, 0.1, 0.3]}

        xgb = XGBClassifier(min_child_weight = 1, gamma =0.1)
        xgb_model = GridSearchCV(xgb, parameters, cv=kfold)
        xgb_model.fit(X_train_scale, y_train)
        print('========================= XGB Classifier ==========================')
        print('\nBest parameter : ', xgb_model.best_params_)
        print('Best score : ', round(xgb_model.best_score_, 6))
        xgb_best = xgb_model.best_estimator_
        xgb_score = round(xgb_model.best_score_, 6)
        xgb_parameter = xgb_model.best_params_

        # predict y
        xgb_y_pred = xgb_best.predict(X_test_scale)
        # predict proba y
        xgb_y_pred_proba = xgb_best.predict_proba(X_test_scale)[:,1]

        # Make confusion matrix
        xgb_cf = confusion_matrix(y_test, xgb_y_pred)
        xgb_total = np.sum(xgb_cf, axis=1)
        xgb_cf = xgb_cf / xgb_total[:, None]
        xgb_cf = pd.DataFrame(xgb_cf, index=["TN", "FN"], columns=["FP", "TP"])

        # visualization
        plt.figure(figsize=(10, 7))
        plt.title("Confusion Matrix with XGB")
        sns.heatmap(xgb_cf, annot=True, annot_kws={"size": 20})
        plt.show()

        # precision, recall, f1 score
        xgb_p = round(precision_score(y_test, xgb_y_pred), 6)
        print("precision score :", xgb_p)
        xgb_r = round(recall_score(y_test, xgb_y_pred), 6)
        print("recall score :", xgb_r)
        xgb_f = round(f1_score(y_test, xgb_y_pred), 6)
        print("F1 score :", xgb_f)
        xgb_roc_auc = roc_auc_score(y_test, xgb_y_pred_proba)
        print("AUC :", xgb_roc_auc)

        return 'XGB Clasifier', xgb_best, xgb_score, xgb_parameter, xgb_p, xgb_r, xgb_f, xgb_y_pred_proba, xgb_roc_auc

    # Initializing the best score, precision score, recall score, F1 score, AUC in 0 to get the best score at the end of the function.
    best_score = 0
    best_precision = 0
    best_recall = 0
    best_F1 = 0
    best_AUC = 0
    # number of cross validation is k
    kfold = KFold(k, shuffle=True)

    # # Storing list of algorithms to use in this module
    Alg_list = [randomForest, logistRegression, KNN, XGB]

    # # Using double loop and some conditions to find out and store the best score, algorithm, scaler, ..etc
    # # Rotate all scaler used in this module
    for e in encoders:
        if e =='LabelEncoder':
            # Do label encoding with all categorical features
            df = labelEnc(df)

            # Set the feature attributes and target attribute
            X = df.drop([target], 1)
            y = df[target]

            # Split the data into train and target data.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            for i in scalers:
                # When it meets MinMaxScaler
                if i == 'MinMaxScaler':
                    # MinMax Scaling using scaling function defined above
                    X_train, X_test = scaling(i, X_train, X_test)
                    print('========================= Label Encoder & MinMax Scaler =========================')
                    # store values while rotating all algorithms used in this module
                    for alg in Alg_list:
                        algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
                        # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
                        if F1 > best_F1:
                            best_encoder = 'Label Encoder'
                            best_scaler = 'MinMax Scaler'
                            best_algorithm = algorithm
                            best_estimator = estimator
                            best_score = score
                            best_precision = precision
                            best_recall = recall
                            best_F1 = F1
                            best_pred_proba = pred_proba
                            best_AUC = AUC
                            best_y_test = y_test
                # When it meets MaxAbsScaler
                elif i == 'MaxAbsScaler':
                    # MaxAbs Scaling using scaling function defined above
                    X_train, X_test = scaling(i, X_train, X_test)
                    print('========================= Label Encoder & MaxAbs Scaler =========================')
                    # store values while rotating all algorithms used in this module
                    for alg in Alg_list:
                        algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
                        # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
                        if F1 > best_F1:
                            best_encoder = 'Label Encoder'
                            best_scaler = 'MaxAbs Scaler'
                            best_algorithm = algorithm
                            best_estimator = estimator
                            best_score = score
                            best_precision = precision
                            best_recall = recall
                            best_F1 = F1
                            best_pred_proba = pred_proba
                            best_AUC = AUC
                            best_y_test = y_test
                # When it meets StandardScaler
                elif i == 'StandardScaler':
                    # Standard Scaling using scaling function defined above
                    X_train, X_test = scaling(i, X_train, X_test)
                    print('========================= Label Encoder & Standard Scaler =========================')
                    # store values while rotating all algorithms used in this module
                    for alg in Alg_list:
                        algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
                        # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
                        if F1 > best_F1:
                            best_encoder = 'Label Encoder'
                            best_scaler = 'Standard Scaler'
                            best_algorithm = algorithm
                            best_estimator = estimator
                            best_score = score
                            best_precision = precision
                            best_recall = recall
                            best_F1 = F1
                            best_pred_proba = pred_proba
                            best_AUC =AUC
                            best_y_test = y_test
                # When it meets RobustScaler
                elif i == 'RobustScaler':
                    # Robust Scaling using scaling function defined above
                    X_train, X_test = scaling(i, X_train, X_test)
                    print('========================= Label Encoder & Robust Scaler =========================')
                    # store values while rotating all algorithms used in this module
                    for alg in Alg_list:
                        algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
                        # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
                        if F1 > best_F1:
                            best_encoder = 'Label Encoder'
                            best_scaler = 'Robust Scaler'
                            best_algorithm = algorithm
                            best_estimator = estimator
                            best_score = score
                            best_precision = precision
                            best_recall = recall
                            best_F1 = F1
                            best_pred_proba = pred_proba
                            best_AUC =AUC
                            best_y_test = y_test
        elif e =='OneHotEncoder':
            # Set the feature attributes and target attribute
            X = df.drop([target], 1)
            y = df[target]
            X = OneHotEnc(X)

            # Split the data into train and target data.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            for i in scalers:
                # When it meets MinMaxScaler
                if i == 'MinMaxScaler':
                    # MinMax Scaling using scaling function defined above
                    X_train, X_test = scaling(i, X_train, X_test)
                    print('========================= OneHot Encoder & MinMax Scaler =========================')
                    # store values while rotating all algorithms used in this module
                    for alg in Alg_list:
                        algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
                        # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
                        if F1 > best_F1:
                            best_encoder = 'OneHot Encoder'
                            best_scaler = 'MinMax Scaler'
                            best_algorithm = algorithm
                            best_estimator = estimator
                            best_score = score
                            best_precision = precision
                            best_recall = recall
                            best_F1 = F1
                            best_pred_proba = pred_proba
                            best_AUC = AUC
                            best_y_test = y_test
                # When it meets MaxAbsScaler
                elif i == 'MaxAbsScaler':
                    # MaxAbs Scaling using scaling function defined above
                    X_train, X_test = scaling(i, X_train, X_test)
                    print('========================= OneHot Encoder & MaxAbs Scaler =========================')
                    # store values while rotating all algorithms used in this module
                    for alg in Alg_list:
                        algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
                        # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
                        if F1 > best_F1:
                            best_encoder = 'OneHot Encoder'
                            best_scaler = 'MaxAbs Scaler'
                            best_algorithm = algorithm
                            best_estimator = estimator
                            best_score = score
                            best_precision = precision
                            best_recall = recall
                            best_F1 = F1
                            best_pred_proba = pred_proba
                            best_AUC = AUC
                            best_y_test = y_test
                # When it meets StandardScaler
                elif i == 'StandardScaler':
                    # Standard Scaling using scaling function defined above
                    X_train, X_test = scaling(i, X_train, X_test)
                    print('========================= OneHot Encoder & Standard Scaler =========================')
                    # store values while rotating all algorithms used in this module
                    for alg in Alg_list:
                        algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
                        # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
                        if F1 > best_F1:
                            best_encoder = 'OneHot Encoder'
                            best_scaler = 'Standard Scaler'
                            best_algorithm = algorithm
                            best_estimator = estimator
                            best_score = score
                            best_precision = precision
                            best_recall = recall
                            best_F1 = F1
                            best_pred_proba = pred_proba
                            best_AUC =AUC
                            best_y_test = y_test
                # When it meets RobustScaler
                elif i == 'RobustScaler':
                    # Robust Scaling using scaling function defined above
                    X_train, X_test = scaling(i, X_train, X_test)
                    print('========================= OneHot Encoder & Robust Scaler =========================')
                    # store values while rotating all algorithms used in this module
                    for alg in Alg_list:
                        algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
                        # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
                        if F1 > best_F1:
                            best_encoder = 'OneHot Encoder'
                            best_scaler = 'Robust Scaler'
                            best_algorithm = algorithm
                            best_estimator = estimator
                            best_score = score
                            best_precision = precision
                            best_recall = recall
                            best_F1 = F1
                            best_pred_proba = pred_proba
                            best_AUC =AUC
                            best_y_test = y_test


    # Print the result
    print('=============== Result =====================')
    print('Best Encoder : ', best_encoder)
    print('Best scaler : ', best_scaler)
    print('Best algorithm : ', best_algorithm)
    print('Best estimator : ', best_estimator)
    print('Best score : ', best_score)
    print('Best precision score : ', best_precision)
    print('Best recall score : ', best_recall)
    print('Best F1 score : ', best_F1)
    print('Best AUC : ', best_AUC)

    # Call function rocvis to visulize roc curve with parameters defined
    rocvis(best_y_test, best_pred_proba, best_algorithm)
    # Setting FPR axis of X and scaling into unit of 0.1, labels
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend(fontsize=18)
    plt.title("Roc Curve", fontsize=25)
    plt.show()