import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import izip
from math import exp, log
from unbalanced_dataset.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
import cPickle as pickle


def preprocess(dropcols=['last_sign_in_at'] ,today='2016-06-20'):
    # Defining today's day for reference
    date = pd.to_datetime(today)

    # Loading csv extracted from database into a Pandas DataFrame
    df = pd.read_csv('dataset.csv', usecols=range(1,17))

    # Dropping Vango's team ids
    vango_ids = [38175, 1, 1326, 587, 736, 45651, 67966, 48516, 84261, 30975, 4260]
    for _id in vango_ids:
        df = df[df.id != _id]

    # Treating date fields and converting them to datetime timestamp
    df.created_at = pd.to_datetime(df['created_at'])
    df.last_session = pd.to_datetime(df['last_session'])
    df.last_favorited_artwork_date = pd.to_datetime(df['last_favorited_artwork_date'])
    df.last_followed_artist_date = pd.to_datetime(df['last_followed_artist_date'])

    # Creating function to extract only the days (int) from TimeDelta objects
    def extract_days(x):
        try:
            return x.days
        except:
            return 0

    # New feature: Extracting the difference between last session and today
    df['days_from_last_session'] = date - df.last_session
    df['days_from_last_session'] = df.days_from_last_session.apply(lambda x: extract_days(x))

    # New feature: Extracting the difference between last session and when user registered
    df['diff_created_to_last'] = df.last_session - df.created_at
    df['diff_created_to_last'] = df.diff_created_to_last.apply(lambda x: extract_days(x))

    # New feature: Extracting the difference between last favorited artwork and today
    df['diff_last_fav_artwork_to_today'] = date - df.last_favorited_artwork_date
    df['diff_last_fav_artwork_to_today'] = df.diff_last_fav_artwork_to_today.apply(lambda x: extract_days(x))

    # New feature: Extracting the difference between last artists followed and today
    df['diff_last_artist_folw_to_today'] = date - df.last_followed_artist_date
    df['diff_last_artist_folw_to_today'] = df.diff_last_artist_folw_to_today.apply(lambda x: extract_days(x))

    df = df.set_index('id')

    # Dropping columns
    df = df.drop(dropcols, axis=1)

    # Filling NaN values with unknown for categorical variables before dummifying
    df.gender.fillna('unknown', inplace=True)
    df.user_type.fillna('unknown', inplace=True)
    df.os.fillna('unknown', inplace=True)

    # Dropping about 100 recent user ids missing all information
    df.num_sessions.dropna(inplace=True)

    # Dropping 3 records based on gender feature
    df = df[(df.gender != 'male (hidden)') & (df.gender != 'female (hidden)')]

    # Dummifying categorical variables (gender, user_type, and os)
    df = pd.concat([df, pd.get_dummies(df.gender, prefix='gender', drop_first=True)], axis=1)
    df = df.drop('gender', axis=1)

#     df = pd.concat([df, pd.get_dummies(df.user_type, prefix='user_type', drop_first=True)], axis=1)
    df = df.drop('user_type', axis=1)

#     df = pd.concat([df, pd.get_dummies(df.os, prefix='os', drop_first=True)], axis=1)
    df = df.drop('os', axis=1)

    # Finally dropping off random Na values (101 records)
    df.dropna(inplace=True)

    return df


def create_label(df, name='label', original_col_name='diff_created_to_last', smaller_than=200):
    df[name] = np.where(df[original_col_name] < smaller_than, 1, 0)

    # Dropping date columns due to high relationship and used to create the label:
    df.drop('diff_last_fav_artwork_to_today', axis=1, inplace=True)
    df.drop('diff_last_artist_folw_to_today', axis=1, inplace=True)
    df.drop('diff_created_to_last', axis=1, inplace=True)
    df.drop('days_from_last_session', axis=1, inplace=True)


def standard_confusion_matrix(y_true, y_predict):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_predict)
    return np.array([[tp, fp], [fn, tn]])


def roc_curve(probabilities, labels):
    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()


def fit_logistic_regression(X, y):

    sm = SMOTE(kind='regular')
    X_resampled, y_resampled = sm.fit_transform(X, y)

    # Splitting train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)

    # Fitting regression and getting its scores
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    predict_log = log_reg.predict(X_test)
    print "\nLogistic Regression Scores:\n"
    print "Accuracy on test set:", log_reg.score(X_test, y_test)
    print "Precision:", precision_score(y_test, predict_log)
    print "Recall:", recall_score(y_test, predict_log)

    # Fitting multiple k-fold cross validations and getting mean scores
    kfold = KFold(len(y))

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold:
        model = LogisticRegression()
        model.fit(X[train_index], y[train_index])
        y_predict = model.predict(X[test_index])
        y_true = y[test_index]
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

    print "\nK-Fold Cross Validation on Logistic Regression Scores:\n"
    print "accuracy:", np.average(accuracies)
    print "precision:", np.average(precisions)
    print "recall:", np.average(recalls)

    cols = list(df.columns)

    print
    print "Beta scores:"
    for name, coef in izip(df.columns, model.coef_[0]):
        print "%s: %.4f" % (name, coef)

    y_predict = log_reg.predict(X_test)
    y_proba = log_reg.predict_proba(X_test)
    cm = standard_confusion_matrix(y_test, y_predict)

    tpr, fpr, thres = roc_curve(y_proba[:,0:1].flatten(), y_test)
    plt.plot(tpr, fpr)
    plt.show()

    fix, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True,  fmt='', square=True, \
                            xticklabels=['1', '0'], \
                            yticklabels=['1', '0']);
    plt.show()

    print
    print "Likelihoods:"
    for i, coef in enumerate(log_reg.coef_[0]):
#         print "beta %s: %.5f" % (cols[i], exp(coef))
        if coef <0:
            print "*Increasing the %s by 1 point decreases the chance of label=1 by a factor of %.4f.*" % (cols[i], exp(coef))
        else:
            print "*Increasing the %s by 1 point increases the chance of label=1 by a factor of %.4f.*" % (cols[i], exp(coef))
        print

    print "To double:"
    for i, coef in enumerate(model.coef_[0]):
#     print "beta %s: %.5f" % (cols[i], log(2) / coef)
        if coef < 0:
            print "*Decreasing the %s score by %d points doubles the chance of label=1.*" % (cols[i], log(2) / coef)
        else:
            print "*Increasing the %s score by %d points doubles the chance of label=1.*" % (cols[i], log(2) / coef)
        print


def preprocess_purchases_and_join_with(df):
    # Loading data
    purch = pd.read_csv('purchases.csv', usecols=range(1,5))

    # Dropping purchases from anonymous non-registered users
    purch.dropna(inplace=True)

    # Aggregating purchase info per user_id
    user_purchs = purch.groupby('user_id').agg({'total_pieces_purchased':np.sum,\
                                                'total_spent':np.sum})

    # Merging with original DataFrame
    user_merged = df.join(user_purchs)

    # Replacing Na's with 0s
    user_merged.total_spent.fillna(0, inplace=True)
    user_merged.total_pieces_purchased.fillna(0, inplace=True)

    user_merged.drop('total_pieces_purchased', axis=1, inplace=True)

    user_merged['purchased'] = np.where(user_merged.total_spent > 0, 1, 0)

    user_merged.drop('total_spent', axis=1, inplace=True)

    return user_merged


def load_and_add_purchase_data():
    drop_cols = ['last_sign_in_at',
             'created_at',
             'last_session',
             'last_favorited_artwork_date',
             'last_followed_artist_date',
             'total_follows',
             'total_favorites',
             'city']
    df = preprocess(drop_cols)

    # Dropping date columns due to high relationship and used to create the label:
    df.drop('diff_last_fav_artwork_to_today', axis=1, inplace=True)
    df.drop('diff_last_artist_folw_to_today', axis=1, inplace=True)
    df.drop('diff_created_to_last', axis=1, inplace=True)
    df.drop('days_from_last_session', axis=1, inplace=True)

    return df



def plot_importance(clf, X, max_features=10):
    '''Plot feature importance'''
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # Show only top features
    pos = pos[-max_features:]
    feature_importance = (feature_importance[sorted_idx])[-max_features:]
    feature_names = (X.columns[sorted_idx])[-max_features:]

    plt.barh(pos, feature_importance, align='center')
    plt.yticks(pos, feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def grid_search_rf():
    rf_grid = {
    'max_depth': [4, 8, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [1, 2, 4],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True], # Mandatory with oob_score=True
    'n_estimators': [50, 100, 200, 400],
    'random_state': [67],
    'oob_score': [True],
    'n_jobs': [-1]
    }

    rf_grid_cv = GridSearchCV(RandomForestClassifier(),
                                 rf_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='roc_auc')

    sm = SMOTE(kind='regular', ratio=0.4)
    X_resampled, y_resampled = sm.fit_transform(X, y)

    # Splitting train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)

    rf_grid_cv.fit(X_train, y_train)

    print "Best Parameters found:\n", rf_grid_cv.best_params_

    best_model = rf_grid_cv.best_estimator_

    print "OOB:", best_model.oob_score_


def fit_random_forest(X, y):
    sm = SMOTE(kind='regular', ratio=0.5)
    X_resampled, y_resampled = sm.fit_transform(X, y)

    # Splitting train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)

    rf = RandomForestClassifier(oob_score=True, n_jobs=-1, bootstrap=True, min_samples_leaf=2,
                                n_estimators=400, min_samples_split=1, random_state=67,
                                max_features=None, max_depth=None)
    rf.fit(X_train, y_train)

    # Draw a confusion matrix for the results
    y_predict = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    cm = standard_confusion_matrix(y_test, y_predict)

    print "\nRandom Forest Scores:\n"
    print "accuracy:", rf.score(X_test, y_test)
    print "precision:", precision_score(y_test, y_predict)
    print "recall:", recall_score(y_test, y_predict)

    tpr, fpr, thres = roc_curve(y_proba[:,0:1].flatten(), y_test)
    plt.plot(tpr, fpr)
    plt.show()

    fix, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True,  fmt='', square=True, \
                            xticklabels=['1', '0'], \
                            yticklabels=['1', '0']);
    plt.show()

    cols = list(df.columns)

    print "\nFeature Importance: \n"
    for name, importance in izip(cols, rf.feature_importances_):
        print round(importance,4), '\t\t', name

    plot_importance(rf, merged_df, max_features=16)

    return rf


if __name__ == '__main__':
    # # Logistic Regression with Label = purchased
    # df = load_and_add_purchase_data()
    # merged_df = preprocess_purchases_and_join_with(df)
    #
    # # Defining y label and X matrix
    # y = merged_df.pop('purchased').values
    # X = merged_df.values
    #
    # fit_logistic_regression(X, y)


    # Random Forests with Label = purchased (with best params after GridSearch)
    df = load_and_add_purchase_data()
    merged_df = preprocess_purchases_and_join_with(df)

    # Scaling features
    merged_df.num_sessions = scale(merged_df.num_sessions)
    merged_df.total_artists_followed = scale(merged_df.total_artists_followed)
    merged_df.total_artworks_favorited = scale(merged_df.total_artworks_favorited)
    merged_df.total_artworks_shared = scale(merged_df.total_artworks_shared)

    # Defining y label and X matrix
    y = merged_df.pop('purchased').values
    X = merged_df.values

    model = fit_random_forest(X, y)

    # Pickling the best model for prediction
    print "Pickling the best model for prediction, please wait..."
    with open("model.pkl", 'w') as f:
        pickle.dump(model, f)
    print "Pickles are done!"
    print"""
              ___________
             [___________]
             /           \.
            /~~^~^~^~^~^~^\.
           |===============|
           | P I C K L E S |
           | ,-.   __      |
           | \ ,'-'. )     |
           |  '._'_;'      |
           ;===============;
            \             /
             ````````````
    """
    print '\n'
