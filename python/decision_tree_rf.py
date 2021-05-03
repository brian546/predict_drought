# %%
from joblib import dump, load
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, MinMaxScaler
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interp
from itertools import cycle


# %%
# --- import CSV
df = pd.read_csv('../data/test_timeseries.csv', parse_dates=['date'])
df
# %%
# --- convert data to mean by week, to match weekly score
new_df = df
new_df = new_df[(df['date'] > pd.Timestamp(2019, 1, 1)) &
                (df['date'] < pd.Timestamp(2020, 12, 30))]
week_df = new_df.groupby('fips').resample(
    'W-TUE', on='date', label='right', closed='right').mean()
week_df = week_df.drop(columns=['fips']).reset_index()

# --- rounding score to int
week_df['score'] = week_df['score'].round().astype(int)
week_df.head()

# # %%
# # --- VIF: finding multi-collinearity issue, in case need statistics to support dropping column


# def calc_vif(X):

#     # Calculating VIF
#     vif = pd.DataFrame()
#     vif["variables"] = X.columns
#     vif["VIF"] = [variance_inflation_factor(
#         X.values, i) for i in range(X.shape[1])]

#     return(vif)


# calc_vif(week_df.iloc[:, 2:])

# %%
# --- Dropping data: make comparison between 2 sets of data
coll_df = week_df[['fips', 'date', 'PRECTOT', 'PS',
                   'T2M', 'T2M_RANGE', 'WS10M', 'WS50M_RANGE', 'score']]
coll_df


# %%
# --- Define methods
dtc = DecisionTreeClassifier()
rf = RandomForestClassifier()

# %%
# --- ROC function, decision tree
# --- a = X_test, b = y_test


def roc_dtc(a, b):

    y_proba_test = grid_dtc_acc.predict_proba(a)
    y_test_bin = label_binarize(b, classes=[0, 1, 2, 3, 4, 5])

    # Compute ROC curve and ROC area for each class

    n_classes = 6

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_bin.ravel(), y_proba_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',
                   'slateblue', 'purple', 'darkgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return y_proba_test

# %%
# --- ROC function, random forest
# --- a = X_test, b = y_test


def roc_rf(a, b):

    y_proba_test = grid_rf_acc.predict_proba(a)
    y_test_bin = label_binarize(b, classes=[0, 1, 2, 3, 4, 5])

    # Compute ROC curve and ROC area for each class

    n_classes = 6

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_bin.ravel(), y_proba_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',
                   'slateblue', 'purple', 'darkgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return y_proba_test


# %%
# ----------- Decision Tree: dropped data
y = coll_df.iloc[:, -1]
X = coll_df.iloc[:, 2:-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---- Scaling data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# ----------- Decision Tree draft, GridSearch
grid_values = {'criterion': ['gini', 'entropy'],
               'max_depth': [10, 15, 20]}
grid_dtc_acc = GridSearchCV(
    dtc, param_grid=grid_values, scoring='f1_macro', cv=5)
grid_dtc_acc.fit(X_train, y_train)
y_pred_acc = grid_dtc_acc.predict(X_test)
y_pred_acc_t = grid_dtc_acc.predict(X_train)
print(f'Best parameter: {grid_dtc_acc.best_params_}')


# # %%
# # --- Save model
# dump(grid_dtc_acc, 'dtc-grid-search.joblib')

# # %%
# # --- Load model
# new_search = load('dtc-grid-search.joblib')
# new_search

# %%
# --- Printing reports
print(f'Best parameter: {grid_dtc_acc.best_params_}\n')
print(
    f'accuracy score of training set: {accuracy_score(y_train, y_pred_acc_t)}')
print(
    f'accuracy score of testing set: {accuracy_score(y_test, y_pred_acc)}\n')

print(classification_report(y_test, y_pred_acc))
print(confusion_matrix(y_test, y_pred_acc))

# %%
roc_dtc(X_test, y_test)
# %%
# --- Print tree
print(tree.export_text(dtc))
#fig = plt.figure(figsize=(30, 30))
#tree.plot_tree(dtc, filled=True)

# %%
# --- Random forest, dropped data

rf = RandomForestClassifier(max_leaf_nodes=16, n_jobs=-1, random_state=42)
grid_values = {'n_estimators': [100, 200, 300, 400, 500]}
grid_rf_acc = GridSearchCV(rf, param_grid=grid_values, cv=5)
grid_rf_acc.fit(X_train, y_train)
y_pred_acc = grid_rf_acc.predict(X_test)
y_pred_acc_t = grid_rf_acc.predict(X_train)
print(f'Best parameter: {grid_rf_acc.best_params_}')

# %%
# --- Save model
# dump(grid_rf_acc, 'rf-grid-search.joblib')

# %%
print(f'Best parameter: {grid_rf_acc.best_params_}')
print(
    f'accuracy score of training set: {accuracy_score(y_train, y_pred_acc_t)}')
print(f'accuracy score of testing set: {accuracy_score(y_test, y_pred_acc)}')
print(classification_report(y_test, y_pred_acc))
print(confusion_matrix(y_test, y_pred_acc))

# %%
roc_rf(X_test, y_test)

# %%
# --- Decision Tree for non-dropped data
y = week_df.iloc[:, -1]
X = week_df.iloc[:, 2:-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---- Scaling data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%

grid_values = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 15, 20]}
grid_dtc_acc = GridSearchCV(dtc, param_grid=grid_values, cv=5)
grid_dtc_acc.fit(X_train, y_train)
y_pred_acc = grid_dtc_acc.predict(X_test)
y_pred_acc_t = grid_dtc_acc.predict(X_train)

# %%
print(f'Best parameter: {grid_dtc_acc.best_params_}\n')
print(
    f'accuracy score of training set: {accuracy_score(y_train, y_pred_acc_t)}')
print(f'accuracy score of testing set: {accuracy_score(y_test, y_pred_acc)}\n')
print(classification_report(y_test, y_pred_acc))
print(confusion_matrix(y_test, y_pred_acc))

# %%
roc_dtc(X_test, y_test)

# %%
