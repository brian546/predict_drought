# %%
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

# %%
df = pd.read_csv('../data/week_score.csv',parse_dates=['date'])


# %%
# split X and y

X = df[['PRECTOT', 'PS', 'T2M', 'T2M_RANGE', 'WS10M', 'WS50M_RANGE']]
# X = df.drop(columns=['fips','date','score'])


y = df['score']

# %%
# split train and test

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42, stratify= y)

# %%

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%

model = xgboost.XGBClassifier()
params = {
    'n_estimators': [100,200,300]
}
search = GridSearchCV(model,param_grid=params,cv=5, scoring='f1_macro')

search.fit(X_train, y_train)

#  %%

print(search.best_params_)
print(search.best_score_)
# %%

y_pred_test = search.predict(X_test)
y_pred_train = search.predict(X_train)
y_proba_test = search.predict_proba(X_test)
y_proba_train = search.predict_proba(X_train)

# %%

# check for underfitting or overfitting
print('best parameters', search.best_params_)
print('')
print('accuracy score of training set: ', accuracy_score(y_train,y_pred_train)) 
print('accuracy score of testing set: ',accuracy_score(y_test,y_pred_test))
print('')
# check for effect of class imbalance
print(classification_report(y_test,y_pred_test))
print(confusion_matrix(y_test,y_pred_test)) 

# check for 
# print(classification_report(y_train,y_pred_train))
# print(confusion_matrix(y_train,y_pred_train)) 




# %%
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy import interp

y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4,5])

# Compute ROC curve and ROC area for each class

n_classes = 6

fpr = dict()
tpr = dict()
roc_auc = dict()


for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba_test.ravel())
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

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','slateblue', 'purple', 'darkgreen'])
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

#  %%
dump(search,'../model/xgb-full-data-grid-search.joblib')
# %%

search = load('../model/xgb-drop-data-grid-search.joblib')


# %%
print(search.best_params_)
# %%
 