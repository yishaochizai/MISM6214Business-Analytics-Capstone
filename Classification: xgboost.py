#import libraries
import pandas as pd 
from xgboost import XGBClassifier, plot_tree
import xgboost as xgb
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import  r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from yellowbrick.regressor import prediction_error
from sklearn.metrics import mean_squared_error as MSE

df = pd.read_csv('cor_vars.csv') 

#Delete variable that cannot be in the model
df = df.drop(columns = ['Unnamed: 0'])

df.insert(0, "d30_minus", df.d30_spend-df.d14_spend, True)

#creat a new colomns : make dif a 0,1 value
df['d30_minus_dummy'] = np.where(df['d30_minus'] != 0, 1, df['d30_minus'])


# df = df.drop(df[df['d30_minus'] == 0].index)
dfy_1 = df.drop(columns = ['d30_minus'])
dfy_2 = dfy_1.drop(columns = ['d30_minus_dummy'])
dfy_3 = dfy_2.drop(columns = ['d30_spend'])
X = dfy_3
y = df.d30_minus_dummy


# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 


# fit model no training data
model = XGBClassifier(max_depth = 2, n_estimators=25, seed=42)
model.fit(X_train, y_train)


# make predictions for test data
y_pred = model.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix', fontsize = 20) # title with fontsize 20
plt.xlabel('Predictions', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('Actuals', fontsize = 15) 
plt.figure(figsize = (20,20))
plt.savefig('confusion.png', transparent=True)
plt.show()
print(classification_report(y_test, y_pred)) 

xgb.plot_importance(model, max_num_features=10)
plt.grid(False)
plt.savefig('ROC',dpi=300, transparent=True)


# predict probabilities
pred_prob1 = model.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])

print(auc_score1)

# plot roc curves
plt.plot(fpr1, tpr1, marker='.',color='orange', label='Xgboost Classification')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.grid(False)
plt.legend(loc='best')
plt.savefig('ROC',dpi=300, transparent=True)
plt.show();



# d14spend != d30_spend dataset
prediction = X_test
result = pd.DataFrame(y_pred)
prediction = np.append(prediction, result, 1)
pd.DataFrame(prediction).to_csv("Prediction.csv")
prediction = pd.read_csv('Prediction.csv') 
prediction_result = prediction[prediction.y_pred != 0]
prediction_result = prediction_result.drop(columns = ['y_pred'])
