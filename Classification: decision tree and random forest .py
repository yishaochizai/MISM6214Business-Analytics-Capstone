#!/usr/bin/env python
# coding: utf-8

# ## 14/11 Decision Trees

# In[1]:


#install libraries
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pyarrow')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install fastparquet')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt
import pyarrow as pa
import sklearn as sk
from sklearn.model_selection import KFold, cross_val_score
from sklearn import ensemble
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[3]:


pip install --upgrade pandas


# In[4]:


df = pd.read_csv('/Users/yishaochizai/Desktop/capstone/cor_vars') 


# In[5]:


#inspect the dataset
df.head()


# In[6]:


#Delete variable that cannot be in the model
dfx = df.drop(columns = ['Unnamed: 0'])


# In[7]:


#double check
dfx.head()


# In[8]:


#descriptive statistics
dfx.describe()


# In[9]:


#create the new columns: day30 - day14 named as dif
dfx['dif'] = dfx['d30_spend'] - dfx['d14_spend']


# In[10]:


#creat a new colomns : make dif a 0,1 value
dfx['dif_dummy'] = np.where(dfx['dif'] != 0, 1, dfx['dif'])


# In[11]:


#creat the dataset for model
dfy_1 = dfx.drop(columns = ['dif'])
dfy_2 = dfy_1.drop(columns = ['dif_dummy'])
dfy_3 = dfy_2.drop(columns = ['d30_spend'])


# ## CLASSIFICATION : Building Decision Tree Model 

# ### Splitting Data

# In[58]:


# define X and y
X = dfy_3
y = dfx.dif_dummy


# In[59]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[60]:


#shape of train and test data
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)


# ### Building Decision Tree Model

# In[61]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth = 3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# ### Evaluate the model

# In[16]:


# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[17]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[18]:


# Print the confusion matrix using Matplotlib

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[19]:



from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[20]:


#precision
print('Precision: %.3f' % precision_score(y_test, y_pred))


# In[21]:


#Recall
print('Recall: %.3f' % recall_score(y_test, y_pred))


# In[22]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[23]:


# F1-score
F1 = 2 * (0.872 * 0.538) / (0.872 + 0.538)
print(F1)


# ### Visualizing Decision Trees

# In[24]:


get_ipython().system('pip install graphviz')
get_ipython().system('pip install pydotplus')


# In[25]:


import six
import sys
sys.modules['sklearn.externals.six'] = six


# In[26]:


from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus


# In[27]:


from sklearn.externals.six import StringIO
get_ipython().system('pip install dtreeviz')
from dtreeviz.trees import dtreeviz


# #### Print Text Representation

# In[28]:


text_representation = tree.export_text(clf)
print(text_representation)


# In[29]:


#save above result
with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)


# In[30]:


tree.plot_tree(clf, class_names=True)


# In[31]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


# In[32]:


import matplotlib.pyplot as plt 


# In[33]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


# In[34]:


# predict probabilities
pred_prob1 = clf.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)



# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[35]:


# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])

print(auc_score1)


# In[36]:


# plot roc curves
plt.plot(fpr1, tpr1, marker='.',color='orange', label='Decision Tree Classification')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')



plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# ## Random Forest Classification

# In[39]:


X = dfy_3
y = dfx.dif_dummy


# In[40]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
x2_train, x2_test, y2_train, y2_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[41]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rfc = RandomForestClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)
rfc.fit(x2_train, y2_train)

rfc_predict = rfc.predict(x2_test)


# In[42]:



from sklearn.metrics import accuracy_score
accuracy_score(y_test, rfc_predict)


# In[43]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
confusion_matrix(y_test, rfc_predict)


# In[44]:


metrics.precision_score(y_test, rfc_predict)


# In[45]:


metrics.recall_score(y_test, rfc_predict)


# In[47]:


prediction_2 = x2_test
result_2 = pd.DataFrame(rfc_predict)
prediction_update_2 = np.append(prediction_2, result_2, 1)

df2_2 = pd.DataFrame(prediction_update_2)
df2_2.to_csv('1212_rf.csv')


# In[48]:


rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring = 'roc_auc')


# In[50]:


#The roc_auc scoring used in the cross-validation model shows the area under the ROC curve.
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())


# In[51]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y2_test, rfc_predict))


# In[52]:


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, 
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


# In[53]:


#variable importance
importance = rfc.feature_importances_
indices = np.argsort(importance)[::-1]
feat_labels = dfx.columns[0:]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))


# In[54]:


import seaborn as sns


# In[55]:


def plot_feature_importance(importance,names,model_type):

#Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

#Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

#Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df= fi_df.head(10)

#Define size of bar plot
    plt.figure(figsize=(10, 10))
#Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[57]:


plot_feature_importance(importance, dfy_3.columns, 'RANDOM FOREST')


# In[ ]:




