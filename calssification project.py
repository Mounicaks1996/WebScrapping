#!/usr/bin/env python
# coding: utf-8

# In[130]:


# Import all the required Libraries

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
#inline statement - displays the graphs in the current notebook
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn import preprocessing


# In[131]:


# important funtions
def datasetShape(df):
    rows, cols = df.shape
    print("The dataframe has",rows,"rows and",cols,"columns.")
    
# select numerical and categorical features
def divideFeatures(df):
    numerical_features = df.select_dtypes(include=[np.number])
    categorical_features = df.select_dtypes(include=[np.object])
    return numerical_features, categorical_features


# In[132]:


# loction
import os
os.getcwd()


# In[133]:


# Read the dataset and display the head. You will get the output as mentioned below
data = pd.read_csv('train.csv')
data


# In[134]:


# set target feature
targetFeature='Response'


# In[135]:


# check dataset shape
data.shape


# In[136]:


# On the dataframe apply info() function and observe the Dtypes and Missing Values
data.info()


# In[137]:


data.dtypes


# In[138]:


data.isnull().values.any()


# In[139]:


data.isnull().sum()


# In[140]:


#to interpolate the missing values  
#dataset.interpolate(method ='linear', limit_direction ='forward')


data.fillna(axis=0, method='ffill', inplace=True)


# In[141]:


data.isnull().sum()


# In[142]:


data.describe()


# In[143]:


data.corr()


# In[ ]:





# In[ ]:





# # Univariate Analysis

# In[144]:


fig = plt.figure(figsize =(10, 10))
data.City_Code.value_counts(normalize=True).plot.barh()
plt.show()


# In[145]:


sns.countplot(data=data,x='Accomodation_Type')


# In[ ]:





# # Bivariate

# In[146]:


ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="Accomodation_Type", y="Response", data=data)
plt.show()


# # City_Code vs Response

# In[147]:


plt.figure(figsize = (20, 10))
sns.barplot(x = 'City_Code', y = 'Response', data = data)

plt.title("City_Code vs Response")


# # Upper_Age vsResponse

# In[148]:


sns.lineplot(data = data, x = 'Upper_Age', y = 'Response')


# # Lower_Age vs Response

# In[149]:


data.plot.hexbin(x='Lower_Age', y='Response', gridsize=15)


# # Response vs Reco_Insurance_Type

# In[150]:


data.Response.value_counts(normalize=True)

data.Reco_Insurance_Type.value_counts(normalize=True).plot.barh()
plt.show()


# # Response vs Is_Spouse

# In[151]:


sns.violinplot( x='Response', y='Is_Spouse', data=data,size=8)
plt.show


# # Holding_Policy_Duration vs Is_Spouse

# In[152]:


data.Is_Spouse.value_counts(normalize=True)

data.Holding_Policy_Duration.value_counts(normalize=True).plot.barh()
plt.show()


# # Response vs Health Indicator

# In[153]:


plt.figure(figsize = (5, 5))

sns.barplot(x = 'Response', y = 'Health Indicator', data = data)

plt.title("Response vs Health Indicator")


# # Holding_Policy_Type vs Response

# In[154]:


plt.figure(figsize = (5, 5))
sns.barplot(x = 'Holding_Policy_Type', y = 'Response', data = data)

plt.title("Holding_Policy_Type vs Response")


# # Response vs Reco_Policy_Cat

# In[155]:


sns.violinplot( x='Response', y='Reco_Policy_Cat', data=data,size=8)
plt.show


# # Response vs Reco_Policy_Premium

# In[156]:


plt.figure(figsize = (5, 5))

sns.barplot(x = 'Response', y = 'Reco_Policy_Premium', data = data)

plt.title("Response vs Reco_Policy_Premium")


# # Correlation Matrix

# In[157]:


# correlation heatmap for all features
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, annot=True)
plt.show()


# # Data Preparation

# In[158]:


cont_features, cat_features = divideFeatures(data)
cat_features.head()


# In[159]:


skewed_features = cont_features.apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_features


# # Handle Missing

# In[160]:


# plot missing values

def calc_missing(data):
    missing = data.isna().sum().sort_values(ascending=False)
    missing = missing[missing != 0]
    missing_perc = missing/df.shape[0]*100
    return missing, missing_perc

if data.isna().any().sum()>0:
    missing, missing_perc = calc_missing(data)
    missing.plot(kind='bar',figsize=(14,5))
    plt.title('Missing Values')
    plt.show()
else:
    print("No Missing Values")


# In[161]:


# remove all columns having no values
data.dropna(axis=1, how="all", inplace=True)
data.dropna(axis=0, how="all", inplace=True)
datasetShape(data)


# # Label encoding

# In[162]:


# label encoding on categorical features
def mapFeature(data, f, data_test=None):
    feat = data[f].unique()
    feat_idx = [x for x in range(len(feat))]

    data[f].replace(feat, feat_idx, inplace=True)
    if data_test is not None:
        data_test[f].replace(feat, feat_idx, inplace=True)


# In[163]:


for col in cat_features.columns:
    mapFeature(data, col, data)
data.head()


# # Checking duplicate rows

# In[164]:


duplicate=data[data.duplicated()]
print(duplicate)


# # Seprating dependent and independent variables

# In[165]:


x=data.drop(['Response'],axis=1) #contain all  independent variable
y=data['Response']           #dependent variable


# # Dropping features not much important for prediction.

# In[166]:


# Drop features, because it is not much important for prediction.

data = data.drop('ID', axis = 1)

data = data.drop('City_Code', axis = 1)
data = data.drop('Region_Code', axis = 1)

data = data.drop('Health Indicator', axis = 1)
data = data.drop('Holding_Policy_Duration', axis = 1)


# # Assigning X and y

# In[167]:


# Assigning X and y

X = data.iloc[:,:8]
y = data.iloc[:,8]


# In[168]:


print(X)


# # Data Modelling

# # Split Train-Test Data

# In[169]:


# helper functions

def log1p(vec):
    return np.log1p(abs(vec))

def expm1(x):
    return np.expm1(x)

def clipExp(vec):
    return np.clip(expm1(vec), 0, None)

def printScore(y_train, y_train_pred):
    print(skm.roc_auc_score(y_train, y_train_pred))


# In[170]:


import sklearn.model_selection as skms
import random
seed = 12
np.random.seed(seed)
# shuffle samples
data_shuffle = data.sample(frac=1, random_state=seed).reset_index(drop=True)

data_y = data_shuffle.pop(targetFeature)
data_X = data_shuffle

# split into train dev and test
X_train, X_test, y_train, y_test = skms.train_test_split(data_X, data_y, train_size=0.8, random_state=seed)
print(f"Train set has {X_train.shape[0]} records out of {len(data_shuffle)} which is {round(X_train.shape[0]/len(data_shuffle)*100)}%")
print(f"Test set has {X_test.shape[0]} records out of {len(data_shuffle)} which is {round(X_test.shape[0]/len(data_shuffle)*100)}%")


# # Feature Scaling

# In[171]:


# scaler = skp.RobustScaler()
# scaler = skp.MinMaxScaler()
import sklearn.preprocessing as skp
scaler = skp.StandardScaler()

# apply scaling to all numerical variables except dummy variables as they are already between 0 and 1
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# scale test data with transform()
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

# view sample data
X_train.describe()


# In[ ]:





# In[ ]:





# # Model Building

# In[172]:


import sklearn.utils as sku
class_weights = sku.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights = dict(enumerate(class_weights))
class_weights


# In[173]:


sample_weights = sku.class_weight.compute_sample_weight('balanced', y_train)
sample_weights


# # Logistic Regression

# In[174]:


# Model 1: Logistic Regression

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model=LogisticRegression(random_state=0)
model=model.fit(X_train,y_train)
pred=model.predict(X_test)
lr_probability =model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

acc_lr=accuracy_score(y_test, y_pred)
recall_lr=recall_score(y_test, y_pred, zero_division = 1)
precision_lr=precision_score(y_test, y_pred, zero_division = 1)
f1score_lr=f1_score(y_test, y_pred, zero_division = 1)
AUC_LR=roc_auc_score(y_test, y_pred)
#print accuracy and Auc values of model
print("Accuracy : ", accuracy_score(y_test,y_pred))
print("ROC_AUC Score:",AUC_LR)


# In[175]:


print(classification_report(y_pred,y_test))


# # ROC CURVE

# In[176]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, lr_probability)

plt.title('Logistic Regression ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()


# # Confusion Matrix

# In[177]:


cm=confusion_matrix(y_test,pred)
print(cm)
sns.heatmap(cm,annot=True,cmap='BuPu')


# # Random Forest Classifier

# In[178]:


# Model 2: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model=RandomForestClassifier(n_estimators = 100, random_state = 0)
model=model.fit(X_train,y_train)
pred=model.predict(X_test)
rf_probability =model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

acc_rf=accuracy_score(y_test, y_pred)
recall_rf=recall_score(y_test, y_pred, zero_division = 1)
precision_rf=precision_score(y_test, y_pred, zero_division = 1)
f1score_rf=f1_score(y_test, y_pred, zero_division = 1)
AUC_RF=roc_auc_score(y_test, y_pred)
#print accuracy and Auc values of model
print("Accuracy : ", accuracy_score(y_test,y_pred))
print("ROC_AUC Score:",AUC_RF)


# In[179]:


print(classification_report(y_pred,y_test))


# # ROC CURVE

# In[180]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, rf_probability)

plt.title('Random Forest ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()


# # Confusion Matrix

# In[181]:


cm=confusion_matrix(y_pred,y_test)
print(cm)
sns.heatmap(cm,annot=True,cmap='RdPu')


# # Decision Tree

# In[182]:


# Model 3: Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model=DecisionTreeClassifier(random_state=0)
model=model.fit(X_train,y_train)
pred=model.predict(X_test)
dt_probability =model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

acc_dt=accuracy_score(y_test, y_pred)
recall_dt=recall_score(y_test, y_pred, zero_division = 1)
precision_dt=precision_score(y_test, y_pred, zero_division = 1)
f1score_dt=f1_score(y_test, y_pred, zero_division = 1)
AUC_DT=roc_auc_score(y_test, y_pred)
#print accuracy and Auc values of model
print("Accuracy : ", accuracy_score(y_test,y_pred))
print("ROC_AUC Score:",AUC_DT)


# In[183]:


print(classification_report(y_pred,y_test))


# # ROC CURVE

# In[184]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, dt_probability)

plt.title('Decision Tree ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()


# # Confusion Matrix

# In[185]:


cm=confusion_matrix(y_pred,y_test)
print(cm)
sns.heatmap(cm,annot=True,cmap='RdPu')


# # SVC

# In[186]:


# Model 4: SVC

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model=svm.SVC(kernel='rbf', probability=True) 
model=model.fit(X_train,y_train)
pred=model.predict(X_test)
svc_probability =model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

acc_svc=accuracy_score(y_test, y_pred)
recall_svc=recall_score(y_test, y_pred, zero_division = 1)
precision_svc=precision_score(y_test, y_pred, zero_division = 1)
f1score_svc=f1_score(y_test, y_pred, zero_division = 1)
AUC_SVC=roc_auc_score(y_test, y_pred)
#print accuracy and Auc values of model
print("Accuracy : ", accuracy_score(y_test,y_pred))
print("ROC_AUC Score:",AUC_SVC)


# In[187]:


print(classification_report(y_pred,y_test))


# # ROC CURVE

# In[188]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, svc_probability)

plt.title('SVC ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()


# # Confusion Matrix

# In[189]:


cm=confusion_matrix(y_pred,y_test)
print(cm)
sns.heatmap(cm,annot=True,cmap='RdPu')


# # KNN

# In[190]:


# Model 5: KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model=KNeighborsClassifier(n_neighbors=2)
model=model.fit(X_train,y_train)
pred=model.predict(X_test)
knn_probability =model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

acc_knn=accuracy_score(y_test, y_pred)
recall_knn=recall_score(y_test, y_pred, zero_division = 1)
precision_knn=precision_score(y_test, y_pred, zero_division = 1)
f1score_knn=f1_score(y_test, y_pred, zero_division = 1)
AUC_KNN=roc_auc_score(y_test, y_pred)
#print accuracy and Auc values of model
print("Accuracy : ", accuracy_score(y_test,y_pred))
print("ROC_AUC Score:",AUC_KNN)


# In[191]:


print(classification_report(y_pred,y_test))


# # ROC CURVE

# In[192]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, knn_probability)

plt.title('KNN ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()


# # Confusion Matrix

# In[193]:


cm=confusion_matrix(y_pred,y_test)
print(cm)
sns.heatmap(cm,annot=True,cmap='RdPu')


# # Gradient Boosting

# In[194]:


# Model 6: Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model=GradientBoostingClassifier(n_estimators=100, random_state=0)
model=model.fit(X_train,y_train)
pred=model.predict(X_test)
gb_probability =model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

acc_gb=accuracy_score(y_test, y_pred)
recall_gb=recall_score(y_test, y_pred, zero_division = 1)
precision_gb=precision_score(y_test, y_pred, zero_division = 1)
f1score_gb=f1_score(y_test, y_pred, zero_division = 1)
AUC_GB=roc_auc_score(y_test, y_pred)
#print accuracy and Auc values of model
print("Accuracy : ", accuracy_score(y_test,y_pred))
print("ROC_AUC Score:",AUC_GB)


# In[195]:


print(classification_report(y_pred,y_test))


# # ROC CURVE

# In[196]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, gb_probability)

plt.title('Gradient Boosting ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()


# # Confusion Matrix

# In[197]:


cm=confusion_matrix(y_pred,y_test)
print(cm)
sns.heatmap(cm,annot=True,cmap='RdPu')


# # AdaBoost

# In[198]:


# Model 7: AdaBoost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model=AdaBoostClassifier(n_estimators=100, random_state=0)
model=model.fit(X_train,y_train)
pred=model.predict(X_test)
ab_probability =model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

acc_ab=accuracy_score(y_test, y_pred)
recall_ab=recall_score(y_test, y_pred, zero_division = 1)
precision_ab=precision_score(y_test, y_pred, zero_division = 1)
f1score_ab=f1_score(y_test, y_pred, zero_division = 1)
AUC_AB=roc_auc_score(y_test, y_pred)
#print accuracy and Auc values of model
print("Accuracy : ", accuracy_score(y_test,y_pred))
print("ROC_AUC Score:",AUC_AB)


# In[199]:


print(classification_report(y_pred,y_test))


# # ROC CURVE

# In[200]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, ab_probability)

plt.title('AdaBoost ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()


# # Confusion Matrix

# In[201]:


cm=confusion_matrix(y_pred,y_test)
print(cm)
sns.heatmap(cm,annot=True,cmap='RdPu')


# # Models Comparision

# In[202]:


ind=['Logistic regression','Randomforest','Decisiontree','SVC','KNN','Gradientboosting','Adaboost']
data={"Accuracy":[acc_lr,acc_rf,acc_dt,acc_svc,acc_knn,acc_gb,acc_ab],
      "Recall":[recall_lr,recall_rf,recall_dt,recall_svc,recall_knn,recall_gb,recall_ab],
      "Precision":[precision_lr,precision_rf,precision_dt,precision_svc,precision_knn,precision_gb,precision_ab],
    'f1_score':[f1score_lr,f1score_rf,f1score_dt,f1score_svc,f1score_knn,f1score_gb,f1score_ab],
      "ROC_AUC":[AUC_LR,AUC_RF,AUC_DT,AUC_SVC,AUC_KNN,AUC_GB,AUC_AB]}
result=pd.DataFrame(data=data,index=ind)
result


# In[ ]:




