#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


import seaborn as sns


# In[4]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[6]:


from sklearn.metrics import classification_report, confusion_matrix


# In[7]:


df = pd.read_csv('C:\\Users\\sagar farkale\\Downloads\\avtrain.csv')


# In[215]:


df_test = pd.read_csv('C:\\Users\\sagar farkale\\Downloads\\test_lAUu6dG.csv')


# In[8]:


df.head()


# In[216]:


df_test.head()


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[217]:


df_test.isnull().sum()


# In[11]:


df['Dependents'].value_counts()


# In[12]:


df['Credit_History'].value_counts()


# In[13]:


df['Education'].value_counts()


# In[14]:


df['Loan_Amount_Term'].value_counts()


# In[15]:


df['Self_Employed'].value_counts()


# In[16]:


df['Property_Area'].value_counts()


# In[17]:


df['ApplicantIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']


# In[218]:


df_test['ApplicantIncome'] = df_test['ApplicantIncome']+df_test['CoapplicantIncome']


# In[19]:


df = df.drop('CoapplicantIncome',axis = 1)


# In[219]:


df_test = df_test.drop('CoapplicantIncome',axis = 1)


# In[20]:


df.head()


# In[220]:


df_test.head()


# In[223]:


df.describe()


# In[224]:


df_test.describe()


# In[22]:


df['Dependents'] = df['Dependents'].replace('3+','3')


# In[225]:


df_test['Dependents'] = df_test['Dependents'].replace('3+','3')


# In[23]:


df['Dependents'] = df['Dependents'].astype(float)


# In[226]:


df_test['Dependents'] = df_test['Dependents'].astype(float)


# In[24]:


df['LoanAmount'].replace(np.nan,np.mean(df['LoanAmount']),inplace=True)
df['Loan_Amount_Term'].replace(np.nan,np.mean(df['Loan_Amount_Term']),inplace=True)
df['Credit_History'].replace(np.nan,1,inplace=True)
df['Self_Employed'].replace(np.nan,'No',inplace=True)
df['Dependents'].replace(np.nan,1,inplace=True)
df['Gender'].replace(np.nan,'Male',inplace=True)
df['Married'].replace(np.nan,'Yes',inplace=True)


# In[227]:


df_test['LoanAmount'].replace(np.nan,np.mean(df_test['LoanAmount']),inplace=True)
df_test['Loan_Amount_Term'].replace(np.nan,np.mean(df_test['Loan_Amount_Term']),inplace=True)
df_test['Credit_History'].replace(np.nan,1,inplace=True)
df_test['Self_Employed'].replace(np.nan,'No',inplace=True)
df_test['Dependents'].replace(np.nan,1,inplace=True)
df_test['Gender'].replace(np.nan,'Male',inplace=True)
df_test['Married'].replace(np.nan,'Yes',inplace=True)


# In[25]:


df.isnull().sum()


# In[228]:


df_test.isnull().sum()


# In[53]:


df.info()


# In[231]:


df_test.info()


# In[80]:


X  = df.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values
y = df.iloc[:,11].values


# In[233]:


X_df_test = df_test.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values


# In[81]:


X[0]


# In[234]:


X_df_test[0]


# In[82]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_label = LabelEncoder()
X[:,0] = X_label.fit_transform(X[:,0])
X[:,1] = X_label.fit_transform(X[:,1])
X[:,3] = X_label.fit_transform(X[:,3])
X[:,4] = X_label.fit_transform(X[:,4])
X[:,9] = X_label.fit_transform(X[:,9])


# In[236]:


X_df_test[:,0] = X_label.fit_transform(X_df_test[:,0])
X_df_test[:,1] = X_label.fit_transform(X_df_test[:,1])
X_df_test[:,3] = X_label.fit_transform(X_df_test[:,3])
X_df_test[:,4] = X_label.fit_transform(X_df_test[:,4])
X_df_test[:,9] = X_label.fit_transform(X_df_test[:,9])


# In[83]:


y = X_label.fit_transform(y)


# In[85]:



from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('X[:,9]',OneHotEncoder(), [9])], remainder = 'passthrough')
X = ct.fit_transform(X)


# In[237]:


ct = ColumnTransformer([('X_df_test[:,9]',OneHotEncoder(), [9])], remainder = 'passthrough')
X_df_test = ct.fit_transform(X_df_test)


# In[86]:


X


# In[238]:


X_df_test


# In[87]:


X = X[:,1:]


# In[239]:


X_df_test = X_df_test[:,1:]


# In[88]:



y.shape


# In[198]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)


# In[ ]:





# In[90]:


X_train.shape,y_train.shape


# In[91]:


X_res.shape,y_res.shape


# In[199]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[240]:


X_df_test = sc_X.transform(X_df_test)


# In[93]:


#from imblearn.combine import SMOTETomek
#smk = SMOTETomek(random_state=42)
#X_res,y_res = smk.fit_sample(X_train,y_train)


# In[200]:


from sklearn.ensemble import RandomForestClassifier
classifierR = RandomForestClassifier(n_estimators=130,criterion='entropy',random_state=0)
classifierR.fit(X_train,y_train)


# In[201]:


y_predR = classifierR.predict(X_test)


# In[202]:


print(classification_report(y_test,y_predR))


# In[203]:


cmR = confusion_matrix(y_test,y_predR)
cmR


# In[204]:


from sklearn.model_selection import cross_val_score
accuR = cross_val_score(estimator=classifierR,X = X_train,y = y_train,cv=10)
print(accuR)
print(accuR.mean())
print(accuR.std())


# In[210]:


from sklearn.model_selection import GridSearchCV
parametersR = [{'n_estimators':[50,75,100,120,130,140,150,160,170],'criterion':['entropy','gini']}]
grid_searchR = GridSearchCV(estimator = classifierR,
                           param_grid = parametersR,
                           scoring = 'accuracy',
                           cv= 10, n_jobs = -1)
grid_searchR  = grid_searchR.fit(X_train, y_train)       
best_accuracyR  = grid_searchR.best_score_
best_parametersR = grid_searchR.best_params_
best_accuracyR , best_parametersR


# In[205]:


from sklearn.svm import SVC
classifierS = SVC(kernel='linear',gamma=0.001,C= 0.5,random_state=0)
classifierS.fit(X_train,y_train)


# In[206]:


y_predS = classifierS.predict(X_test)


# In[207]:


print(classification_report(y_test,y_predS))


# In[208]:


cmS = confusion_matrix(y_test,y_predS)
cmS


# In[209]:


from sklearn.model_selection import cross_val_score
accuS = cross_val_score(estimator=classifierS,X = X_train,y = y_train,cv=10)
print(accuS)
print(accuS.mean())
print(accuS.std())


# In[167]:


from sklearn.model_selection import GridSearchCV
parametersS = [{'C':[0.01,0.05,0.5],'gamma':[0.00001,0.00005,0.0001,0.0005],'kernel':['rbf','linear','poly']}]
grid_searchS = GridSearchCV(estimator = classifierS,
                           param_grid = parametersS,
                           scoring = 'accuracy',
                           cv= 10, n_jobs = -1)
grid_searchS  = grid_searchS.fit(X_train, y_train)       
best_accuracyS  = grid_searchS.best_score_
best_parametersS = grid_searchS.best_params_
best_accuracyS , best_parametersS


# In[168]:


from xgboost import XGBClassifier
classifierX = XGBClassifier(gamma = 1, learning_rate = 0.005, n_estimators = 75)
classifierX.fit(X_train,y_train)


# In[169]:


y_predX = classifierX.predict(X_test)


# In[170]:


print(classification_report(y_test,y_predX))


# In[171]:


cmX = confusion_matrix(y_test,y_predX)
cmX


# In[172]:


from sklearn.model_selection import cross_val_score
accuX = cross_val_score(estimator=classifierX,X = X_train,y = y_train,cv=10)
print(accuX)
print(accuX.mean())
print(accuX.std())


# In[173]:


from sklearn.model_selection import GridSearchCV
parametersX = [{'n_estimators':[60,70,75,80,85,90],'gamma':[1,5,10,15],'learning_rate':[0.005,0.01,0.05]}]
grid_searchX = GridSearchCV(estimator = classifierX,
                           param_grid = parametersX,
                           scoring = 'accuracy',
                           cv= 10, n_jobs = -1)
grid_searchX = grid_searchX.fit(X_train, y_train)       
best_accuracyX  = grid_searchX.best_score_
best_parametersX = grid_searchX.best_params_
best_accuracyX , best_parametersX


# In[250]:


sns.distplot(df["ApplicantIncome"], kde = bool)


# In[211]:


from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train,y_train)


# In[212]:


y_predNB = classifierNB.predict(X_test)


# In[213]:


print(classification_report(y_test,y_predNB))


# In[214]:


cmNB = confusion_matrix(y_test,y_predNB)
cmNB


# In[179]:


from sklearn.model_selection import cross_val_score
accuNB = cross_val_score(estimator=classifierNB,X = X_train,y = y_train,cv=10)
print(accuNB)
print(accuNB.mean())
print(accuNB.std())


# In[ ]:


from sklearn.model_selection import GridSearchCV
parametersNB = [{}]
grid_searchNB = GridSearchCV(estimator = classifierNB,
                           param_grid = parametersNB,
                           scoring = 'accuracy',
                           cv= 10, n_jobs = -1)
grid_searchNB = grid_searchNB.fit(X_train, y_train)       
best_accuracyNB  = grid_searchNB.best_score_
best_parametersNB = grid_searchNB.best_params_
best_accuracyNB , best_parametersNB


# In[182]:


nb = 12800.0/154
nb


# In[184]:


x = 12500.0/154
x


# In[185]:


s = 12800.0/154
s


# In[191]:


r = 12500.0/154
r


# In[192]:


sample = pd.read_csv('C:\\Users\\sagar farkale\\Downloads\\sample_submission_49d68Cx (1).csv')


# 

# In[193]:


sample


# In[341]:


y_predRR = classifierR.predict(X_df_test)


# In[305]:


y_predSS= classifierS.predict(X_df_test)


# In[243]:


y_predXX = classifierX.predict(X_df_test)


# In[292]:


y_predNBB = classifierNB.predict(X_df_test)


# In[293]:


y_predNBB = pd.DataFrame(y_predNBB, columns = ['Loan_Status'])


# In[294]:


y_predNBB['Loan_Status'] = y_predNBB['Loan_Status'].map({1:"Y",0:"N"})


# In[298]:


sub_nb = pd.concat([df_test['Loan_ID'], y_predNBB], axis=1)


# In[306]:


y_predSS = pd.DataFrame(y_predSS, columns = ['Loan_Status'])
y_predSS['Loan_Status'] = y_predSS['Loan_Status'].map({1:"Y",0:"N"})
sub_svm = pd.concat([df_test['Loan_ID'], y_predSS], axis=1)


# In[312]:


sub_svm.to_csv('LoansubSVM.csv',index = False)


# In[314]:


sub_nb.to_csv('LoansubNB.csv',index=False)


# In[318]:


c = pd.concat([sub_svm,df_test['Credit_History']],axis=1)


# In[322]:


c['Loan_Status']=c['Loan_Status'].map({'Y':1,'N':0})


# In[325]:


c['Loan_Status'] = c['Loan_Status']*c['Credit_History']


# In[327]:


c['Loan_Status']=c['Loan_Status'].map({1:'Y',0:'N'})


# In[333]:


c.drop('Credit_History',axis=1,inplace=True)


# In[335]:


c.to_csv('Credit_History.csv',index= False)


# In[336]:


c


# In[342]:


y_predRR = pd.DataFrame(y_predRR, columns = ['Loan_Status'])
y_predRR['Loan_Status'] = y_predRR['Loan_Status'].map({1:"Y",0:"N"})
sub_rf = pd.concat([df_test['Loan_ID'], y_predRR], axis=1)


# In[344]:


sub_rf.to_csv('LoansubRF.csv',index=False)


# In[343]:


sub_rf


# In[ ]:




