import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("artifacts/framingham.csv")
# print(df.head())

df['education'].fillna(df['education'].mean(),inplace=True)
df['cigsPerDay'].fillna(df['cigsPerDay'].median(),inplace=True)
df['totChol'].fillna(df['totChol'].mean(),inplace=True)
df['BMI'].fillna(df['BMI'].median(),inplace=True)
df['glucose'].fillna(df['glucose'].mean(),inplace=True)
df['BPMeds'].fillna(df['BPMeds'].mean(),inplace=True)
df.dropna(axis=0,inplace=True)

# print(df.info())

sns.boxplot(y=df['cigsPerDay'])

df.loc[df['cigsPerDay']>50,'cigsPerDay']=df.loc[df['cigsPerDay']<50,'cigsPerDay'].mean()

sns.boxplot(y=df['cigsPerDay'])

df['BPMeds']=np.around(df['BPMeds'],1)

# sns.boxplot(y=df['totChol'])

q1=df['totChol'].quantile(0.25)
q2=df['totChol'].quantile(0.5)
q3=df['totChol'].quantile(0.75)
iqr=q3-q1
upper_tail=q3+1.5*iqr
lower_tail=q1-1.5*iqr
mean=df[(df['totChol']<upper_tail) | (df['totChol']>lower_tail)]['totChol'].mean()
for j,i in np.ndenumerate(df['totChol']):
    if i>upper_tail or i<lower_tail:
        df['totChol'].replace({i:mean},inplace=True)

# sns.boxplot(y=df['totChol'])

# sns.boxplot(y=df['sysBP'])

q1=df['sysBP'].quantile(0.25)
q2=df['sysBP'].quantile(0.5)
q3=df['sysBP'].quantile(0.75)
iqr=q3-q1
upper_tail=q3+1.5*iqr
lower_tail=q1-1.5*iqr
mean=df[(df['sysBP']<upper_tail) | (df['sysBP']>lower_tail)]['sysBP'].mean()
for j,i in np.ndenumerate(df['sysBP']):
    if i>upper_tail or i<lower_tail:
        df['sysBP'].replace({i:mean},inplace=True)

# sns.boxplot(y=df['sysBP'])

# sns.boxplot(y=df['diaBP'])

q1=df['diaBP'].quantile(0.25)
q2=df['diaBP'].quantile(0.5)
q3=df['diaBP'].quantile(0.75)
iqr=q3-q1
upper_tail=q3+1.5*iqr
lower_tail=q1-1.5*iqr
mean=df[(df['diaBP']<upper_tail) | (df['diaBP']>lower_tail)]['diaBP'].mean()
for j,i in np.ndenumerate(df['diaBP']):
    if i>upper_tail or i<lower_tail:
        df['diaBP'].replace({i:mean},inplace=True)

# sns.boxplot(y=df['diaBP'])

# sns.boxplot(y=df['BMI'])

q1=df['BMI'].quantile(0.25)
q2=df['BMI'].quantile(0.5)
q3=df['BMI'].quantile(0.75)
iqr=q3-q1
upper_tail=q3+1.5*iqr
lower_tail=q1-1.5*iqr
mean=df[df['BMI']<upper_tail]['BMI'].mean()
for j,i in np.ndenumerate(df['BMI']):
    if i>upper_tail:
        df['BMI'].replace({i:mean},inplace=True)

# sns.boxplot(y=df['BMI'])

# sns.boxplot(y=df['heartRate'])

q1=df['heartRate'].quantile(0.25)
q2=df['heartRate'].quantile(0.5)
q3=df['heartRate'].quantile(0.75)
iqr=q3-q1
upper_tail=q3+1.5*iqr
lower_tail=q1-1.5*iqr
mean=df[df['heartRate']<upper_tail]['heartRate'].mean()
for j,i in np.ndenumerate(df['heartRate']):
    if i>upper_tail:
        df['heartRate'].replace({i:mean},inplace=True)

# sns.boxplot(y=df['heartRate'])

# sns.boxplot(y=df['glucose'])

q1=df['glucose'].quantile(0.25)
q2=df['glucose'].quantile(0.5)
q3=df['glucose'].quantile(0.75)
iqr=q3-q1
upper_tail=q3+1.5*iqr
lower_tail=q1-1.5*iqr
mean=df[(df['glucose']<upper_tail) | (df['glucose']>lower_tail)]['glucose'].mean()
for j,i in np.ndenumerate(df['glucose']):
    if i>upper_tail or i<lower_tail:
        df['glucose'].replace({i:mean},inplace=True)

# sns.boxplot(y=df['glucose'])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

x=df.drop('TenYearCHD',axis=1)
y=df['TenYearCHD']

std=StandardScaler()
x_scaled=std.fit_transform(x)
x=pd.DataFrame(x_scaled,columns=x.columns)

from imblearn.over_sampling import RandomOverSampler

ros=RandomOverSampler()
x_ros,y_ros=ros.fit_resample(x,y)

xtr,xte,ytr,yte=train_test_split(x_ros,y_ros,train_size=0.8,
                                 stratify=y_ros,random_state=1)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(xtr, ytr)

ypredrfc = rfc.predict(xte)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(ypredrfc, yte)

with mlflow.start_run():
    mlflow.log_param("n_estimators", rfc.n_estimators)
    mlflow.log_param("max_depth", rfc.max_depth)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(rfc, "random_forest_model")
