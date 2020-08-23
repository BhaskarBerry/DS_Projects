# -*- coding: utf-8 -*-
"""
1. To Simple Impute all the missing values for the numerical variables and we 
   are going to impute CONSTANT values for SEX
2. One hot encode the nominal variables -> ISLAND and SEX

Column Transformers can be used
1. SimpleImputer
2. One hot encoding
3. Label encoding
4. Scaling
5. CountVectorizer
6. Custom transformers are also available to suit our needs.

Pro-tip => In case of pipeline and transformers if the column values contains 
           NULL or N/A values always consider imputing them, before passing 
           them to a column transformer for one hot encoding or scaling.

The above can be achieved through creating a pipeline first then passing this 
pipeline as a parameter for column transfomer. 
If not you can directly pass them to the column transformer.
"""
# Import Libraries
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#make_pipeline and make_column_transformer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold


import eli5

import warnings
import sklearn
sklearn.__version__ #0.23.2

#-------------------------------------------
# set options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#-------------------------------------------
#Read the data  
path = os.path.abspath("Data/Penguins.csv")

df = pd.read_csv(path)

df.head()
df.info()
df.columns

df['species'].value_counts()
df['sex'].value_counts()

#-------------------------------------------
#EDA
for i in df.select_dtypes(include = 'object').columns:
    print('Categories present in {}'.format(i))
    print(df[i].value_counts())
    sns.countplot(df[i])
    plt.show()
    
sns.violinplot(x=df['species'], y=df['body_mass_g'])
plt.show()

#checking the null values
df.isnull().sum()

sns.pairplot(df, hue= 'species')

cor=df[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']].corr()

sns.heatmap(cor,annot=True,cmap='plasma', vmin=-1, vmax=1)

## function to convert Target as label
def func(a):
    dict_y = {'Adelie':0,'Gentoo':1,'Chinstrap':2}
    return(dict_y[a])
#-------------------------------------------
# Handling missing values - make_pipeline and make_column_transformer 

target = df['species']
target = target.apply(func)
features = df[['island', 'bill_length_mm', 'bill_depth_mm',
               'flipper_length_mm', 'body_mass_g', 'sex']].copy()
    
##Splitting into train test split and since we have data imbalance we can also 
##include stratify  parameter        

X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,
                                                 random_state=10,
                                                 stratify=target)

#-------------------------------------------
#missing values , lets compute it with constant now

imp_constant = SimpleImputer(strategy='constant',fill_value='missing')
ohe = OneHotEncoder()
imp = SimpleImputer()

# Create a pipline by first passing the imputer and then the transformer

imp_ohe = make_pipeline(imp_constant, ohe)

#Name out the columns , Numeric Columns and ctegorical features Seperately

num_features= features.dtypes == 'float'
cat_features = ~num_features

#sklearn.23.0 we are using the parameter - remainder= 'passthrough' ,this will
# not apply to other columns , there are other parameters can go through them as well
ct = make_column_transformer(
        (imp_ohe,['island','sex']),
        (imp, num_features),
        remainder='passthrough')

ct.fit_transform(X_train)
ct.named_transformers_

ct.named_transformers_['pipeline'].named_steps['onehotencoder'].get_feature_names()
#-------------------------------------------
"""
Now we have created column transformers let us see how to add them to a pipeline
Step 1: Create an object of the ML algorithm you are going to execute.
Step 2: Make a pipeline by combining the column transformer and the object created for ML algorithm.
Step 3: Fit the model, this will have all the properties of the model.
Step 4: Pass the above and predict it against any new/test data set.
"""
logReg =LogisticRegression(solver='liblinear', random_state =1)
#use calss_weight parameter incase of data imbalance

pipe = make_pipeline(ct,logReg)
pipe.fit(X_train,y_train)

pipe.predict(X_test)

print("logistic regreesion score : %f" %pipe.score(X_test, y_test))
#----------------------------------------------
pipe
pipe.named_steps.columntransformer
pipe.named_steps.logisticregression
pipe.named_steps.logisticregression.coef_
#----------------------------------------------

num_features = df.select_dtypes(include=['int64', 'float64']).columns
onehot_columns = list(ct.named_transformers_['pipeline'].
                      named_steps['onehotencoder'].get_feature_names())

num_features_list = list(num_features)
num_features_list.extend(onehot_columns)

#----------------------------------------------
from sklearn import set_config
set_config(display='diagram')
pipe
#----------------------------------------------
eli5.explain_weights(pipe.named_steps['logisticregression'], 
                     top=10,feature_names=num_features_list)    

#----------------------------------------------
#Grid Search CV and cross_val_Score can also be performed using pipe
from sklearn.model_selection import GridSearchCV
pipe.named_steps.keys()
params = {}
params['logisticregression__penalty'] = ['l1','l2']
params['logisticregression__C'] = [0.1, 1, 10]
params
#----------------------------------------------
"""
Steps to be observed for the parameter tuning is that
Step 1: Create a dictionary for parameter tuning
Step 2: Step name has to be created followed by two underscores and finally the
 paramater name , for eg in the above logisticregression followed by penalty 
 (here it is the parameter that is used)
"""
gd_Search= GridSearchCV(pipe, params, cv=5, scoring='accuracy')
gd_Search.fit(X_train,y_train);


Output_df = pd.DataFrame(gd_Search.cv_results_)
Output_df

Output_df.sort_values('rank_test_score')

#let us try to see the contents in our named transformer
pipe.named_steps.columntransformer.named_transformers_

from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X_train, y_train, cv=5)
#----------------------------------------------
import joblib
joblib.dump(pipe, 'pipeline.joblib')
pipeline_from_joblib = joblib.load('pipeline.joblib')
pipeline_from_joblib.predict(X_test)
#----------------------------------------------
"""
Summary:
   ML Pipeline
"""    
