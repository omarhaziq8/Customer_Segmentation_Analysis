# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:20:15 2022

@author: pc
"""

#-->> DATA SCREENING <<--
# From the CSV file we can analyse that:
# it is categorical data
# id,days_since_prev_campaign,month is not include in pattern/categorical, can drop
# balance,day_of_month,month,last_contact_duration,num_contacts 
# term_deposit_subscribed is TARGET/OUTPUT
# categorical vs categorical used Cramer's V
# classification model 

#-->> STEPS <<--
# Data Loading
# Data Inspection
# Data Cleaning 
# Features Selection
# Preprocessing
# Model Development
# Deep Learning
# Model Evaluation

import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

from ModulesClass_Customer import ModelCreation,Model_Evaluation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping 


#%% Function

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


#%% Statics

CSV_PATH = os.path.join(os.getcwd(),'Train.csv')
JOBTYPE_ENCODER_PATH = os.path.join(os.getcwd(),'Jobtype_encoder.pkl')
MARITAL_ENCODER_PATH = os.path.join(os.getcwd(),'Marital_encoder.pkl')
EDUCATION_ENCODER_PATH = os.path.join(os.getcwd(),'Education_encoder.pkl')
DEFAULT_ENCODER_PATH = os.path.join(os.getcwd(),'Default_encoder.pkl')
HOUSING_LOAN_ENCODER_PATH = os.path.join(os.getcwd(),'HousingLoan_encoder.pkl')
PERSONAL_LOAN_ENCODER_PATH = os.path.join(os.getcwd(),'PersonalLoan_encoder.pkl')
COMMUNICATION_TYPE_ENCODER_PATH = os.path.join(os.getcwd(),'CommType_encoder.pkl')
PREVIOUS_ENCODER_PATH = os.path.join(os.getcwd(),'Previous_encoder.pkl')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

#%% Data Loading

df = pd.read_csv(CSV_PATH)

#%% Data Inspection 

# To check dtype of data features
df.info()\
# To check the name of the 18 columns
df.columns
# 0   id
# 1   customer_age
# 2   job_type
# 3   marital
# 4   education
# 5   default
# 6   balance
# 7   housing_loan
# 8   personal_loan
# 9   communication_type
# 10  day_of_month
# 11  month
# 12  last_contact_duration
# 13  num_contacts_in_campaign
# 14  days_since_prev_campaign_contact
# 15  num_contacts_prev_campaign
# 16  prev_campaign_outcome
# 17  term_deposit_subscribed

# To check statistics for the data
# temp = df.describe()
# print(temp)

# Categorical column
cat_column = ['job_type','marital','education','default',
              'housing_loan','personal_loan','communication_type',
              'prev_campaign_outcome','term_deposit_subscribed']# can see from info it shows object dtype
# Continous column
con_column = ['customer_age','balance','day_of_month','last_contact_duration',
              'num_contacts_in_campaign','num_contacts_prev_campaign']

# continous data
for i in con_column:
    plt.figure()
    sns.distplot(df[i])
    plt.show()

# categorical data
for i in cat_column:
    plt.figure()
    sns.countplot(df[i])
    plt.show()

# To visualise based on count to see the correlation by using groupby
df.groupby(['term_deposit_subscribed','job_type']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','job_type','marital','education']).agg({'term_deposit_subscribed':'count'})

# To visualise bar plot for categorical data
for i in cat_column:
    
    plt.figure()
    sns.countplot(df[i],hue=df['term_deposit_subscribed']) # hue is category
    plt.show()

#%% Data Cleaning

# To check the NaN value
df.isnull().sum()
# customer_age:619
# marital: 150
# balance: 399
# personal_loan: 149
# last_contact_duration: 311
# num_contacts_in_campaign: 112
# days_since_prev_campaign_contact: 25831 
df.duplicated().sum()
# No duplicated value

# since days_since_prev_campaign_contact has so many Nans value, need to drop
# id and month do not have correlation
df = df.drop(labels='days_since_prev_campaign_contact',axis=1)
df = df.drop(labels='id',axis=1)
df = df.drop(labels='month',axis=1)

df_dummy = df.copy() # To create copy incase has mistake in term of conversion


# To change all the dtype object into numerical by using labelEncoder
le = LabelEncoder()

paths = [JOBTYPE_ENCODER_PATH,MARITAL_ENCODER_PATH,EDUCATION_ENCODER_PATH,
         DEFAULT_ENCODER_PATH,HOUSING_LOAN_ENCODER_PATH,PERSONAL_LOAN_ENCODER_PATH,
         COMMUNICATION_TYPE_ENCODER_PATH,PREVIOUS_ENCODER_PATH]

for index,i in enumerate(cat_column):
    temp = df_dummy[i]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()]) 
    # To extract elements which is not null and overide it
    df_dummy[i] = pd.to_numeric(temp,errors='coerce')
    with open(paths[7],'wb') as file:
        pickle.dump(le,file)

# To impute the NaN value inside each column by using KNNImputer

knn_imp = KNNImputer()
df_dummy = knn_imp.fit_transform(df_dummy)
df_dummy = pd.DataFrame(df_dummy)
df_dummy.columns = df.columns

# So, no NaN value inside, can proceed to Features selection,preprocessing and DL

#%% Features selection

# categorical vs categorical
# Cramer's V

for i in cat_column:
    print(i)
    confussion_mat = pd.crosstab(df_dummy[i],df_dummy['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))

# job_type
# 0.13596388961197914
# marital
# 0.06339946467275404
# education
# 0.07207191785440015
# default
# 0.018498692474409054
# housing_loan
# 0.144442194297714
# personal_loan
# 0.06592546992472705
# communication_type
# 0.14713602417060107
# prev_campaign_outcome
# 0.3410607961880476  >> 2nd highest correlation but still low score which is 34%
# term_deposit_subscribed
# 0.9998349787513988  >> highest cause it compared to itself(target), not going to select

# continous vs categorical
# logistic regression
for con in con_column:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df_dummy[con], axis=-1),df_dummy['term_deposit_subscribed'])
    print(lr.score(np.expand_dims(df_dummy[con], axis=-1),df_dummy['term_deposit_subscribed']))

# customer_age
# 0.892754447498973
# balance
# 0.8925648560685057
# day_of_month
# 0.892754447498973
# last_contact_duration
# 0.8997061332827756  >> Highest score 
# num_contacts_in_campaign
# 0.892754447498973
# num_contacts_prev_campaign
# 0.8918064903466363

# Finalise to be select features
# >> last_contact_duration,prev_campaign_outcome,
# customer_age,num_contacts_in_campaign

x = df_dummy.loc[:,['last_contact_duration','prev_campaign_outcome','customer_age','num_contacts_in_campaign']]
y = df_dummy['term_deposit_subscribed']


#%% Preprocessing for OHE (Target)

# For DL training, the target need to change to OHE, so that it would in array

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))
# model ohe saving
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)

#%% Model Development

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,
                                                 random_state=123)


mc = ModelCreation()
model = mc.simple_bn_layer(x_train,num_node=32)

# nb_features = np.shape(x_train)[1:]
# nb_classes = len(np.unique(y_train,axis=0))

# model = Sequential() # to create container 
# model.add(Input(shape=(nb_features)))
# model.add(Dense(32,activation='relu',name='HL1'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(32,activation='relu',name='HL2'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(nb_classes,activation='relu',name='Output'))
# model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics='acc')

# To plot model flowchart
plot_model(model,show_layer_names=(True),show_shapes=(True))

#%% Callback

early_stopping_callback = EarlyStopping(monitor='loss', patience=3)
LOG_PATH = os.path.join(os.getcwd(),'Logs')
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
log_dir = datetime.datetime.now()

hist = model.fit(x_train,y_train,batch_size=64,epochs=10,
                 validation_data=(x_test,y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])


# From the epoch training, as it ends at 10 as I set into 10 epochs to avoid 
# any flaws for the DL training,also to set as minimal time 
# for the algorithm working the entire dataset in short time
# i intro the training with earlystopping callback to prevent overfitting
# From the training, we can observe the acc is  increasing by the epoch
# but the val_acc is getting decrease cause of the stopping_callback function
# in order to prevent it from overfitting.
# The accuracy is nearly 90% which is already good for model deployment


#%% Plot Visualisation

# Use history.keys to display the keyword loss,val_loss,acc,val_acc
hist.history.keys()
# Use Module class to potray the graph within 2 lines of codes!
hist_me = Model_Evaluation()
hist_me.plot_hist_graph(hist)

# plt.figure()
# plt.plot(hist.history['loss'],label='Training loss')
# plt.plot(hist.history['val_loss'],label='Validation loss')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(hist.history['acc'],label='Training acc')
# plt.plot(hist.history['val_acc'],label='Validation acc')
# plt.legend()
# plt.show()

# From the graph plot, it shows abit underfitting 
# but the pattern still acceptable, we can increase epoch,
# get the model development into complex which is increase layer of dense
# increase dropout rate for NN,increase training data


#%% Model evaluation 

y_true = y_test 
y_pred = model.predict(x_test)

y_true = np.argmax(y_true,axis=1) # to convert 0/1 
y_pred = np.argmax(y_pred,axis=1)

print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))

# To conclude, the accuracy score for DL training is 90% 
# Therefore, it can be use for model deployment 

#%% Model H5 saving 

#H5 model save
model.save(MODEL_SAVE_PATH)























