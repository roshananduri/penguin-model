import pandas as pd
import streamlit as st
penguin=pd.read_csv('penguins_cleaned.csv')
df=penguin.copy()

target=['species']
encode = ['island','sex']

for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]

target_mapper={'Adelie':0,'Chinstrap':1,'Gentoo':2}

def target_encoder(val):
    return target_mapper[val]

df['species']=df['species'].apply(target_encoder)
X=df.drop('species',axis=1)
y=df['species']

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

clf=xgb.XGBClassifier()
clf.fit(X,y)

import pickle
pickle.dump(clf,open('penguins_clf.pkl','wb'))