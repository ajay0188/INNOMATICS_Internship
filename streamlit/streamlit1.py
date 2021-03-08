import streamlit as st

import pandas as pd

import seaborn as Sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pickle import load
import numpy as np
from sklearn import metrics

def predict(arr):
    classifier=load(open('pickle/random_model.pkl','rb'))
    prediction=classifier.predict(arr)
    return prediction

col1=st.number_input('Enter col1 values ')
col2=st.number_input('Enter col1 values')
arr = np.array([col1, col2]).reshape(1,-1)
arr = arr.astype('float64')
prediction = predict(arr)
click = st.button('SUBMIT')
if click:
    if(arr.any()):
        for i in prediction:
            if i==0:
                st.write('yes')
            else:
                st.write("no")
