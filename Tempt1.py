#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
# import pip
# pip.main(["install","streamlit"])

import streamlit as st
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[32]:


st.title("Diabetes Detection System")


# In[4]:


image=Image.open('E:\\DS Course\\Diabetes-Prediction-with-ML\\home.jpg')


# In[5]:


image


# In[40]:


st.image(image)


# In[7]:


data=pd.read_csv('E:\DS Course\Diabetes-Prediction-with-ML\Diabetes\diabetes.csv')


# In[8]:


data.head()


# In[9]:


st.subheader("Data")


# In[10]:


st.write(data)


# In[37]:


data.iloc[:,:8].describe()


# In[36]:


st.subheader("Data Description")
st.markdown("**Data Description**", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    table {
        color: white;
        background-color: #336699;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write(data.iloc[:,:8].describe())


# In[14]:


x=data.iloc[:,:8].values
y=data.iloc[:,8].values


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[16]:


model=RandomForestClassifier(n_estimators=500)


# In[17]:


model.fit(x_train,y_train)


# In[18]:


y_pred=model.predict(x_test)


# In[19]:


accuracy_score(y_test,y_pred)


# In[20]:


st.subheader("Accuracy of the Trained Model")


# In[21]:


st.write(accuracy_score(y_test,y_pred))


# In[22]:


# st.text_input("Enter Your Age")


# In[23]:


# st.text_area(label="Describe you")


# In[38]:


st.subheader("Enter Your Input Data")
st.markdown("### Enter Your Input Data", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .input-section {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="input-section">', unsafe_allow_html=True)
# Rest of the code for user input section
st.markdown('</div>', unsafe_allow_html=True)


# In[25]:


def user_input():
  preg=st.slider("Pregnancy",0,20,0)
  glu=st.slider("Glucose",0,200,0)
  bp=st.slider("Blood Pressure",0,130,0)
  sthick=st.slider("Skin Thickness",0,100,0)
  ins=st.slider("Insulin",0.0,100.0,0.0)
  bmi=st.slider("BMI",0.0,1000.0,0.0)
  dpf=st.slider("Diabete Predigree Function",0.000,3.000,0.000)
  age=st.slider("Age",0,100,0)

  input_dict={"Pregnancy":preg,"Glucose":glu,"Blood Pressure":bp,"Skin Thickness":sthick,
             "Insulin":ins,"BMI":bmi,"Diabete Predigree Function":dpf,"Age":age}
  return pd.DataFrame(input_dict,index=["User Input Values"])


# In[26]:


ui=user_input()


# In[27]:


st.write("Entered Input Data")


# In[28]:


st.write(ui)


# In[31]:


st.subheader("Prediction")


# In[30]:


prediction = model.predict(ui)

if prediction == 1:
    st.write("You have Diabetes")
else:
    st.write("You are free from Diabetes")


# In[ ]:




