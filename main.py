# importing all the libraries
import io
from itertools import groupby
from nbformat import write
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn import preprocessing

# initialize dataset
heart_disease_df = pd.read_csv(r'~/Desktop/Documents/repos/heart_disease_forecasting/heart_2020_cleaned.csv')

# all columns names to lowercase
heart_disease_df.columns = heart_disease_df.columns.map(lambda x : x.lower())

# defining the url where the data have been taken
url = 'https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease'

# titles
st.markdown("<h1 style='text-align: center; color: brown;'>Heart Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: grey;'>Programming Project 2021/2022</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Michael Fidanza - VR472909</h3>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'><a href=" + url + ">dataset source</a></h6>", unsafe_allow_html=True)

# empty spaces
st.text('')
st.text('')
st.text('')

st.markdown("<h2 style='text-align: center; color: black;'>Data Exploration</h2>", unsafe_allow_html=True)

# empty spaces
st.text('')
st.text('')

# show the raw dataset
st.markdown("<h6 style=color: black;'>Raw data</h6>", unsafe_allow_html=True)
st.write(heart_disease_df)

# empty spaces
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show the correlation matrix
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# convert non numerical values to numbers using an encoder from sklearn
# initialize the encoder
le = preprocessing.LabelEncoder()

# make a copy of the dataframe
heart_disease_df_encoded = heart_disease_df.copy()

# find the columns which need encoding
categorical_columns = heart_disease_df_encoded.dtypes[heart_disease_df_encoded.dtypes != 'float64'].index.to_list()

# and encode them
for col in categorical_columns:
    heart_disease_df_encoded[col] = le.fit_transform(heart_disease_df_encoded[col])

# correlation matrix
st.markdown("<h6 style=color: black;'>Correlation Matrix</h6>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(heart_disease_df_encoded.corr(), annot=True, ax=ax, cmap='Reds')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)
st.write("As we can see from the correlation plot, there is no strong correlation between single variables. \
We can see that heart disease has the strongest positive correlation with alcoholdrinking, physicalhealth, diffwalking, agecategory, diabetic and kidneydisease, but in any case they're weak correlation ~ 20 %")

# empty spaces
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show how many people in the dataset have heart disease vs how many don't have heart disease
# ----------------------------------------------------------------------------------------------------------------------------------------------------

fig2 = plt.figure(figsize=(3,3))
ax2 = fig2.add_subplot(111)
ax2.pie(
    [len(heart_disease_df_encoded['heartdisease'] - sum(heart_disease_df_encoded['heartdisease'])),
     sum(heart_disease_df_encoded['heartdisease'])]
    ,labels=("Healthy", "Heart disease")
    ,explode=(0, 0.2)
    ,startangle=45
    ,autopct='%1.1f%%'
    ,shadow=True
    ,colors=['bisque', 'indianred']
    )
plt.title("How many people in the dataset have heartdisease?", fontsize=8)
st.pyplot(fig2)
st.write("By looking at the proportion between people who had/have heart disease and those who don't, \
    we notice that the dataset is unbalanced, and this will be a problem later when applying ML algorithms")

# empty spaces
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show how BMI is related to other variables
# ----------------------------------------------------------------------------------------------------------------------------------------------------

fig4, ax4 = plt.subplots(3, 2, figsize=(20, 40))

# order by age category to obtain better graph
heart_disease_df_sorted = heart_disease_df.sort_values(by='agecategory')

sns.boxplot(
    x = 'sex'
    ,y = 'bmi'
    ,data = heart_disease_df
    ,ax=ax4[0,0]
)
sns.boxplot(
    x = 'agecategory'
    ,y = 'bmi'
    # use the sorted df to obtain better results graphically
    ,data = heart_disease_df_sorted
    ,ax=ax4[0,1]
)
sns.boxplot(
    x = 'race'
    ,y = 'bmi'
    ,data = heart_disease_df
    ,ax=ax4[1,0]
)
sns.boxplot(
    x = 'diabetic'
    ,y = 'bmi'
    ,data = heart_disease_df
    ,ax=ax4[1,1]
)
sns.boxplot(
    x = 'diffwalking'
    ,y = 'bmi'
    ,data = heart_disease_df
    ,ax=ax4[2,0]
)
sns.boxplot(
    x = 'physicalactivity'
    ,y = 'bmi'
    ,data = heart_disease_df
    ,ax=ax4[2,1]
)

ax4[0,1].tick_params(axis='x', rotation=45)
ax4[1,0].tick_params(axis='x', rotation=20)
ax4[1,1].tick_params(axis='x', rotation=20)
ax4[0,0].set_title('Are BMI and sex related?', fontsize=15)
ax4[0,1].set_title('Are BMI and age category related?', fontsize=15)
ax4[1,0].set_title('Are BMI and race related?', fontsize=15)
ax4[1,1].set_title('Are BMI and diabetic state related?', fontsize=15)
ax4[2,0].set_title('Are BMI and difficulty in walking related?', fontsize=15)
ax4[2,1].set_title('Are BMI and physical activity related?', fontsize=15)

fig4.suptitle('How  BMI is related to other variables', fontsize=20)

st.pyplot(fig4)

# empty spaces
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show percentages of diabetic people for each age category
# ----------------------------------------------------------------------------------------------------------------------------------------------------

fig3, ax3 = plt.subplots(1, 2, figsize=(20,10))

# encode values manually to decide how to encode them
heart_disease_df_sorted['diabetic'] = heart_disease_df_sorted['diabetic'].map({'Yes' : 1, 'No' : 0, 'Yes (during pregnancy)' : 1, 'No, borderline diabetes' : 0})

# group by age category to find the total number of people and of people with diabtes for each age category
diabetic_statistics = heart_disease_df_sorted.groupby('agecategory')['diabetic'].aggregate([sum, 'count'])

# calculate the percentage using sum and count
diabetic_statistics['percentage'] = diabetic_statistics['sum'] / diabetic_statistics['count'] * 100

# graph for age distribution
ax3[0].bar(diabetic_statistics.index, diabetic_statistics['count'], align='center', alpha=0.5)
# graph about percentage of diabetic people for age category
ax3[1].bar(diabetic_statistics.index, diabetic_statistics['percentage'], align='center', alpha=0.5)

ax3[0].set_title('Distribution of age category', fontsize=15)
ax3[1].set_title('% of diabetic people for each age category', fontsize=15)

ax3[0].tick_params(axis='x', rotation=45)
ax3[1].tick_params(axis='x', rotation=45)

st.pyplot(fig3)
st.write("We can notice that as age grows, also the probability to be diabetic grows")

# empty spaces
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show how age category is related to other variables
# ----------------------------------------------------------------------------------------------------------------------------------------------------


fig5, ax5 = plt.subplots(figsize=(3,3))

# divide the web page in 2 columns
col1, col2 = st.columns(2)

# use the sorted df and map the variables with integer values
heart_disease_df_sorted['stroke'] = heart_disease_df_sorted['stroke'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['smoking'] = heart_disease_df_sorted['smoking'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['alcoholdrinking'] = heart_disease_df_sorted['alcoholdrinking'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['skincancer'] = heart_disease_df_sorted['skincancer'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['heartdisease'] = heart_disease_df_sorted['heartdisease'].map({"Yes" : 1, "No" : 0})

# calculate % of "positive" to the features by age category
heart_disease_df_avg = heart_disease_df_sorted[['agecategory', 'stroke', 'smoking', 'alcoholdrinking', 'skincancer', 'heartdisease']].groupby('agecategory').mean()

with col1:
    ax5.bar(
        heart_disease_df_avg.index
        ,heart_disease_df_avg['stroke']*100
        ,align='center'
        ,alpha=0.5
    )
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('Relation between agecategory and stroke')
    st.write(fig5)

fig5, ax5 = plt.subplots(figsize=(3,3))

with col2:
    ax5.bar(
        heart_disease_df_avg.index
        ,heart_disease_df_avg['smoking']*100
        ,align='center'
        ,alpha=0.5
        ,color='indianred'
    )
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('Relation between agecategory and smoking')
    st.write(fig5)

fig5, ax5 = plt.subplots(figsize=(3,3))

with col1:
    ax5.bar(
        heart_disease_df_avg.index
        ,heart_disease_df_avg['alcoholdrinking']*100
        ,align='center'
        ,alpha=0.5
        ,color='green'
    )
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('Relation between agecategory and drinking')
    st.write(fig5)

fig5, ax5 = plt.subplots(figsize=(3,3))

with col2:
    ax5.bar(
        heart_disease_df_avg.index
        ,heart_disease_df_avg['skincancer']*100
        ,align='center'
        ,alpha=0.5
        ,color='purple'
    )
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('Relation between agecategory and skincancer')
    st.write(fig5)

fig5, ax5 = plt.subplots(figsize=(3,3))

with col1:
    ax5.bar(
        heart_disease_df_avg.index
        ,heart_disease_df_avg['heartdisease']*100
        ,align='center'
        ,alpha=0.5
        ,color='green'
    )
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('Relation between agecategory and heartdisease')
    st.write(fig5)
