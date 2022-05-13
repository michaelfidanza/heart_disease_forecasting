# importing all the libraries
import io
from itertools import groupby
from nbformat import write
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn import preprocessing
import streamlit as st
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
st.text('')
st.text('')
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
st.text('')
st.text('')
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show how many people in the dataset have heart disease vs how many don't have heart disease
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# divide the web page in 2 columns
col1, col2 = st.columns(2)

with col1:
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

with col2:
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.write("By looking at the proportion between people who had/have heart disease and those who don't, \
    we notice that the dataset is unbalanced, and this will be a problem later when applying ML algorithms")

# empty spaces
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show how BMI is related to other variables
# ----------------------------------------------------------------------------------------------------------------------------------------------------

col1 = st.columns(1)

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
st.text('')
st.text('')
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show percentages of diabetic people for each age category ---- OLD ----
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# fig3, ax3 = plt.subplots(1, 2, figsize=(20,10))

# # encode values manually to decide how to encode them
# heart_disease_df_sorted['diabetic'] = heart_disease_df_sorted['diabetic'].map({'Yes' : 1, 'No' : 0, 'Yes (during pregnancy)' : 1, 'No, borderline diabetes' : 0})

# # group by age category to find the total number of people and of people with diabtes for each age category
# diabetic_statistics = heart_disease_df_sorted.groupby('agecategory')['diabetic'].aggregate([sum, 'count'])

# # calculate the percentage using sum and count
# diabetic_statistics['percentage'] = diabetic_statistics['sum'] / diabetic_statistics['count'] * 100

# # graph for age distribution
# ax3[0].bar(diabetic_statistics.index, diabetic_statistics['count'], align='center', alpha=0.5)
# # graph about percentage of diabetic people for age category
# ax3[1].bar(diabetic_statistics.index, diabetic_statistics['percentage'], align='center', alpha=0.5)

# ax3[0].set_title('Distribution of age category', fontsize=15)
# ax3[1].set_title('% of diabetic people for each age category', fontsize=15)

# ax3[0].tick_params(axis='x', rotation=45)
# ax3[1].tick_params(axis='x', rotation=45)

# st.pyplot(fig3)
# st.write("We can notice that as age grows, also the probability to be diabetic grows")

# # empty spaces
# st.text('')
# st.text('')
# st.text('')
# st.text('')
# st.text('')
# st.text('')


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# show how selected feature is related to other variables
# ----------------------------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns(2)

# use the sorted df and map the variables with integer values
heart_disease_df_sorted['stroke'] = heart_disease_df_sorted['stroke'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['smoking'] = heart_disease_df_sorted['smoking'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['alcoholdrinking'] = heart_disease_df_sorted['alcoholdrinking'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['skincancer'] = heart_disease_df_sorted['skincancer'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['heartdisease'] = heart_disease_df_sorted['heartdisease'].map({"Yes" : 1, "No" : 0})
heart_disease_df_sorted['diabetic'] = heart_disease_df_sorted['diabetic'].map({'Yes' : 1, 'No' : 0, 'Yes (during pregnancy)' : 1, 'No, borderline diabetes' : 0})

# let the user decide based on which feature he wants to aggregate the dataset
with col1:
    aggregation_feature = st.selectbox('Select the features used to group the dataset', ['agecategory', 'sex', 'race'])

# let the used decide the type of graph to visualize: 1: total number over positive number, 2: % graphs
with col2:
    type_of_visualization = st.selectbox('Select the type of visualization', ['1: Comparison between total and positive', '2: Percentage of positive people'])
if '1:' in type_of_visualization:
    type_of_visualization = 1
else:
    type_of_visualization = 2

# calculate number of "positive" to the features and total number of people group by age category
heart_disease_df_aggregate = heart_disease_df_sorted[[aggregation_feature, 'stroke', 'smoking', 'alcoholdrinking', 'skincancer', 'heartdisease', 'diabetic']]\
    .groupby(aggregation_feature).aggregate([sum, 'count', 'mean'])

# plot the distribution of people grouped by the selected feature
st.columns(1)
fig5, ax5 = plt.subplots(figsize=(3,3))
ax5.bar(
    heart_disease_df_aggregate.index
    ,heart_disease_df_aggregate['diabetic', 'count']
    ,align='center'
    ,color='bisque'
    ,alpha=1
)
ax5.tick_params(axis='x', rotation=90)
ax5.set_title('Distribution of people grouped by the feature: ' + aggregation_feature)
st.write(fig5)

col1, col2 = st.columns(2)
# show all the plots for the various features grouped by the feature selected by the user
fig5, ax5 = plt.subplots(figsize=(3,3))
with col1:
    
    # type 1: total number in the category over positive number
    if type_of_visualization == 1:
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['stroke', 'count']
            ,align='center'
            ,color='bisque'
        )
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['stroke', 'sum']
            ,align='center'
            ,color='indianred'
        )
    
    # type 2: % graphs
    else:
        ax5.bar(
        heart_disease_df_aggregate.index
        ,heart_disease_df_aggregate['stroke', 'mean']*100
        ,align='center'
        ,color='indianred'
        ,alpha=0.5
        )
        ax5.set_ylabel('%')
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('People who had stroke')
    st.write(fig5)

fig5, ax5 = plt.subplots(figsize=(3,3))
with col2:
    if type_of_visualization == 1:
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['heartdisease','count']
            ,align='center'
            ,color='bisque'
        )
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['heartdisease','sum']
            ,align='center'
            ,color='indianred'
        )
    else:
        ax5.bar(
        heart_disease_df_aggregate.index
        ,heart_disease_df_aggregate['heartdisease', 'mean']*100
        ,align='center'
        ,color='indianred'
        ,alpha=0.5
        )
        ax5.set_ylabel('%')
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('People whith heart disease')
    st.write(fig5)

fig5, ax5 = plt.subplots(figsize=(3,3))
with col1:
    if type_of_visualization == 1:
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['skincancer','count']
            ,align='center'
            ,color='bisque'
        )
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['skincancer','sum']
            ,align='center'
            ,color='indianred'
        )
    else:
        ax5.bar(
        heart_disease_df_aggregate.index
        ,heart_disease_df_aggregate['skincancer', 'mean']*100
        ,align='center'
        ,color='indianred'
        ,alpha=0.5
        )
        ax5.set_ylabel('%')
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('People who have skin cancer')
    st.write(fig5)

fig5, ax5 = plt.subplots(figsize=(3,3))
with col2:
    if type_of_visualization == 1:
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['alcoholdrinking','count']
            ,align='center'
            ,color='bisque'
        )
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['alcoholdrinking','sum']
            ,align='center'
            ,color='indianred'
        )
    else:
        ax5.bar(
        heart_disease_df_aggregate.index
        ,heart_disease_df_aggregate['alcoholdrinking', 'mean']*100
        ,align='center'
        ,color='indianred'
        ,alpha=0.5
        )
        ax5.set_ylabel('%')
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('People who drinks')
    st.write(fig5)
    
fig5, ax5 = plt.subplots(figsize=(3,3))
with col1:
    if type_of_visualization == 1:
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['smoking','count']
            ,align='center'
            ,color='bisque'
        )
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['smoking','sum']
            ,align='center'
            ,color='indianred'
        )
    else:
        ax5.bar(
        heart_disease_df_aggregate.index
        ,heart_disease_df_aggregate['smoking', 'mean']*100
        ,align='center'
        ,color='indianred'
        ,alpha=0.5
        )
        ax5.set_ylabel('%')
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('People who smokes')
    st.write(fig5)

fig5, ax5 = plt.subplots(figsize=(3,3))
with col2:
    if type_of_visualization == 1:
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['diabetic','count']
            ,align='center'
            ,color='bisque'
        )
        ax5.bar(
            heart_disease_df_aggregate.index
            ,heart_disease_df_aggregate['diabetic','sum']
            ,align='center'
            ,color='indianred'
        )
    else:
        ax5.bar(
        heart_disease_df_aggregate.index
        ,heart_disease_df_aggregate['diabetic', 'mean']*100
        ,align='center'
        ,color='indianred'
        ,alpha=0.5
        )
        ax5.set_ylabel('%')
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('People who are diabetics')
    st.write(fig5)

# setting back the web page to 1 column
st.columns(1)

# general plot to check how % of positive people vary with selected feature (only for age category)
if aggregation_feature == 'agecategory':
    fig5, ax5 = plt.subplots(figsize=(8,6))
    ax5.plot( heart_disease_df_aggregate.index, (heart_disease_df_aggregate['stroke','sum']/heart_disease_df_aggregate['stroke','count']*100), label='stroke')
    ax5.plot( heart_disease_df_aggregate.index, (heart_disease_df_aggregate['heartdisease','sum']/heart_disease_df_aggregate['heartdisease','count']*100), label='heartdisease')
    ax5.plot( heart_disease_df_aggregate.index, (heart_disease_df_aggregate['skincancer','sum']/heart_disease_df_aggregate['skincancer','count']*100), label='skincancer')
    ax5.plot( heart_disease_df_aggregate.index, (heart_disease_df_aggregate['alcoholdrinking','sum']/heart_disease_df_aggregate['alcoholdrinking','count']*100), label='alcoholdrinking')
    ax5.plot( heart_disease_df_aggregate.index, (heart_disease_df_aggregate['smoking','sum']/heart_disease_df_aggregate['smoking','count']*100), label='smoking')
    ax5.plot( heart_disease_df_aggregate.index, (heart_disease_df_aggregate['diabetic','sum']/heart_disease_df_aggregate['diabetic','count']*100), label='diabetic')
    ax5.tick_params(axis='x', rotation=90)
    ax5.set_title('Features varying with age')
    ax5.set_ylabel('%')
    ax5.legend(loc='upper left', prop={"size":8})
    st.pyplot(fig5)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ADD AVERAGE BMI, AVERAGE HEALTH INDICATORS??
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# SCATTERPLOT FOR BMI GROUPED BY FEATURE?
# ----------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# MACHINE LEARNING
# ----------------------------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns(2)

# take inputs from the user to decide which model and how to use it
with col1:
    type_of_model = st.selectbox('Select the ML model to use', ['1: Gaussian Naive Bayes', '2: Random Forest', '3: Decision Tree', '4: Linear Regression'])
    kfold_method = st.checkbox('Use kfold method')

with col2:
    type_of_sampling = st.selectbox('Select how to handle unbalanced dataset', ['1: Undersampling', '2: Oversampling', '3: Undersampling and oversampling', '4: No action'])
    pca_method = st.checkbox('Use PCA')

st.columns(1)
feature_to_predict = st.selectbox('Select the feature to predict', ['stroke', 'heartdisease'])
test_size_input = st.slider('Select the test size for the model', min_value=0.1, max_value=0.9, step=0.05)

y = heart_disease_df_encoded[feature_to_predict]
x = heart_disease_df_encoded.drop(feature_to_predict, axis=1)

# use the selected sampler to handle unbalanced dataset
if '1:' in type_of_sampling:
    sampler = RandomUnderSampler()
    x_sample, y_sample = sampler.fit_resample(x, y)
    st.write('help')
elif '2:' in type_of_sampling:
    sampler = RandomOverSampler()
    x_sample, y_sample = sampler.fit_resample(x, y)
    st.write('help')
elif '3:' in type_of_sampling:
    st.write('help')

else:
    st.write('help')

# assign the selected model
if '1:' in type_of_model:
    model = GaussianNB()
elif '2:' in type_of_model:
    model = RandomForestClassifier()
elif '3:' in type_of_model:
    model = DecisionTreeClassifier()
else:
    st.write('help')
    #model = LinearRegression()

if kfold_method:
    # declare kfold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    i = 0
    accuracies = []

    # ten splits of indexes of my data to use for training/test and find average accuracy of the model
    for train_index, test_index in kf.split(x_sample):
        x_train, x_test = x_sample.iloc[train_index], x_sample.iloc[test_index]
        y_train, y_test = y_sample.iloc[train_index], y_sample.iloc[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        accuracies.append(accuracy)
        i += 1
        st.write('Accuracy training number ' + str(i) + ': ' + str(accuracy))

    st.write('Mean accuracy of the model:', np.array(accuracies).mean())
else:
    # split the dataset into train/test data using a library function
    x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=test_size_input, random_state=42, shuffle=True)

    # train the model
    model.fit(x_train, y_train)

    # predict the values
    y_pred = model.predict(x_test)

    # show the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy of the model: ' + str(accuracy))