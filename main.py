# importing all the libraries
import io
from utils import sampler, cf_matrix_plot, pie_pos_neg_chart
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.utils import shuffle
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

# pie chart for positive/negative to heart disease
with col1:
    st.pyplot(pie_pos_neg_chart(heart_disease_df_encoded['heartdisease'], \
    ("Negative (" + str(len(heart_disease_df_encoded['heartdisease']) - sum(heart_disease_df_encoded['heartdisease'])) + ')', "Positive (" + str(sum(heart_disease_df_encoded['heartdisease'])) + ')'), \
        "People with heart disease?"))

# pie chart for positive/negative to stroke
with col2:
    st.pyplot(pie_pos_neg_chart(heart_disease_df_encoded['stroke'],\
        ("Negative (" + str(len(heart_disease_df_encoded['stroke']) - sum(heart_disease_df_encoded['stroke'])) + ')', \
            "Positive (" + str(sum(heart_disease_df_encoded['stroke'])) + ')'),\
               "People with stroke?" ))

st.columns(1)

st.write("By looking at the proportion between people who had/have heart disease/stroke and those who don't, \
we notice that the dataset is imbalanced, and this could be a problem later when applying ML algorithms")

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
ax4[0,0].set_title('Are BMI and sex related?\n', fontsize=15)
ax4[0,1].set_title('Are BMI and age category related?\n', fontsize=15)
ax4[1,0].set_title('Are BMI and race related\n?', fontsize=15)
ax4[1,1].set_title('Are BMI and diabetic state related?\n', fontsize=15)
ax4[2,0].set_title('Are BMI and difficulty in walking related?\n', fontsize=15)
ax4[2,1].set_title('Are BMI and physical activity related?\n', fontsize=15)

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


# empty spaces
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# MACHINE LEARNING
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# title
st.markdown("<h2 style='text-align: center; color: black;'>Machine Learning</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# take inputs from the user to decide which model and how to use it
with col1:
    type_of_model = st.selectbox('Select the ML model to use', ['1: Gaussian Naive Bayes', '2: Random Forest', '3: Decision Tree', '4: Linear Regression'])
with col2:
    type_of_sampling = st.selectbox('Select how to handle imbalanced dataset', ['1: No action', '2: Undersampling', '3: Oversampling', '4: Undersampling and oversampling', '5: SMOTE'])

st.columns(1)

# make the user decide what he wants to predict
feature_to_predict = st.selectbox('Select the feature to predict', ['heartdisease','stroke'])

# prepare feature to predict and predictors based on user input
y = heart_disease_df_encoded[feature_to_predict]
x = heart_disease_df_encoded.drop(feature_to_predict, axis=1)

# other choices for the user
col1, col2, col3 = st.columns(3)
with col1:
    shuffle_dataset = st.checkbox('Shuffle dataset', value=True)
with col2:
    pca_method = st.checkbox('Use PCA')
with col3:
    kfold_method = st.checkbox('Use kfold method')
    if kfold_method:
        kfold_splits = st.selectbox('Choose the number of splits for kfold cross validation', ['5', '6', '7', '8', '9', '10'])

st.columns(1)

# size of the test set
test_size_input = st.slider('Select the test size for the model', min_value=0.1, max_value=0.9, step=0.05, value=0.2)

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

col1, col2 = st.columns(2)

if kfold_method:
    # declare kfold
    if shuffle_dataset:
        kf = KFold(n_splits=int(kfold_splits), shuffle=shuffle_dataset, random_state=42)
    else:
        kf = KFold(n_splits=int(kfold_splits), shuffle=shuffle_dataset)
    i = 0
    accuracies = []

    # ten splits of indexes of my data to use for training/test and find average accuracy of the model
    with st.spinner('Training and predicting...'):
        for train_index, test_index in kf.split(x):

            # find train and test datasets
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # use the selected sampler on training set to handle imbalanced dataset
            x_sample, y_sample = sampler(x_train, y_train, type_of_sampling)

            model.fit(x_sample, y_sample)

            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_pred, y_test)
            accuracies.append(accuracy)
            
            if i%2 == 0:
                with col1:
                    # display the confusion matrix
                    st.pyplot(cf_matrix_plot(y_test, y_pred))
                    st.write('')
                    st.write('')
            else:
                with col2:
                    # display the confusion matrix
                    st.pyplot(cf_matrix_plot(y_test, y_pred))
                    st.write('')
                    st.write('')
            i += 1

    st.write('Mean accuracy of the model:', np.array(accuracies).mean())
else:
    # split the dataset into train/test data using a library function
    if shuffle_dataset:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_input, shuffle=shuffle_dataset, random_state=42)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_input, shuffle=shuffle_dataset)
    
    # use the selected sampler on training set to handle imbalanced dataset
    x_sample, y_sample = sampler(x_train, y_train, type_of_sampling)
               
    with col1:
        # plot proportion of positive/negative examples in training set
        st.pyplot(pie_pos_neg_chart(y_sample, ("Healthy (" + str(len(y_sample) - sum(y_sample)) + ')', feature_to_predict + " (" + str(sum(y_sample)) + ')'),
         "How many people in the training set have/had " +  feature_to_predict + "?"))
    
    with col2:
        # plot proportion of positive/negative examples in test set
        st.pyplot(pie_pos_neg_chart(y_test, ("Healthy (" + str(len(y_test) - sum(y_test)) + ')', feature_to_predict + " (" + str(sum(y_test)) + ')'),\
             "How many people in the test set have/had " +  feature_to_predict + "?"))
    
    with st.spinner('Training and predicting...'):
        # train the model
        model.fit(x_sample, y_sample)

        # predict the values
        y_pred = model.predict(x_test)


    # display the confusion matrix
    st.pyplot(cf_matrix_plot(y_test, y_pred))
