# importing all the libraries
from email.policy import default
import io
from time import sleep
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
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import streamlit as st
from scipy import stats

# initialize dataset
heart_disease_df = pd.read_csv(r'~/Desktop/Documents/repos/heart_disease_forecasting/heart_2020_cleaned.csv')

# all columns names to lowercase
heart_disease_df.columns = heart_disease_df.columns.map(lambda x : x.lower())

# defining the urls for dataset source and github profile
url_dataset = 'https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease'
url_github = 'https://github.com/michaelfidanza'

# titles
st.markdown("<h1 style='text-align: center; color: brown;'>Heart Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: grey;'>Programming Project 2021/2022</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Michael Fidanza - VR472909</h3>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'><a href=" + url_github + ">GitHub profile</a> - <a href=" + url_dataset + ">Dataset source</a></h6>", unsafe_allow_html=True)

# empty spaces
st.text('')
st.text('')
st.text('')
st.text('')   
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Data cleaning: to drop genhealth and check on outliers for different features
# ----------------------------------------------------------------------------------------------------------------------------------------------------

heart_disease_df_before_encoding = heart_disease_df.copy()

# remove outliers
# Calculate the z-scores
with st.spinner('calculating zscores'):
    z_scores = stats.zscore(heart_disease_df_before_encoding[['bmi', 'sleeptime','physicalhealth','mentalhealth']])

fig, ax = plt.subplots(figsize=(3,3))

#This is what I drop using zscore
# st.write(heart_disease_df_before_encoding[['sleeptime', 'bmi','physicalhealth', 'mentalhealth']][(z_scores['bmi'] < -3) | (z_scores['bmi'] > 3) | (z_scores['sleeptime'] < -3) |  (z_scores['sleeptime'] > 3) | \
#                                                                        (z_scores['physicalhealth'] < -3) | (z_scores['physicalhealth'] > 3) |  \
#                                                                            (z_scores['mentalhealth'] < -3) |  (z_scores['mentalhealth'] > 3) ])


heart_disease_df_before_encoding = heart_disease_df_before_encoding[(z_scores['bmi'] > -3) &  (z_scores['bmi'] < 3) & \
                                                                     (z_scores['sleeptime'] > -3) &  (z_scores['sleeptime'] < 3) & \
                                                                       (z_scores['physicalhealth'] > -3) &  (z_scores['physicalhealth'] < 3) &  \
                                                                           (z_scores['mentalhealth'] > -3) &  (z_scores['mentalhealth'] < 3) ]

# drop gen health (retrievable from the other factors)
heart_disease_df_before_encoding = heart_disease_df_before_encoding.drop('genhealth', axis=1)

#reduce diabetes category
heart_disease_df_before_encoding['diabetic'] = heart_disease_df_before_encoding['diabetic'].map({'Yes': 'Yes', 'No' : 'No', 'Yes (during pregnancy)' : 'Yes', 'No, borderline diabetes' : 'No'})

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# convert non numerical values to numbers using an encoder from sklearn
# initialize the encoder
le = preprocessing.LabelEncoder()

# make a copy of the dataframe
heart_disease_df_encoded = heart_disease_df_before_encoding.copy()

#group BMI
bins = [0, 18.4, 24.9, 29.9, np.inf]
names = ['<18.5 Underweight', '18.5-24.9 Normal weight', '25-29.9 Overweight', '>=30 Obese']
heart_disease_df_encoded['bmi'] = pd.cut(heart_disease_df_encoded['bmi'], bins, labels=names)

# find the columns which need encoding
categorical_columns = heart_disease_df_encoded.dtypes[heart_disease_df_encoded.dtypes != 'float64'].index.to_list()

# and encode them
for col in categorical_columns:
    heart_disease_df_encoded[col] = le.fit_transform(heart_disease_df_encoded[col])

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Pages navigation
# ----------------------------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns(2)
with col2:
    page = st.selectbox('Page:', ['Data Exploration','Machine Learning'])

if page == 'Data Exploration':
    st.columns(1)

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

    # show the "cleaned" dataset
    st.markdown("<h6 style=color: black;'>Modified data</h6>", unsafe_allow_html=True)
    st.write(heart_disease_df_encoded)

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
    # show how many people in the dataset have heart disease / stroke vs how many don't have heart disease / stroke
    # ----------------------------------------------------------------------------------------------------------------------------------------------------


    st.markdown("<h6 style=color: black;'>Proportion of people interested by predictable features</h6>", unsafe_allow_html=True)

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

    st.write("By looking at the proportion between people who had/have heart disease/stroke and those who don't, \
    we notice that the dataset is imbalanced, and this could be a problem later when applying ML algorithms")

    # empty spaces
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # show distribution of features
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    st.markdown("<h6 style=color: black;'>Distribution of features</h6>", unsafe_allow_html=True)
    st.text('')
    st.text('')

    # order by age category to obtain better graph
    heart_disease_df_sorted = heart_disease_df_before_encoding.sort_values(by='agecategory')

    feature_distribution = st.selectbox('Select the feature of which you want to see the distribution', \
        ['smoking', 'alcoholdrinking', 'skincancer', 'diabetic', 'diffwalking', 'physicalactivity', 'asthma', 'kidneydisease', 'physicalhealth', 'mentalhealth', 'sleeptime'])


    # bar plot for ['sleeptime', 'mentalhealth', 'physicalhealth']
    if feature_distribution in ['sleeptime', 'mentalhealth', 'physicalhealth']:
        
        # bar plot
        fig, ax = plt.subplots(figsize=(3, 3))
        
        ax.bar(
            heart_disease_df_sorted[feature_distribution].value_counts().sort_index().index.map(lambda x : int(x))
            ,heart_disease_df_sorted[feature_distribution].value_counts().sort_index().values
            ,align='center'
            ,color='bisque'
            ,alpha=1
        )
        ax.set_xticks(heart_disease_df_sorted[feature_distribution].value_counts().sort_index().index.map(lambda x : int(x)))
        ax.set_xticklabels(heart_disease_df_sorted[feature_distribution].value_counts().sort_index().index.map(lambda x : int(x)), rotation=90, size=8)

        ax.set_ylabel('Number of people')
        ax.set_title('Distribution of ' + feature_distribution)
        st.write(fig)

    # pie chart for other features which are only "yes" - "no" features
    else:        
        # pie charts
        st.pyplot(pie_pos_neg_chart(heart_disease_df_encoded[feature_distribution], \
            ("No (" + str(len(heart_disease_df_encoded[feature_distribution]) - sum(heart_disease_df_encoded[feature_distribution])) + ')', "Yes (" + str(sum(heart_disease_df_encoded[feature_distribution])) + ')'), \
                "Distribution of " + feature_distribution))

        

    # empty spaces
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # show how BMI is related to other variables
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots(3, 2, figsize=(20, 40))

    

    sns.boxplot(
        x = 'sex'
        ,y = 'bmi'
        ,data = heart_disease_df_sorted
        ,ax=ax[0,0]
        ,palette='Reds'
    )
    sns.boxplot(
        x = 'agecategory'
        ,y = 'bmi'
        # use the sorted df to obtain better results graphically
        ,data = heart_disease_df_sorted
        ,ax=ax[0,1]
        ,palette='Reds'
    )
    sns.boxplot(
        x = 'race'
        ,y = 'bmi'
        ,data = heart_disease_df_sorted
        ,ax=ax[1,0]
        ,palette='Reds'
    )
    sns.boxplot(
        x = 'diabetic'
        ,y = 'bmi'
        ,data = heart_disease_df_sorted
        ,ax=ax[1,1]
        ,palette='Reds'
    )
    sns.boxplot(
        x = 'diffwalking'
        ,y = 'bmi'
        ,data = heart_disease_df_sorted
        ,ax=ax[2,0]
        ,palette='Reds'
    )
    sns.boxplot(
        x = 'physicalactivity'
        ,y = 'bmi'
        ,data = heart_disease_df_sorted
        ,ax=ax[2,1]
        ,palette='Reds'
    )


    ax[0,1].tick_params(axis='x', rotation=45)
    ax[1,0].tick_params(axis='x', rotation=20)
    ax[1,1].tick_params(axis='x', rotation=20)
    ax[0,0].set_title('Are BMI and sex related?\n', fontsize=15)
    ax[0,1].set_title('Are BMI and age category related?\n', fontsize=15)
    ax[1,0].set_title('Are BMI and race related\n?', fontsize=15)
    ax[1,1].set_title('Are BMI and diabetic state related?\n', fontsize=15)
    ax[2,0].set_title('Are BMI and difficulty in walking related?\n', fontsize=15)
    ax[2,1].set_title('Are BMI and physical activity related?\n', fontsize=15)
    fig.suptitle('How  BMI is related to other variables', fontsize=20)

    st.pyplot(fig)

    # empty spaces
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # show how selected feature (agecategory, sex or race) is related to other variables
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    col1, col2 = st.columns(2)

    # use the sorted df and map the variables with integer values
    heart_disease_df_sorted['stroke'] = heart_disease_df_sorted['stroke'].map({"Yes" : 1, "No" : 0})
    heart_disease_df_sorted['smoking'] = heart_disease_df_sorted['smoking'].map({"Yes" : 1, "No" : 0})
    heart_disease_df_sorted['alcoholdrinking'] = heart_disease_df_sorted['alcoholdrinking'].map({"Yes" : 1, "No" : 0})
    heart_disease_df_sorted['skincancer'] = heart_disease_df_sorted['skincancer'].map({"Yes" : 1, "No" : 0})
    heart_disease_df_sorted['heartdisease'] = heart_disease_df_sorted['heartdisease'].map({"Yes" : 1, "No" : 0})
    heart_disease_df_sorted['diabetic'] = heart_disease_df_sorted['diabetic'].map({'Yes' : 1, 'No' : 0, 'Yes (during pregnancy)' : 1, 'No, borderline diabetes' : 0})
    heart_disease_df_sorted['diffwalking'] = heart_disease_df_sorted['diffwalking'].map({'Yes' : 1, 'No' : 0})
    heart_disease_df_sorted['physicalactivity'] = heart_disease_df_sorted['physicalactivity'].map({'Yes' : 1, 'No' : 0})
    heart_disease_df_sorted['asthma'] = heart_disease_df_sorted['asthma'].map({'Yes' : 1, 'No' : 0})
    heart_disease_df_sorted['kidneydisease'] = heart_disease_df_sorted['kidneydisease'].map({'Yes' : 1, 'No' : 0})

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
    heart_disease_df_aggregate = heart_disease_df_sorted[[aggregation_feature, 'stroke', 'smoking', 'alcoholdrinking', 'skincancer', 'heartdisease', 'diabetic', 'diffwalking', 'physicalactivity',\
        'asthma', 'kidneydisease']].groupby(aggregation_feature).aggregate([sum, 'count', 'mean'])

    # plot the distribution of people grouped by the selected feature
    st.columns(1)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.bar(
        heart_disease_df_aggregate.index
        ,heart_disease_df_aggregate['diabetic', 'count']
        ,align='center'
        ,color='bisque'
        ,alpha=1
    )

    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Distribution of people grouped by the feature: ' + aggregation_feature)
    st.write(fig)

    col1, col2 = st.columns(2)

    # show all the plots for the various features grouped by the feature selected by the user
    i = 0
    plottable_features = ['stroke', 'smoking', 'alcoholdrinking', 'skincancer', 'heartdisease', 'diabetic', 'diffwalking', 'physicalactivity', 'asthma', 'kidneydisease']

    for feature in plottable_features:
        fig, ax = plt.subplots(figsize=(3,3))
        if i%2 == 0:
            with col1:
                # type 1: total number in the category over positive number
                if type_of_visualization == 1:
                    ax.bar(
                        heart_disease_df_aggregate.index
                        ,heart_disease_df_aggregate[feature, 'count']
                        ,align='center'
                        ,color='bisque'
                    )
                    ax.bar(
                        heart_disease_df_aggregate.index
                        ,heart_disease_df_aggregate[feature, 'sum']
                        ,align='center'
                        ,color='indianred'
                    )
                
                # type 2: % graphs
                else:
                    ax.bar(
                    heart_disease_df_aggregate.index
                    ,heart_disease_df_aggregate[feature, 'mean']*100
                    ,align='center'
                    ,color='indianred'
                    ,alpha=0.5
                    )
                    ax.set_ylabel('%')
                ax.tick_params(axis='x', rotation=90)
                ax.set_title(aggregation_feature + ' vs ' + feature)
                st.write(fig)

        else:
            with col2:
                # type 1: total number in the category over positive number
                if type_of_visualization == 1:
                    ax.bar(
                        heart_disease_df_aggregate.index
                        ,heart_disease_df_aggregate[feature, 'count']
                        ,align='center'
                        ,color='bisque'
                    )
                    ax.bar(
                        heart_disease_df_aggregate.index
                        ,heart_disease_df_aggregate[feature, 'sum']
                        ,align='center'
                        ,color='indianred'
                    )
                
                # type 2: % graphs
                else:
                    ax.bar(
                    heart_disease_df_aggregate.index
                    ,heart_disease_df_aggregate[feature, 'mean']*100
                    ,align='center'
                    ,color='indianred'
                    ,alpha=0.5
                    )
                    ax.set_ylabel('%')
                ax.tick_params(axis='x', rotation=90)
                ax.set_title(aggregation_feature + ' vs ' + feature)
                st.write(fig)
        i += 1

    # general plot to check how % of positive people vary with selected feature (only for age category)
    if aggregation_feature == 'agecategory':
        
        # user can select wchich feature to plot againt age category
        features_to_plot = st.multiselect('Select the features to plot', plottable_features, default=plottable_features)
        
        fig, ax = plt.subplots(figsize=(8,6))
        
        # plot the selected features
        for feature in features_to_plot:
            ax.plot( heart_disease_df_aggregate.index, (heart_disease_df_aggregate[feature,'sum']/heart_disease_df_aggregate[feature,'count']*100), label=feature)
        
        ax.tick_params(axis='x', rotation=90)
        ax.set_title('Features varying with age')
        ax.set_ylabel('%')
        ax.legend(loc='upper left', prop={"size":8})
        st.pyplot(fig)

    # empty spaces
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')

elif page == 'Machine Learning':
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # MACHINE LEARNING
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    # title
    st.markdown("<h2 style='text-align: center; color: black;'>Machine Learning</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # take inputs from the user to decide which model and how to use it
    with col1:
        type_of_model = st.selectbox('Select the ML model to use', ['1: Gaussian Naive Bayes', '2: Random Forest', '3: Decision Tree', '4: Logistic Regression'])
    with col2:
        type_of_sampling = st.selectbox('Select how to handle imbalanced dataset', ['1: No action', '2: Undersampling', '3: Oversampling', '4: Undersampling and oversampling', '5: SMOTE'])

    st.columns(1)

    # make the user decide what he wants to predict
    feature_to_predict = st.selectbox('Select the feature to predict', ['heartdisease','stroke'])

    # prepare feature to predict and predictors based on user input
    y = heart_disease_df_encoded[feature_to_predict]
    x = heart_disease_df_encoded.drop(feature_to_predict, axis=1)

    # make the user decide which features to use as predictors
    predictors = st.multiselect('Select the predictors', x.columns, default=list(x.columns))

    # drop the columns that were not selected by the user
    for col in x.columns:
        if col not in predictors:
            x = x.drop(col, axis=1)


    # other choices for the user
    col1, col2, col3 = st.columns(3)
    with col1:
        shuffle_dataset = st.checkbox('Shuffle dataset', value=True)
    with col2:
        pca_method = st.checkbox('Use PCA')
    with col3:
        kfold_method = st.checkbox('Use kfold method')
        if kfold_method:
            kfold_splits = st.selectbox('Choose the number of splits for kfold cross validation', [5, 6, 7, 8, 9, 10])

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
        model = LogisticRegression()

    col1, col2 = st.columns(2)

    if kfold_method:
        # declare kfold
        if shuffle_dataset:
            kf = KFold(n_splits=kfold_splits, shuffle=shuffle_dataset, random_state=42)
        else:
            kf = KFold(n_splits=kfold_splits, shuffle=shuffle_dataset)
        i = 0
        accuracies = []

        if pca_method:
            with st.spinner('Applying PCA'):
                # pca does the orthogonal projection of the feature to obtain 2 dimensions
                pca = PCA(n_components=2)
                # return the 2 features that have most explained variance
                x = pca.fit(x).transform(x)

        # ten splits of indexes of my data to use for training/test and find average accuracy of the model
        with st.spinner('Training and predicting...'):
            for train_index, test_index in kf.split(x):
                if pca_method:
                    # find train and test datasets
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                else:
                    # find train and test datasets
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # use the selected sampler on training set to handle imbalanced dataset
                with st.spinner('Sampling data...'):
                    x_sample, y_sample = sampler(x_train, y_train, type_of_sampling)

                # train
                model.fit(x_sample, y_sample)
                
                # predict
                y_pred = model.predict(x_test)
                
                #calculate accuracy
                accuracy = accuracy_score(y_pred, y_test)

                # append accuracy to array
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
        if pca_method:
            with st.spinner('Applying PCA'):
                # pca does the orthogonal projection of the feature to obtain 2 dimensions
                pca = PCA(n_components=2)
                # return the 2 features that have most explained variance
                x = pca.fit(x).transform(x)

        # split the dataset into train/test data using a library function
        if shuffle_dataset:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_input, shuffle=shuffle_dataset, random_state=42)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_input, shuffle=shuffle_dataset)

        # use the selected sampler on training set to handle imbalanced dataset
        with st.spinner('Sampling data...'):
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


        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        # SIMULATOR PART
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        st.write('')
        st.write('')
        st.write('')

        # title
        st.markdown("<h2 style='text-align: center; color: black;'>Simulator</h2>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center; color: black;'>Insert your data and find the probability to have " + feature_to_predict + "</h6>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        # ask user for data only if necessary for predicting
        heart_disease_dict_user = {}
        with col1:
            if 'sex' in predictors:
                heart_disease_dict_user['sex'] = [st.selectbox('Sex', ['Male', 'Female'])]

            if 'bmi' in predictors:
                heart_disease_dict_user['bmi'] = [st.selectbox('BMI', ['<18.5 Underweight', '18.5-24.9 Normal weight', '25-29.9 Overweight', '>=30 Obese'], index=1)]

            if 'alcoholdrinking' in predictors:    
                heart_disease_dict_user['alcoholdrinking'] = [st.selectbox('Drink?', ['No', 'Yes'])]
                
            if 'diffwalking' in predictors:    
                heart_disease_dict_user['diffwalking'] = [st.selectbox('Difficulty walking', ['No', 'Yes'])]

            if 'sleeptime' in predictors:    
                heart_disease_dict_user['sleeptime'] = [float(st.selectbox('Hours of sleep', list(range(4,17))))]

            if 'kidneydisease' in predictors:    
                heart_disease_dict_user['kidneydisease'] = [st.selectbox('Kidney disease', ['No', 'Yes'])]

        with col2:
            if 'race' in predictors:    
                heart_disease_dict_user['race'] = [st.selectbox('Race', ['White', 'Black', 'Hispanic', 'Asian', 'American Indian/Alaskan Native', 'Other'])]

            if 'smoking' in predictors:    
                heart_disease_dict_user['smoking'] = [st.selectbox('Smoked at least 100 cigarettes?', ['No', 'Yes'])]

            if 'physicalhealth' in predictors:    
                heart_disease_dict_user['physicalhealth'] = [float(st.selectbox('Phys health',list(range(0,31))))]

            if 'diabetic' in predictors:    
                heart_disease_dict_user['diabetic'] = [st.selectbox('Diabetic', ['No', 'Yes'])]

            if 'skincancer' in predictors:    
                heart_disease_dict_user['skincancer'] = [st.selectbox('Skin cancer', ['No', 'Yes'])]
            
        with col3:
            if 'agecategory' in predictors:       
                heart_disease_dict_user['agecategory'] = [st.selectbox('Age', ['18-24', '25-29', '30-34','35-39','40-44','45-49','50-54','55-69','70-74','75-79','80 or older'])]
            
            if 'stroke' in predictors:      
                heart_disease_dict_user['stroke'] = [st.selectbox('Ever had stroke?', ['No', 'Yes'])]

            if 'heartdisease' in predictors:
                heart_disease_dict_user['heartdisease'] = [st.selectbox('Heart Disease', ['No', 'Yes'])]
            
            if 'mentalhealth' in predictors:    
                heart_disease_dict_user['mentalhealth'] = [float(st.selectbox('Mental health',list(range(0,31))))]
            
            if 'physicalactivity' in predictors:    
                heart_disease_dict_user['physicalactivity'] = [st.selectbox('Phys activity', ['No', 'Yes'])]
            
            if 'asthma' in predictors:    
                heart_disease_dict_user['asthma'] = [st.selectbox('Asthma', ['No', 'Yes'])]

    
        heart_disease_df_user = pd.DataFrame.from_dict(
                heart_disease_dict_user,
                orient='columns'
        )
        
        # reencode the dataframe adding user data
        heart_disease_df_encoded = heart_disease_df_before_encoding.copy()
        
        #group BMI on dataframe
        bins = [0, 18.4, 24.9, 29.9, np.inf]
        names = ['<18.5 Underweight', '18.5-24.9 Normal weight', '25-29.9 Overweight', '>=30 Obese']
        heart_disease_df_encoded['bmi'] = pd.cut(heart_disease_df_encoded['bmi'], bins, labels=names)

        # add user data to dataframe (this is necessary to obtain the automatic encoding because the function needs to see all possible values in the dataset, not only the user ones.)
        heart_disease_df_encoded = heart_disease_df_encoded.append(heart_disease_df_user,ignore_index = True)
        
        # find the columns which need encoding
        categorical_columns = heart_disease_df_encoded.dtypes[heart_disease_df_encoded.dtypes != 'float64'].index.to_list()

        # and encode them
        for col in categorical_columns:
            heart_disease_df_encoded[col] = le.fit_transform(heart_disease_df_encoded[col])

        # separate user data from dataframe
        heart_disease_df_user = heart_disease_df_encoded.iloc[-1]
        
        # keep only features used to predict
        heart_disease_df_user = heart_disease_df_user.drop(feature_to_predict)
        
        if pca_method:
            
            # pca already fit
            heart_disease_df_user = pca.transform(heart_disease_df_user.to_numpy().reshape(1, -1))
            
            # if logistic regression or random forest, we can also say the probability of being positive
            if '2:' in type_of_model or '4:' in type_of_model:
                st.write("You'll have " + feature_to_predict + " with probability", model.predict_proba(heart_disease_df_user)[0, 1])
            else:
                st.write(("You'll have " + feature_to_predict) if model.predict(heart_disease_df_user) == 1 else "You won't have " + feature_to_predict)
        else:
            # if logistic regression or random forest, we can also say the probability of being positive
            if '2:' in type_of_model or '4:' in type_of_model:
                st.write("You'll have " + feature_to_predict + " with probability", model.predict_proba(heart_disease_df_user.to_numpy().reshape(1, -1))[0, 1])
            else:
                st.write(("You'll have " + feature_to_predict) if model.predict(heart_disease_df_user.to_numpy().reshape(1, -1)) == 1 else "You won't have " + feature_to_predict )