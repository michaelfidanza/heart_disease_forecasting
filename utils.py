from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import  confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# function that apply the selected type balancing on the input dataset (imblearn library) and returns the new x and y
def sampler(x, y, type_of_sampling):
    # use the selected sampler to handle imbalanced dataset
    if '2:' in type_of_sampling:
        sampler = RandomUnderSampler(random_state=42)
        x_sample, y_sample = sampler.fit_resample(x, y)
    elif '3:' in type_of_sampling:
        sampler = RandomOverSampler(random_state=42)
        x_sample, y_sample = sampler.fit_resample(x, y)
    elif '4:' in type_of_sampling:
        sampler = RandomOverSampler(sampling_strategy=0.5,random_state=42)
        x_sample, y_sample = sampler.fit_resample(x, y)
        sampler = RandomUnderSampler(random_state=42)
        x_sample, y_sample = sampler.fit_resample(x_sample, y_sample)
    elif '5:' in type_of_sampling:
        sampler = SMOTE(random_state=42)
        x_sample, y_sample = sampler.fit_resample(x, y)
    else:
        x_sample, y_sample = x, y
    return x_sample, y_sample

# function that returns the pie chart of the positive/negative percentages of a single feature
def pie_pos_neg_chart(feature, chart_labels, chart_title):
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.pie(
        [len(feature) - sum(feature), sum(feature)]
        ,labels=chart_labels
        ,explode=(0, 0.2)
        ,startangle=45
        ,autopct='%1.1f%%'
        ,shadow=True
        ,colors=['bisque', 'indianred']
        )
    plt.title(chart_title, fontsize=8)
    return fig

# function that taken the set of test values and predicted values, returns the heatmap plot of the confusion matrix with accuracy score
def cf_matrix_plot(y_test, y_pred):
    # calculate the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    labels = ['True Neg\n\n' + str(round(cf_matrix[0,0]/len(y_test)*100, 2)) + ' %\n'+ str(cf_matrix[0,0]),'False Pos\n\n' + str(round(cf_matrix[0,1]/len(y_test)*100, 2)) + ' %\n'+ str(cf_matrix[0,1]),\
        'False Neg\n\n' + str(round(cf_matrix[1,0]/len(y_test)*100, 2)) + ' %\n'+ str(cf_matrix[1,0]),'True Pos\n\n' + str(round(cf_matrix[1,1]/len(y_test)*100, 2)) + ' %\n'+ str(cf_matrix[1,1])]
    labels = np.asarray(labels).reshape(2,2)
    
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds')

    # calculate the accuracy of the model
    accuracy = str(accuracy_score(y_test, y_pred)* 100)[:5]
    ax.set_title('Confusion matrix\n\nAccuracy score: ' + accuracy + ' %\n');
    
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Real Values ');

    # axes labels
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    # return the visualization of the Confusion Matrix.
    return fig