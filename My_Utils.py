#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import (precision_score,
                            accuracy_score,
                            recall_score,
                            f1_score)

# Adding percent or count to plots:
def plot_percent_weight(plot_name,data,feature,show_value=None):
    '''
    plot_name: The name of the variable assigned to the plot.
    data: The dataframe to compute the weight against.
    feature: The feature being plotted against.
    show_value: Determines the weight to show whether only percent or percent and weight (True or False).
    '''
    #Calculating the weights:
    for attribute in plot_name.patches:
        if show_value==True:
            value = '{:.1f}% ({})'.format(100 * attribute.get_height() / len(data[feature]), attribute.get_height())
        elif show_value==False:
            value = '{:.1f}%'.format(100 * attribute.get_height() / len(data[feature]))
            
        #Defining annotation coordinatites:    
        x_cord = attribute.get_x() + attribute.get_width() / 6
        y_cord = attribute.get_height()
        plot_name.annotate(value,(x_cord,y_cord),textcoords='offset points',xytext=(1,2))
    return

# countplot function:
def countplot_func(data,main_data,plot_size,xlabel,plot_title,plot_variable=None,add_count=False):
    '''
    data: The data containing the feature being plotted.
    main_data: The dataframe to compute the feature weight against.
    plot_size: size of the plot.
    xlabel: Name of x-axis.
    plot_title: Title of plot.
    plot_variable: Feature from main_data to be weighted against.
    add_count: Determines whether to add percent or both percent and weight (True or False).
    '''
    plt.figure(figsize=plot_size)
    plot_name = sns.countplot(data=data, x=xlabel)  #Instance of the plot created
    plot_percent_weight(plot_name,main_data,plot_variable,add_count)
    plt.xlabel(xlabel)
    plt.title(plot_title)
    plt.grid(linestyle='--',linewidth=0.4)
    plt.show()
    return

# barplot function:
def barplot_func(data,main_data,plot_size,x,y,xlabel,plot_title,plot_variable,add_count):
    

    plt.figure(figsize=plot_size)
    plot_name = sns.barplot(data=data, x=x, y=y)
    plot_percent_weight(plot_name,main_data,plot_variable,show_value=add_count)
    plt.xlabel(xlabel)
    plt.title(plot_title)
    plt.grid(linestyle='--',linewidth=0.5)
    plt.show()
    return

# catplot function:
def catplot_func(data,x,y,kind,height,aspect,plot_title,xlabel):
    '''
    data: Dataframe being considered.
    x: feature on x-axis.
    y: feature on y_axis.
    kind: Specifies the type of plot (e.g box).
    height,aspect: Specifies the size of plot.
    plot_title: Title of plot'.
    xlabel: Title on x-axis.
    '''
    sns.catplot(data=data, x=x, y=y, kind=kind,height=height, aspect=aspect)
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.grid(linestyle='--',linewidth=0.4)
    plt.show()
    return

def treat_outliers_columns(data,feature_list):
    for feature in feature_list:
        Q_1= data[feature].quantile(0.25)
        Q_3 = data[feature].quantile(0.75)
        Iq_range = Q_3 - Q_1
        lower = Q_1 - (1.5 * Iq_range)
        upper = Q_3 + (1.5 * Iq_range)
        new_feature = np.clip(data[feature], lower, upper)
        data[feature] = new_feature
    return data


# Classification and Confusion Matrix Display plot function:
def classification_confusion_matrix(fig,yTest,yTrain,yPrediction_train, yPrediction_test,target_labels):
    print('Classification Report for Training:')
    print(classification_report(yTrain, yPrediction_train))
    print('='*80)
    print('Classification Report for Testing:')
    print(classification_report(yTest, yPrediction_test))
    print('='*80)
    fig, ax = plt.subplots(1,2,figsize=fig)
    ConfusionMatrixDisplay.from_predictions(yTrain,yPrediction_train, display_labels=target_labels,ax=ax[0])
    ConfusionMatrixDisplay.from_predictions(yTest,yPrediction_test, display_labels=target_labels,ax=ax[1])
    ax[0].title.set_text('Training')
    ax[1].title.set_text('Testing')
    plt.show()
    return

# Model building function:
def building_model(model,Xtrain_data,X_data,Ytrain_data):
    model.fit(Xtrain_data,Ytrain_data)
    prediction = model.predict(X_data)
    return prediction

# Performance Evaluation Function:
def performance_evaluation(prediction_test,prediction_train,Ytest_data,Ytrain_data):
    
    #Training:
    prec_train = precision_score(Ytrain_data,prediction_train)
    accu_train = accuracy_score(Ytrain_data,prediction_train)
    rec_train = recall_score(Ytrain_data,prediction_train)
    f1_train = f1_score(Ytrain_data,prediction_train)
    
    #Testing:    
    prec_test = precision_score(Ytest_data,prediction_test)
    accu_test = accuracy_score(Ytest_data,prediction_test)
    rec_test = recall_score(Ytest_data,prediction_test)
    f1_test = f1_score(Ytest_data,prediction_test)
    
    #Performance Dataframe:
    perform_eval = pd.DataFrame({'Precision':[f'{prec_train:.2f}',f'{prec_test:.2f}'],                                 'Recall':[f'{rec_train:.2f}',f'{rec_test:.2f}'],                                 'F1 Score':[f'{f1_train:.2f}',f'{f1_test:.2f}'],                               'Accuracy':[f'{accu_train:.2f}',f'{accu_test:.2f}']},                                index=['Training','Testing'])
    return perform_eval


# In[ ]:




