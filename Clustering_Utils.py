#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering

# Internal Measures:
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

# External Measures:
from sklearn.metrics import adjusted_rand_score, v_measure_score, adjusted_mutual_info_score

#KMeans:
def internal_measure_KMeans(data,k):
    '''
    Used for evaluating KMeans performance.
    data: Dataframe
    k: No of clusters in integers
    '''
    kmeans = KMeans(n_clusters=k,n_init='auto',random_state=42)
    clusters = kmeans.fit_predict(data)
    # Calculate the Davies-Bouldin index:
    db_score = davies_bouldin_score(data,clusters)

    # Calculate the Silhouette coefficient:
    s_score = silhouette_score(data,clusters)

    # Calculate Calinski Harabasz Score:
    c_score = calinski_harabasz_score(data,clusters)
   
    internal_measure_df = pd.DataFrame({'Davies-Bouldin Index':f'{db_score:.3f}','Silhouette Coeff':f'{s_score:.3f}',                                       'Calinski Harabasz Score':f'{c_score:.3f}'},index=['Result'])
    return internal_measure_df

def external_measure_KMeans(X_data,Y_target,k):
    kmeans = KMeans(n_clusters=k,n_init='auto',random_state=42)
    clusters = kmeans.fit_predict(X_data)
    
    # Calculate the v-measure score:
    v_measure = v_measure_score(Y_target, clusters)

    # Calculate the Rand index score:
    rand_index = adjusted_rand_score(Y_target, clusters)

    # Calculate the mutual information score:
    mi_score = adjusted_mutual_info_score(Y_target, clusters)

    external_measure_df = pd.DataFrame({'V-measure Score':f'{v_measure:.3f}','Rand Index Score':f'{rand_index:.3f}',                                       'Mutual Information Score':f'{mi_score:.3f}'},index=['Result'])
    return external_measure_df


#DBSCAN:
def internal_measure_DBSCAN(data,eps,min_samples):
    '''
    Used for evaluating DBSCAN performance.
    data: Dataframe
    eps: Maximum distance between data points.
    min_samples: least number of points in a cluster.
    '''
    db_model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db_model.fit_predict(data)

    # Calculate the Davies-Bouldin index:
    db_score = davies_bouldin_score(data,clusters)

    # Calculate the Silhouette coefficient:
    s_score = silhouette_score(data,clusters)

    # Calculate Calinski Harabasz Score:
    c_score = calinski_harabasz_score(data,clusters)
   
    internal_measure_df = pd.DataFrame({'Davies-Bouldin Index':f'{db_score:.3f}','Silhouette Coeff':f'{s_score:.3f}',                                       'Calinski Harabasz Score':f'{c_score:.3f}'},index=['Result'])
    return internal_measure_df

def external_measure_DBSCAN(X_data,eps,min_samples,Y_target):
    db_model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db_model.fit_predict(X_data)
    
    # Calculate the v-measure score:
    v_measure = v_measure_score(Y_target, clusters)

    # Calculate the Rand index score:
    rand_index = adjusted_rand_score(Y_target, clusters)

    # Calculate the mutual information score:
    mi_score = adjusted_mutual_info_score(Y_target, clusters)

    external_measure_df = pd.DataFrame({'V-measure Score':f'{v_measure:.3f}','Rand Index Score':f'{rand_index:.3f}',                                       'Mutual Information Score':f'{mi_score:.3f}'},index=['Result'])
    return external_measure_df


#Hierarchical Clustering:
def internal_measure_Hierachy(data,n_clusters,linkage):
    h_model= AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = h_model.fit_predict(data)

    # Calculate the Davies-Bouldin index:
    db_score = davies_bouldin_score(data,clusters)

    # Calculate the Silhouette coefficient:
    s_score = silhouette_score(data,clusters)

    # Calculate Calinski Harabasz Score:
    c_score = calinski_harabasz_score(data,clusters)
   
    internal_measure_df = pd.DataFrame({'Davies-Bouldin Index':f'{db_score:.3f}','Silhouette Coeff':f'{s_score:.3f}',                                       'Calinski Harabasz Score':f'{c_score:.3f}'},index=['Result'])
    return internal_measure_df

def external_measure_Hierachy(X_data,n_clusters,linkage,Y_target):
    h_model= AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = h_model.fit_predict(X_data)
    
    # Calculate the v-measure score:
    v_measure = v_measure_score(Y_target, clusters)

    # Calculate the Rand index score:
    rand_index = adjusted_rand_score(Y_target, clusters)

    # Calculate the mutual information score:
    mi_score = adjusted_mutual_info_score(Y_target, clusters)

    external_measure_df = pd.DataFrame({'V-measure Score':f'{v_measure:.3f}','Rand Index Score':f'{rand_index:.3f}',                                       'Mutual Information Score':f'{mi_score:.3f}'},index=['Result'])
    return external_measure_df


# In[ ]:




