
import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


import matplotlib.pyplot as plt
from _datetime import date

from sklearn.decomposition import PCA

import os


def write_file(fn, str):
    with open(now_str+'/'+fn, "w") as text_file:
        print(str, file=text_file)

def createBarChart(x_axis, y_axis,filename):
    y_pos = np.arange(len(x_axis))
    plt.clf()
    plt.bar(y_pos, y_axis, align='center', alpha=0.5)

    for x,y in zip(y_pos,y_axis):

        label = y # "{:.2f}".format(y)

        plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center


    plt.xticks(y_pos, x_axis)
    plt.ylabel('Customer Count')
    plt.xlabel('Customer Group')
    plt.title('Cluster Size By Groups')
#    plt.show()
    plt.savefig(filename)

#script parameters
file_name='PROJECT_DATA_FULL'
seperator=';'
cluster_range=100
metric=['euclidean', 'l2', 'l1', 'manhattan', 'correlation', 'cosine']#, 'mahalanobis']

#data load
train = pd.read_csv(file_name+'.csv',seperator, low_memory=False)
train = train.stack().str.replace(',','.').unstack()

#define feature list from file 
feature_list = list(filter(lambda x: x[:2]=="TR", train.columns.to_series()))

customer_list = train['CUSTOMER_NUMBER'].values
customer_group_list = train['CUSTOMER_GROUP'].values
D=[] 

now=datetime.datetime.now()
now_str=now.strftime("%Y%m%d%H%M%S")

os.mkdir(now_str)
os.mkdir(now_str + "/PCA")
os.mkdir(now_str + "/export")
#os.mkdir(now_str + "/Group_Size")

temp_str=""
#kmeans

print("Number of Customer: ",len(customer_list))
print("Number of Feautures: ",len(feature_list))

sseX=[]
sseY=[]

for feature in feature_list:
    for f in train[feature].values:
        D.append(f)

plt.clf()
D=np.array(D)
D = D.astype(np.float64)
customer_size=len(customer_list)
feature_size=len(D)//customer_size
X = D.reshape((customer_size,feature_size))

silhouette=[]
sseX=[]
sseY=[]



for cs in range(cluster_range):
    x=datetime.datetime.now()
    cluster_size=cs+2
    kmeans = KMeans(n_clusters=cluster_size, max_iter=1000, init='k-means++').fit(X)
#silhouette_score calculation
    for m in metric:
        score = silhouette_score (X, kmeans.fit_predict(X), metric=m)
        silhouette.append([m,cluster_size,score])
#elbow calculation
    sseX.append(cluster_size)
    sseY.append(kmeans.inertia_)
    export=""

#    x_axis=[]
#    y_axis=[]
#    for c in range(0, cluster_size):
        #print(c, len(filter(c, kmeans.labels_)))
#        print(c, sum(1 if c==x else 0 for x in kmeans.labels_))
#        x_axis.append(c)
#        y_axis.append(sum(1 if c==x else 0 for x in kmeans.labels_))
#    createBarChart(x_axis, y_axis, now_str+"/Group_Size/Group_Size_For_Cluster_" + str(c))


    for i in range(len(customer_list)):
        export+=str(customer_list[i])+";"+str(customer_group_list[i])+";"+str(kmeans.labels_[i])+"\n"
    write_file('export/export'+file_name+'_'+str(cluster_size)+'.csv', export)
    #temp_temp_str="Total time =| {} | Calculation time=| {} | For n_clusters = | {} | silhouette score = | {} | metric used: | {} | sse: |{}".format((datetime.datetime.now()-now),(datetime.datetime.now()-x), cluster_size, score, m, kmeans.inertia_)
    temp_temp_str="Total time =| {} | Calculation time=| {} | For n_clusters = | {} | sse: |{}".format((datetime.datetime.now()-now),(datetime.datetime.now()-x), cluster_size, kmeans.inertia_)
    temp_str+=temp_temp_str+'\n'
    
#Export PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
            , columns = ['PC1', 'PC2'])
    plt.clf()
    pd.DataFrame(data = principalComponents).to_csv(now_str+'/PCA/PCA_For_Cluster_'+str(cluster_size)+'_'+file_name+'.csv')
    plt.scatter(principalDf['PC1'].values,principalDf['PC2'].values, c=kmeans.labels_, cmap='rainbow')
    plt.savefig(now_str+'/PCA/PCA_For_Cluster_'+str(cluster_size)+'_'+file_name+'.png')
    

    print(temp_temp_str)


#Export Elbow
plt.clf()
plt.plot(sseX, sseY, linewidth=3)
plt.savefig(now_str+'/elbow_'+str(cluster_range)+'_'+file_name+'.png')

#Export Log
write_file('log_'+file_name+'.txt', temp_str)


#Export silhouette
for m in metric:
    silX = [i[1] for i in silhouette if i[0]==m]
    silY = [i[2] for i in silhouette if i[0]==m]
    plt.clf()
    plt.plot(silX, silY, linewidth=2)
    plt.savefig(now_str+'/silhouette_'+m+'.png')
