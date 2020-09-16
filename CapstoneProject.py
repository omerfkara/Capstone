
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

#data load
train = pd.read_csv("PROJECT_DATA.csv",";")
#data read
full_feature_list=['B001','B002','B003','B004','B005','B006','B007','B008','B009','B010','B011','B012','B013','B014','B015','B016','B017','B018','B019','B020','B021','B022','B023','B024','B025','B026','B027','B028','B029','B030','B031','B032','B033','B034','B035','B036','B037','B038','B039','B040','B041','B042','B043','B044','B045','B046','B047','B048','B049','B050','B051','B052','B053','B054','B055','B056','B057','B058','B059','B060','B061','B062','B063','B064','B065','B066','B067','B068','B069','B070','B071','B072','B073','B074','B075','B076','B077','B078','B079','B080','B081','B082','B083','B084','B085','B086','B087','B088','B089','B090','B091','B092','B093','B094','B095','B096','B097','B098','B099','B100','B101','B102','B103','B104','B105','B106','B107','B108','B109','B110','B111','B112','B113','B114','B115','B116','B117','B118','B119','B120','B121','B122','B123','B124','B125','B126','B127','B128','B129','B130','B131','B132','B133','B134','B135','B136','B137','B138','B139','B140','B141','B142','B143','B144','B145','B146','B147','B148','B149','B150','B151','B152','B153','B154','B155','B156','B157','B158','B159','B160','B161','B162','B163','B164','B165','B166','B167','B168','B169','B170','B171','B172','B173','B174','B175','B176','B177','B178','B179','B180','B181','B182','B183','B184','B185','B186','B187','B188','B189','B190','B191','B192','B193','B194','B195','B196','B197','B198','B199','B200','B201','B202','B203','B204','B205','B206','B207','B208','B209','B210','B211','B212','B213','B214','B215','B216','B217','B218','B219','B220','B221','B222','B223','B224','B225','B226','B227','B228','B229','B230','B231','B232','B233','B234','B235','B236','B237','B238','B239','B240','B241','B242','B243','B244','B245','B246','B247','B248','B249','B250','B251','B252']
selected_feature_list = ['B116', 'B098', 'B094', 'B081', 'B121', 'B126', 'B141', 'B159', 'B185', 'B195', 'B196', 'B201', 'B209', 'B211']
feature_list = full_feature_list
a1 = train['CUSTOMER'].values
D=[] 

for feature in feature_list:
    for f in train[feature].values:
        D.append(f)


plt.clf()
D=np.array(D)
customer_size=len(a1)
feature_size=len(D)//customer_size
X = D.reshape((customer_size,feature_size))

silhouetteX= []
silhouetteY= []

file_name='_featuresize'+str(feature_size)+'_'+str(datetime.datetime.now())
file_name='_featuresize'+str(feature_size)

temp_str=""
#kmeans
x0=datetime.datetime.now()
now_str=x0.strftime("%Y%m%d%H%M%S")
os.mkdir(now_str)
cluster_range=100
for cs in range(cluster_range):
    x=datetime.datetime.now()
    cluster_size=cs+2
    kmeans = KMeans(n_clusters=cluster_size).fit(X)
    score = silhouette_score (X, kmeans.fit_predict(X), metric='euclidean')
    silhouetteX.append(cluster_size)
    silhouetteY.append(score)
    export=""
    for i in range(len(a1)):
        export+=str(a1[i])+";"+str(kmeans.labels_[i])+"\n"
    write_file('export'+file_name+'_'+str(cluster_size)+'.csv', export)
    temp_str+="Total time = {} | Calculation time= {} | For n_clusters = {} | silhouette score is {})".format((datetime.datetime.now()-x0),(datetime.datetime.now()-x), cluster_size, score)+'\n'
    
#PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2'])
#    print(principalDf.head())
#    print(principalDf['PC1'].values)
    plt.clf()
    plt.scatter(principalDf['PC1'].values,principalDf['PC2'].values, c=kmeans.labels_, cmap='rainbow')
    plt.savefig(now_str+'/PCA_'+file_name+'_'+str(cluster_size)+'.png')
#/PCA
    
    print("Total time = {} | Calculation time= {} | For n_clusters = {} | silhouette score is {})".format((datetime.datetime.now()-x0),(datetime.datetime.now()-x), cluster_size, score))
plt.clf()
#plt.scatter(silhouetteX, silhouetteY)
plt.plot(silhouetteX, silhouetteY, linewidth=3)
#plt.show()
plt.savefig(now_str+'/silhouette_'+file_name+'_'+str(cluster_range)+'.png')

write_file('log_'+file_name+'.txt', temp_str)

'''
#Grafik
'''


print('fin')