
import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


import matplotlib.pyplot as plt
from _datetime import date

from sklearn.decomposition import PCA

import os
import pdfkit


def write_file(fn, str):
    with open(now_str+'/'+fn, "w") as text_file:
        print(str, file=text_file)

def createBarChart(x_axis, y_axis,filename):
    y_pos = np.arange(len(x_axis))
    plt.clf()
    plt.bar(y_pos, y_axis, align='center', alpha=0.5)
    plt.xticks(y_pos, x_axis)
    plt.ylabel('Customer Count')
    plt.xlabel('Customer Group')
    plt.title('Cluster Size By Groups')
#    plt.show()
    plt.savefig(filename)

def cleanMe(s):
    s=s.replace(",","")
    s=s.replace("-","0")
    s=int(s)
    return s


#parameters

cluster_range = 2
metric_euclidean = 'euclidean'
metric_mahalanobis = 'mahalanobis'

metric = metric_mahalanobis

#data load
train_full = pd.read_csv("PROJECT_DATA_2020_05_19.csv",",")
full_customer_size=len(train_full["CUSTOMER_DESC"])
print("Toplam Müşteri Sayısı: {}",full_customer_size)
#train=train_full
#train=train_full[train_full.apply(lambda xx: int(xx["Total"].replace(".","")) > 10000, axis=1)]
train=train_full[train_full.apply(lambda xx: cleanMe(xx["Total"]) > 1000, axis=1)]
feature_list = list(filter(lambda x: x[:2]=="TR", train.columns.to_series()))
selected_customer_size=len(train["CUSTOMER_DESC"])

print("Seçilen Müşteri Sayısı: {}",selected_customer_size)

#data read
#full_feature_list=['TRAY020', 'TRAY030', 'TRAY010010', 'TRAY010030', 'TRAY010040', 'TRAY010050', 'TRAY010060', 'TRAY010070', 'TRAY010080', 'TRAY010090', 'TRAY010110', 'TRAY010120', 'TRAY010020010', 'TRAY010020020', 'TRAY010100010', 'TRAY010100020', 'TRBD010', 'TRBD030', 'TRBD060', 'TRBD070', 'TRBD080', 'TRBD090', 'TRBD100', 'TRBD020010', 'TRBD020020', 'TRBD020030', 'TRBD020040', 'TRBD020050', 'TRBD020060', 'TRBD040010', 'TRBD040020', 'TRBD050010', 'TRBD050020', 'TRBD050030', 'TRDY010010', 'TRDY010020', 'TRDY010040', 'TRDY010050', 'TRDY010060', 'TRDY010080', 'TRDY010100', 'TRDY010110', 'TRDY010120', 'TRDY010130', 'TRDY010140', 'TRDY010150', 'TRDY010160', 'TRDY010170', 'TRDY010180', 'TRDY010190', 'TRDY010200', 'TRDY010210', 'TRDY010220', 'TRDY020010', 'TRDY020020', 'TRDY020030', 'TREK010', 'TREK020', 'TREK030', 'TREK040', 'TRIT010', 'TRIT020', 'TRIT030', 'TRIT040', 'TRIT060', 'TRKD010010', 'TRKD010020', 'TRKD010030', 'TRKD020020', 'TRKD020030', 'TRKD020040', 'TRKD020050', 'TRKD020060', 'TRKD030010', 'TRKD030020', 'TRKD030030', 'TRKD030040', 'TRKD030050', 'TRKD030060', 'TRKD030070', 'TRKD030080', 'TRKD030090', 'TRKD030100', 'TRKD030110', 'TRKD030120', 'TRKD040010', 'TRKD040020', 'TRKD040030', 'TRKD040040', 'TRKD040050', 'TRKD040060', 'TRKD050010', 'TRKD050020', 'TRKD050040', 'TRKD050050', 'TRKD050060', 'TRKD050070', 'TRKD050080', 'TRKD050090', 'TRKD060010', 'TRKD060020', 'TRKD060030', 'TRKD060040', 'TRKD060050', 'TRKD060060', 'TRKD060070', 'TRKD070010', 'TRKD050030010', 'TRKD050030020', 'TRKD050030030', 'TRKD050030040', 'TRKD050030050', 'TRKD050030060', 'TRKD050030070', 'TRKD050030080', 'TRKD050030090', 'TRKD050030100', 'TRKD050030110', 'TRKD050030120', 'TRKD050030130', 'TRKK010', 'TRKK020', 'TRKK030', 'TRKK040', 'TRKK050', 'TRKK060', 'TRKY020', 'TRKY010010', 'TRKY010030', 'TRKY010040', 'TRKY010050', 'TRKY010060', 'TRKY010070', 'TRKY030010', 'TRKY030020', 'TRKY030070', 'TRKY030080', 'TRKY030090', 'TRMD020', 'TRMD130', 'TRMD010010', 'TRMD010020', 'TRMD040010', 'TRMD040020', 'TRMD040030', 'TRMD040040', 'TRMD040050', 'TRMD040060', 'TRMD040070', 'TRMD040080', 'TRMD060010', 'TRMD060020', 'TRMD060030', 'TRMD060040', 'TRMD060050', 'TRMD070010', 'TRMD070020', 'TRMD070040', 'TRMD070050', 'TRMD070060', 'TRMD070070', 'TRMD070080', 'TRMD070090', 'TRMD070100', 'TRMD070140', 'TRMD070160', 'TRMD070170', 'TRMD070180', 'TRMD070190', 'TRMD070200', 'TRMD070210', 'TRMD070220', 'TRMD070230', 'TRMD070240', 'TRMD070250', 'TRMD070270', 'TRMD070280', 'TRMD070300', 'TRMD070310', 'TRMD070340', 'TRMD070350', 'TRMD070360', 'TRMD070370', 'TRMD070380', 'TRMD070390', 'TRMD070400', 'TRMD070410', 'TRMD070420', 'TRMD080010', 'TRMD080020', 'TRMD080030', 'TRMD080050', 'TRMD080070', 'TRMD090010', 'TRMD100010', 'TRMD100020', 'TRMD100030', 'TRMD100040', 'TRMD110010', 'TRMD110020', 'TRMD110030', 'TRMD120010', 'TRMD120020', 'TRMD140020', 'TRMD140030', 'TRMD140040', 'TRMD140050', 'TRMD030010010', 'TRMD030010020', 'TRMD030010040', 'TRMD030010050', 'TRMD030010060', 'TRMD030020020', 'TRMD030020030', 'TRMD030020040', 'TRMD070030010', 'TRMD070030020', 'TRMD070150010', 'TRMD070150020', 'TRMD070320010', 'TRMD070320020', 'TRMD090020010', 'TRMD090020020', 'TRMD090020030', 'TRMK010', 'TRMK020', 'TRMU010', 'TRMU030', 'TRMU060', 'TRMU070', 'TRMU090', 'TRMU100', 'TRMU110', 'TRMU120', 'TRMU130', 'TRMU150', 'TRMU160', 'TRMU170', 'TRMU180', 'TRMU190', 'TRMU040010', 'TRMU040020', 'TRMU080010', 'TRMU080020', 'TRMU080030', 'TRMU080040', 'TRMU080050', 'TRMU080070', 'TRMU080090', 'TRMU080120', 'TRMU080130', 'TRMU140010', 'TRMU140020']
#selected_feature_list = ['B116', 'B098', 'B094', 'B081', 'B121', 'B126', 'B141', 'B159', 'B185', 'B195', 'B196', 'B201', 'B209', 'B211']

#print(train)

customer_list = train["CUSTOMER_DESC"].values
totals= train["Total"].values
D=[]

for feature in feature_list:
    i=0
    for f in train[feature].values:
        w=cleanMe(f)
        int_total=cleanMe(totals[i])
        if int_total > 15:
            w = 1
        else:
#            w=int(100*w/int_total)
            w = 0  # int(100 * w / int_total)
        D.append(w)
        i = i+1


plt.clf()
D=np.array(D)
customer_size=len(customer_list)
feature_size=len(D)//customer_size
X = D.reshape((customer_size,feature_size))

silhouetteX= []
silhouetteY= []

#File Folder Operation-----------------------------------------------------------
file_name=''+str(feature_size)+'_'+str(datetime.datetime.now())
file_name=''+str(feature_size)
now=datetime.datetime.now()
now_str=now.strftime("%Y%m%d%H%M%S")
os.mkdir(now_str)
os.mkdir(now_str + "/PCA")
os.mkdir(now_str + "/export")
os.mkdir(now_str + "/Group_Size")

#--------------------------------------------------------------------------------

temp_str=""


for cs in range(cluster_range):
    html_str=""
    x=datetime.datetime.now()
    cluster_size=cs+5
    kmeans = KMeans(n_clusters=cluster_size).fit(X)
    score = silhouette_score (X, kmeans.fit_predict(X), metric=metric)
    silhouetteX.append(cluster_size)
    silhouetteY.append(score)
    export=""
    #print(kmeans.labels_.groupby(0).agg(['count']))
    x_axis=[]
    y_axis=[]
    for c in range(0, cluster_size):
        #print(c, len(filter(c, kmeans.labels_)))
#        print(c, sum(1 if c==x else 0 for x in kmeans.labels_))
        x_axis.append(c)
        y_axis.append(sum(1 if c==x else 0 for x in kmeans.labels_))
    createBarChart(x_axis, y_axis, now_str+"/Group_Size/Group_Size_For_Cluster_" + str(c))

    for i in range(len(customer_list)):
        export+=str(customer_list[i])+";"+str(kmeans.labels_[i])+"\n"

    write_file('export/export'+file_name+'_'+str(cluster_size)+'.csv', export)
    temp_str+="Total time = {} | Calculation time= {} | For n_clusters = {} | silhouette score is {})".format((datetime.datetime.now()-now),(datetime.datetime.now()-x), cluster_size, score)+'\n'
    
#PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2'])
#    print(principalDf.head())
#    print(principalDf['PC1'].values)
    plt.clf()
    plt.scatter(principalDf['PC1'].values,principalDf['PC2'].values, c=kmeans.labels_, cmap='rainbow')
    plt.savefig(now_str+'/PCA/PCA_For_Cluster_'+str(cluster_size)+'.png')
#/PCA
    
    print("Total time = {} | Calculation time= {} | For n_clusters = {} | silhouette score is {})".format((datetime.datetime.now()-now),(datetime.datetime.now()-x), cluster_size, score))
plt.clf()
#plt.scatter(silhouetteX, silhouetteY)
plt.plot(silhouetteX, silhouetteY, linewidth=3)
#plt.show()
plt.savefig(now_str+'/silhouette_graph.png')

write_file('log_'+file_name+'.txt', temp_str)



feature_size=len(feature_list)
selection_criteria="Total Gross > 10000"
clustering_algo="KMeans"
html_output=""
html_output = html_output + "<html>"
html_output = html_output + "<head>"
html_output = html_output + "<style>"
html_output = html_output + ".pagebreak { page-break-before: always; } "
html_output = html_output + "</style>"
html_output = html_output + "<body>"

html_output = html_output + "<h2>Data Details</h2>"
html_output = html_output + "<p><b>Feature Size: </b>" + str(feature_size) + "</p>"
html_output = html_output + "<p><b>Total Customer Count: </b>" + str(full_customer_size) + "</p>"
html_output = html_output + "<p><b>Selected Customer Count: </b>" + str(selected_customer_size) + "</p>"
html_output = html_output + "<p><b>Selection Criteria: </b>" + selection_criteria + "</p>"
html_output = html_output + "<p><b>Clustering Algorithm: </b>" + clustering_algo + "</p>"
html_output = html_output + "<p><b>Metric: </b>" + metric + "</p>"

html_output = html_output + "<h2>Silhouette Graph</h2>"
html_output = html_output + "<img src='silhouette_graph.png'>"
for i in range(cluster_range-1):
    i=i+2
    html_output = html_output + "<div class='pagebreak'> </div>"

    html_output = html_output + "<h2>PCA Graph For Cluster Size:"+ str(i) +" </h2>"
    html_output = html_output + "<img src='PCA/PCA_For_Cluster_" + str(i) + ".png'>"

    html_output = html_output + "<h2>Customer Group Size For Cluster Size:" + str(i) + " </h2>"
    html_output = html_output + "<img src='Group_Size/Group_Size_For_Cluster_" + str(i) + ".png'>"
    html_output = html_output + "<div class='pagebreak'> </div>"

html_output = html_output + "</body>"
html_output = html_output + "</html>"

print(html_output)
write_file('report.html', html_output)

pdfkit.from_url(now_str+'/report.html', now_str+'/report.pdf')


print('fin')