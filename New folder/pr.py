import pandas as pd
from tkinter import Y
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv(r'A:\BELAJAR_DATA\6 bos gigih\20092.csv', delimiter=';', decimal=',')
print(df1)
# fig, axes = plt.subplots(1,1,figsize=(12,12))
# sns.regplot(x="total_quantity", y="ticket_size", data=df1)
# plt.show()

from sklearn.preprocessing import MinMaxScaler
x=df1[['total_quantity', 'ticket_size']]

cols= x.columns
ms=MinMaxScaler()
X= ms.fit_transform(x)

df1[['total_quantity_ms', 'ticket_size_ms']] = X

    #Find Optimal Number of Cluster (K)
# from sklearn.cluster import KMeans
# cs=[]
# for i in range (1,11):
#     kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=300,n_init=10, random_state=0)
#     kmeans.fit(df1[['total_spends_ms', 'avg_ticket_size_amt_ms']])
#     cs.append(kmeans.inertia_)
# plt.plot(range(1,11),cs)
# plt.title('THE ELBOW METHOD')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Distortion')
# plt.show()

    #Apply Kmeans Clustering
kmeans=KMeans(n_clusters=3)
kmeans.fit(df1[['total_spends_ms', 'avg_ticket_size_amt_ms']].copy())
df1['cluster']=kmeans.predict(df1[['total_spends_ms', 'avg_ticket_size_amt_ms']].copy())

graph = sns.lmplot(x='total_spends', y='AVERAGE_QT_AMT', data=df1, hue='cluster', fit_reg=False)
sns.regplot(x="total_spends", y="avg_qty_amt", data=df1, scatter=True, scatter_kws={"zorder:-1"})

plt.show()




