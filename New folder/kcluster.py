
import pandas as pd
from tkinter import Y
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'A:\BELAJAR_DATA\6 bos gigih\3399797.csv')
df.head()

    #data distribution check
# fig, axes = plt.subplots(1,3,figsize=(20,10))
# sns.boxplot(ax=axes[0], y="total_gross_revenue_amt", data=df)
# sns.boxplot(ax=axes[1], y="total_sales_count", data=df, color="yellow")
# sns.boxplot(ax=axes[2], y="avg_ticket_size_amt", data=df, color="red")

# plt.show()

    #Regression plot
fig, axes = plt.subplots(1,1,figsize=(12,12))
sns.regplot(x="total_sales_count", y="avg_ticket_size_amt", data=df)
plt.show()

    #Data Manipulation(minmaxscalar)
from sklearn.preprocessing import MinMaxScaler
x=df[['total_sales_count', 'avg_ticket_size_amt']]

cols= x.columns
ms=MinMaxScaler()
X= ms.fit_transform(x)

df[['total_sales_count_ms', 'avg_ticket_size_amt_ms']] = X

#df.head()

    #Find Optimal Number of Cluster (K)
from sklearn.cluster import KMeans
# cs=[]
# for i in range (1,11):
#     kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=300,n_init=10, random_state=0)
#     kmeans.fit(df[['total_sales_count_ms', 'avg_ticket_size_amt_ms']])
#     cs.append(kmeans.inertia_)
# plt.plot(range(1,11),cs)
# plt.title('THE ELBOW METHOD')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Distortion')
# plt.show()

    #Apply Kmeans Clustering
kmeans=KMeans(n_clusters=3)
kmeans.fit(df[['total_sales_count_ms', 'avg_ticket_size_amt_ms']].copy())
df['cluster']=kmeans.predict(df[['total_sales_count_ms', 'avg_ticket_size_amt_ms']].copy())

graph = sns.lmplot(x='total_sales_count', y='avg_ticket_size_amt', data=df, hue='cluster', fit_reg=False)
sns.regplot(x="total_sales_count", y="avg_ticket_size_amt", data=df, scatter=True, scatter_kws={"zorder:-1"})

plt.show()

