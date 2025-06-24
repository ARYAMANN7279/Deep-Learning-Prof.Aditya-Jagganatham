import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import datasets

irisset = datasets.load_iris()
X = irisset.data
y = irisset.target

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet',s=10)
plt.title('Original Data')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

ypred=KMeans(n_clusters=3,random_state=0).fit_predict(X)
plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=ypred, cmap='jet',s=10)
plt.title('Predicted Data')
plt.grid(2,which='both')
plt.axis('tight')
plt.show()

SSE = []
for k in range(1, 11):
    kmeans_SSE = KMeans(n_clusters=k, random_state=0)
    kmeans_SSE.fit(X)
    SSE.append(kmeans_SSE.inertia_)
    
plt.figure(3)
plt.plot(range(1, 11), SSE)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title('SSE Plot for Elbow Method')
plt.grid(3)
plt.axis('tight')
plt.show()

SS = []
for k in range(2, 11):
    kmeans_SS = KMeans(n_clusters=k, random_state=0)
    kmeans_SS.fit_predict(X)
    score = silhouette_score(X, kmeans_SS.labels_, metric='euclidean')
    SS.append(score)
    
plt.figure(4)
plt.plot(range(2, 11), SS)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette score")
plt.title('Silhouette Score Plot for Kmeans')
plt.grid(4)
plt.axis('tight')
plt.show()


