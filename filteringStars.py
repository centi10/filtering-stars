import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('finalGravityStars.csv')
list = df.iloc[:,[3,4]].values

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
dist = df["Distance"].to_list()
gravity = df["Gravity"].to_list()

mass.sort()
radius.sort()
gravity.sort()
plt.plot(radius,mass)

plt.title("Radius & Mass of the Star")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

plt.plot(mass,gravity)

plt.title("Mass vs Gravity")
plt.xlabel("Mass")
plt.ylabel("Gravity")
plt.show()

plt.scatter(radius,mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 42)
  kmeans.fit(list)
  wcss.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=3,init = 'k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(list)
plt.figure(figsize = (10,6))
sns.scatterplot(list[y_kmeans == 0,0],list[y_kmeans == 0,1],color = 'yellow',label = 'Cluster 1')
sns.scatterplot(list[y_kmeans == 1,0],list[y_kmeans == 1,1],color = 'blue',label = 'Cluster 2')
sns.scatterplot(list[y_kmeans == 2,0],list[y_kmeans == 2,1],color = 'green',label = 'Cluster 3')
sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color = 'red',label = 'Centroids',s = 100,marker = ',')
plt.title('Clustering')
plt.xlabel('Mass')
plt.ylabel('Radius')
plt.legend()
plt.show()

df.head()

bools =[]
for d in df.Distance:
    if d<=100:
        bools.append(True)
    else:
        bools.append(False)

is_dist = pd.Series(bools)
star_dist=df[is_dist]
is_dist.head()
star_dist=df[is_dist]

star_dist.reset_index(inplace=True,drop=True)
star_dist.head()
star_dist.shape

gravity_bool = []
for g in star_dist.Gravity:
    if g<=350 and g>=150:
        gravity_bool.append(True)
    else :
        gravity_bool.append(False)

is_gravity = pd.Series(gravity_bool)
final_stars = star_dist[is_gravity]
final_stars.head()
final_stars.shape

final_stars.reset_index(inplace=True,drop=True)
final_stars.head()

final_stars.to_csv("filteredStars.csv")