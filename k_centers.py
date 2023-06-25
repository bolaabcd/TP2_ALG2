import numpy as np
from utils import mink_dist, dist_matrix

# Returns the centroids given a KMeans classification result
def get_kmeans_centroids(points, cindexes):
	k = np.unique(cindexes).shape[0]
	cnt = np.zeros(k)
	for ind in range(points.shape[0]):
		cnt[cindexes[ind]] += 1

	centroids = np.zeros((k,points.shape[1]))
	i = 0
	for p in points:
		assert(cnt[cindexes[i]] != 0)
		centroids[cindexes[i]] += p/cnt[cindexes[i]]
		i = i+1
	return centroids

# Computes the radiuses of each cluster, given the centroids
def get_kmeans_cluster_radiuses(points,centroids,cindexes,dist):
	radiuses = np.zeros(centroids.shape[0])
	for i in range(points.shape[0]):
		radiuses[cindexes[i]] = max(radiuses[cindexes[i]],dist(points[i],centroids[cindexes[i]]))
	return radiuses

# Lists the cluster of each point, given the indexes of each center computer with the K-centers method we learned in class
def cluster_list(points,centers,dists):
	ans = np.zeros(points.shape[0],dtype=int)
	for i in range(points.shape[0]):
		mini = 1e18
		center = -1
		for j in centers:
			if dists[i][j] < mini:
				mini = dists[i][j]
				center = j
		assert(center != -1)
		ans[i] = center
	return ans

# Computes the radius of each cluster, if given which cluster each point belongs to
def get_cluster_radiuses(points,cluster_of_point,dists):
	radiuses = np.zeros(cluster_of_point.shape[0])
	for i in range(points.shape[0]):
		radiuses[i] = max(radiuses[i],dists[i][cluster_of_point[i]])
	return radiuses

# Computes the maximum radius, of all clusteres
def get_r(radiuses_list):
	return radiuses_list.max()

# Computing k centers (returns the INDEXES of the centers), given distance matrix:
def k_centers(points, k, dists):
	n = points.shape[0]

	if k >= n:
		return np.array(range(points.shape[0]))

	ans = np.zeros(k,dtype=int)
	# First point is chosen randomly:
	ans[0] = np.random.randint(0,points.shape[0])

	# Vector of distances: dist_from_ans[i] = dist(i,C)
	dist_from_ans = np.zeros(n)
	for i in range(1,n):
		dist_from_ans[i] = dists[ans[0]][i]
	# Adding each point to the solution
	for i in range(1,k):
		selected = np.argmax(dist_from_ans)
		# Updating distances
		for j in range(n):
			dist_from_ans[j] = min(dist_from_ans[j],dists[j][selected])
		ans[i] = selected
	return ans

# The code below only runs when this file is not used as a module. This should be used only for debugging, the experiments are implemented in another file.
if __name__ == "__main__":
	np.random.seed(123456)
	n = int(input("Type the number of the data points: "))
	dim = int(input("Type the dimension of the data points: "))

	print("Now inform all the points (one point per line, dimensions space-separated): ")
	points = np.zeros((n,dim),dtype=float)
	for i in range(n):
		points[i] = np.array(input().split(' '),dtype=float)

	p = int(input("Type the value of p for the Minkowski Distance: "))
	# Defining Minkowski Distance for this value of p:
	dist_f = lambda v1,v2: mink_dist(v1,v2,p)
	# Find matrix of distances for this value of p:
	dist_matr = dist_matrix(points,dist_f)

	k = int(input("Type the number of centers to find: "))

	n_iter = int(input("Type the number of iterations you want to run: "))
	for i in range(n_iter):
		centers = k_centers(points, k, dist_matr)
		print("Indexes of the selected points: ")
		print(centers)
		clist = cluster_list(points,centers,dist_matr)
		print("Cluster list of each point: ")
		print(clist)
		radiuses = get_cluster_radiuses(points,clist,dist_matr)
		print("Cluster radiuses:")
		print(radiuses)
		r = get_r(radiuses)
		print(f"r = {r}")
