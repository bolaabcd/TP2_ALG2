import numpy as np

# Minkowski Distance:
def mink_dist(v1, v2, p):
	return (np.abs((v1-v2))**p).sum()**(1/p)

# Get matrix of distances between all pairs of points
def dist_matrix(points, dist_f):
	n = points.shape[0]
	ans = np.zeros((n,n),dtype=float)
	for i in range(n):
		for j in range(n):
			ans[i][j] = dist_f(points[i],points[j])
	return ans
