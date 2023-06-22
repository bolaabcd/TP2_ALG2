import numpy as np
from utils import mink_dist, dist_matrix

# Computing k centers, given distance matrix:
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
		dist_from_ans[i] = dists[0][i]
	
	# Adding each point to the solution
	for i in range(1,k):
		selected = np.argmax(dist_from_ans)
		# Updating distances
		for j in range(n):
			dist_from_ans[j] = min(dist_from_ans[j],dists[j][selected])
		ans[i] = selected
	return points[ans]

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
		print("Pontos selecionados: ")
		print(k_centers(points, k, dist_matr))
