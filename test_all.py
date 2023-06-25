from prepare_datasets import get_all_datasets
from k_centers import k_centers,cluster_list,get_cluster_radiuses,get_r
from utils import mink_dist, dist_matrix

from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import numpy as np
import time


if __name__ == "__main__":
	np.random.seed(123456)
	datasets = get_all_datasets()
	j = 1
	start_total = time.time()
	plist = [1,2,10]
	for points,clss in datasets:
		assert(points.shape[0] >= 700)
		dist_matrixes = []
		start_dataset = time.time()
		print("_"*100)
		print(f"Now for dataset number {j}:")
		if j <= 8:
			j = j+1
			continue;
		k = np.unique(clss).shape[0]
		n = points.shape[0]
		dim = points.shape[1]
		print(f"k = {k}, n = {n}, dim = {dim}")
		for p in plist:
			start_p = time.time()
			print("="*50)
			print(f"For p = {p} in the Minkowski Distance:")
			dist_f = lambda v1,v2: mink_dist(v1,v2,p)
			print("Computing distance matrix...")
			start_dist_matr = time.time()
			dist_matr = dist_matrix(points,dist_f)
			dist_matrixes += [dist_matr]
			print(f"Took {time.time()-start_dist_matr} seconds to compute distance matrix with p={p}")
			n_iter = 30
			print(f"Will now run {n_iter} executions of our algorithm with p={p}")
			smallest_r = 1e18
			biggest_sc = -1
			biggest_ar = -1
			for i in range(n_iter):
				print("-"*50)
				start_time_i = time.time()
				centers = k_centers(points,k,dist_matr)
				clist = cluster_list(points,centers,dist_matr)
				radiuses = get_cluster_radiuses(points,clist,dist_matr)
				r = get_r(radiuses)
				print(f"r = {r}")
				smallest_r = min(smallest_r,r)
				assert(clss.shape == clist.shape)
				ar = adjusted_rand_score(clss,clist)
				print(f"Adjusted rand score = {ar}")
				biggest_ar = max(ar,biggest_ar)
				sc = silhouette_score(dist_matr,clist,metric="precomputed")
				print(f"Sillhouette score = {sc}")
				biggest_sc = max(biggest_sc,sc)
				print(f"Time for this execution = {time.time()-start_time_i} secs")
			print("-"*50)
			print(f"Smallest radius for our method was = {smallest_r}")
			print(f"Biggest Adjusted Rand Score for our method was = {biggest_ar}")
			print(f"Biggest Sillhouette Score for our method was = {biggest_sc}")

			print(f"Total time for p={p} on dataset {j} = {time.time()-start_p} secs")
		print("-"*50)
		print(f"Now the KMeans on dataset {j}:")
		start_time_km = time.time()
		clist2 = KMeans(k,n_init=n_iter).fit_predict(points)
		assert(clss.shape == clist2.shape)
		ar2 = adjusted_rand_score(clss,clist2)
		print(f"adjusted rand score km = {ar2}")
		x = 0
		for dist_matr in dist_matrixes:
			radiuses2 = get_cluster_radiuses(points,clist2,dist_matr)
			r2 = get_r(radiuses2)
			print(f"For p = {plist[x]}, r for KMeans = {r2}")
			sc2 = silhouette_score(dist_matr,clist2,metric="precomputed")
			print(f"For p = {plist[x]}, sillhouette score km = {sc2}")
			x = x+1
		print(f"Time for this execution km = {time.time()-start_time_km} secs")
		print("-"*50)
		print(f"Total time for dataset {j} = {time.time()-start_dataset} secs")

		j = j+1	
	print("=-="*33)
	print(f"Total execution time = {time.time()-start_total} seconds")
