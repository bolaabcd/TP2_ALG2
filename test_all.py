from prepare_datasets import get_all_datasets
from k_centers import k_centers,cluster_list,get_cluster_radiuses,get_r, get_kmeans_centroids,get_kmeans_cluster_radiuses
from utils import mink_dist, dist_matrix

from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import numpy as np
import time


# This file is intended to be used as a script only, but the 'if __name__ == "__main__"' part avoids it executing if someone tries to import it as a module
if __name__ == "__main__":
	np.random.seed(123456)
	outst = ""
	datasets = get_all_datasets()
	j = 1
	start_total = time.time()
	plist = [1,2,10]
	for points,clss in datasets:
		assert(points.shape[0] >= 700)
		start_dataset = time.time()
		print("_"*100)
		print(f"Now for dataset number {j}:")
		k = np.unique(clss).shape[0]
		n = points.shape[0]
		dim = points.shape[1]
		print(f"k = {k}, n = {n}, dim = {dim}")
		n_iter = 30
		for p in plist:
			outst += f"p = {p}, dataset = {j}\n"
			start_p = time.time()
			print("="*50)
			print(f"For p = {p} in the Minkowski Distance:")
			dist_f = lambda v1,v2: mink_dist(v1,v2,p)
			print("Computing distance matrix...")
			start_dist_matr = time.time()
			dist_matr = dist_matrix(points,dist_f)
			print(f"Took {time.time()-start_dist_matr} seconds to compute distance matrix with p={p}")
			print(f"Will now run {n_iter} executions of our algorithm with p={p}")
			smallest_r = 1e18
			biggest_sc = -1
			biggest_ar = -1
			rads = np.zeros(n_iter)
			ars = np.zeros(n_iter)
			scs = np.zeros(n_iter)
			ts = np.zeros(n_iter)
			rads_km = np.zeros(n_iter)
			ars_km = np.zeros(n_iter)
			scs_km = np.zeros(n_iter)
			ts_km = np.zeros(n_iter)
			for i in range(n_iter):
				print("-"*50)
				start_time_i = time.time()
				centers = k_centers(points,k,dist_matr)
				clist = cluster_list(points,centers,dist_matr)
				radiuses = get_cluster_radiuses(points,clist,dist_matr)
				r = get_r(radiuses)
				rads[i] = r
				print(f"r = {r}")
				assert(clss.shape == clist.shape)
				ar = adjusted_rand_score(clss,clist)
				ars[i] = ar
				print(f"Adjusted rand score = {ar}")
				sc = silhouette_score(dist_matr,clist,metric="precomputed")
				scs[i] = sc
				print(f"Sillhouette score = {sc}")
				tim = time.time()-start_time_i
				ts[i] = tim
				print(f"Time for this execution K-Centers= {tim} secs")
				print(f"Now the KMeans:")
				start_time_km = time.time()
				km = KMeans(k,n_init=1)
				clist2 = km.fit_predict(points)
				assert(clss.shape == clist2.shape)
				ar2 = adjusted_rand_score(clss,clist2)
				ars_km[i] = ar2
				print(f"adjusted rand score km = {ar2}")
				centroids = get_kmeans_centroids(points,clist2)
				radiuses2 = get_kmeans_cluster_radiuses(points,centroids,clist2,dist_f)
				r2 = get_r(radiuses2)
				rads_km[i] = r2
				print(f"r for KMeans = {r2}")
				sc2 = silhouette_score(dist_matr,clist2,metric="precomputed")
				scs_km[i] = sc2
				print(f"Sillhouette score K-Means = {sc2}")
				tim2 = time.time()-start_time_km
				ts_km[i] = tim2
				print(f"Time for this execution K-Means = {tim2} secs")
			print("-"*50)

			print(f"Total time for p={p} on dataset {j} = {time.time()-start_p} secs")
			print(f"Average Ajusted Rand Score for p={p} on dataset {j} = {ars.mean()}")
			print(f"Standard Deviation for Ajusted Rand Score for p={p} on dataset {j} = {ars.std()}")
			print(f"Average Silhouette Score for p={p} on dataset {j} = {scs.mean()}")
			print(f"Standard Deviation for Silhouette Score for p={p} on dataset {j} = {scs.std()}")
			print(f"Average Maximum Radius for p={p} on dataset {j} = {rads.mean()}")
			print(f"Standard Deviation for Maximum Radius Score for p={p} on dataset {j} = {rads.std()}")
			print(f"Average Time for p={p} on dataset {j} = {ts.mean()}")
			print(f"Standard Deviation of Time for p={p} on dataset {j} = {ts.std()}")
			print(f"KMeans Average Ajusted Rand Score for p={p} on dataset {j} = {ars_km.mean()}")
			print(f"KMeans Standard Deviation for Ajusted Rand Score for p={p} on dataset {j} = {ars_km.std()}")
			print(f"KMeans Average Silhouette Score for p={p} on dataset {j} = {scs_km.mean()}")
			print(f"KMeans Standard Deviation for Silhouette Score for p={p} on dataset {j} = {scs_km.std()}")
			print(f"KMeans Average Maximum Radius for p={p} on dataset {j} = {rads_km.mean()}")
			print(f"KMeans Standard Deviation for Maximum Radius Score for p={p} on dataset {j} = {rads_km.std()}")
			print(f"KMeans Average Time for p={p} on dataset {j} = {ts_km.mean()}")
			print(f"KMeans Standard Deviation of Time for p={p} on dataset {j} = {ts_km.std()}")
			outst += "K-Centers\n"
			outst += f"    Dataset {j}&{rads.mean():.3f}&{scs.mean():.3f}&{ars.mean():.3f}&{ts.mean():.3f}&{rads.std():.3f}&{scs.std():.3f}&{ars.std():.3f}&{ts.std():.3f}\\\\\n"
			outst += "K-Means\n"
			outst += f"    Dataset {j}&{rads_km.mean():.3f}&{scs_km.mean():.3f}&{ars_km.mean():.3f}&{ts_km.mean():.3f}&{rads_km.std():.3f}&{scs_km.std():.3f}&{ars_km.std():.3f}&{ts_km.std():.3f}\\\\\n"
		print(f"Total time for dataset {j} = {time.time()-start_dataset} secs")

		j = j+1	
	print("=-="*33)
	print(f"Total execution time = {time.time()-start_total} seconds")
	
	out = open("results.txt",'w')
	out.write(outst)
	out.close()
