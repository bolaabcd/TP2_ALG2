import urllib.request
import os
import zipfile
import numpy as np
import time

# Basic function to download and unzip a dataset
def download_and_unzip(url,name,unzipped_name):
	os.makedirs("data/",exist_ok = True)
	os.makedirs("data/",exist_ok = True)
	file_path = "data/"+name
	if not os.path.exists(file_path):
		urllib.request.urlretrieve(url,file_path)
	if not os.path.exists("data/"+unzipped_name):
		zipfile.ZipFile(file_path, 'r').extractall("data/")

def get_Handwritten_Digit_data(unzipped_name, normalize = False):
	time_hand = time.time()
	file_path = "data/"+unzipped_name
	f = open(file_path,'r')
	data = np.zeros((1593,256)) # each line is an instance, each column a pixel
	classes = np.zeros((1593)) # the correct class of each instance
	j = 0
	for line in f.readlines():
		bits = line.split(' ')
		digit = -1
		for i in range(10): # find out the digit
			if bits[-11:-1][i] == '1':
				digit = i
				break
		data[j] = bits[:-11]
		assert(digit != -1)
		classes[j] = digit
		j = j+1
	f.close()
	print(f"Took {time.time()-time_hand} seconds to prepare Handwritten Data.")
	if normalize == True:
		data = data-data.mean(axis = 0) # Centralization
		data = data/data.std(axis = 0) # Making the standard deviation become 1
	return (data,classes)

def get_Wine_Data(unzipped_name, normalize = False):
	time_hand = time.time()
	file_path = "data/"+unzipped_name
	f = open(file_path,'r')
	n = 1599 if unzipped_name == "winequality-red.csv" else 4898
	data = np.zeros((n,11)) # each line is an instance, each column a pixel
	classes = np.zeros((n)) # the correct class of each instance
	j = -1
	for line in f.readlines():
		# First line is header
		if j == -1:
			j = j+1
			continue
		info = line.split(';')
		data[j] = info[:-1]
		classes[j] = info[-1]
		j = j+1
	f.close()
	print(f"Took {time.time()-time_hand} seconds to prepare Handwritten Data.")
	if normalize == True:
		data = data-data.mean(axis = 0) # Centralization
		data = data/data.std(axis = 0) # Making the standard deviation become 1
	return (data,classes)

# Return datasets in the format matrix, class; in which the matrix is n*m, n instances and m (numeric) attributes
def get_all_datasets():
	print("Preparing datasets...")
	time_data = time.time()
	urls_names = [
		("https://archive.ics.uci.edu/static/public/178/semeion+handwritten+digit.zip", "handwritten_digits.zip", "semeion.data"),
		("https://archive.ics.uci.edu/static/public/186/wine+quality.zip","wine+quality.zip","winequality-red.csv"),
		("https://archive.ics.uci.edu/static/public/186/wine+quality.zip","wine+quality.zip","winequality-white.csv")
	]
	i = 1
	for url, name, unzipped_name in urls_names:
		time_i = time.time()
		download_and_unzip(url,name,unzipped_name)
		print(f"Took {time.time()-time_i} seconds to download and unzip dataset {i}")
		i = i+1
	print("Will now prepare all datasets...")
	ans = [
		get_Handwritten_Digit_data(urls_names[0][2]),
		get_Wine_Data(urls_names[1][2]),
		get_Wine_Data(urls_names[1][2])
	]
	print(f"Took {time.time()-time_data} seconds to prepare all datasets")
	return ans

if __name__ == "__main__":
	datasets = get_all_datasets()
	print(f"We have {len(datasets)} datasets.")
