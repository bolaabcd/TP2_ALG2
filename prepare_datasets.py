import urllib.request
import os
import shutil
import numpy as np
import time
import subprocess

# Basic function to download and unzip a dataset
def download_and_unzip(url,name,unzipped_name):
	os.makedirs("data/",exist_ok = True)
	os.makedirs("data/",exist_ok = True)
	file_path = "data/"+name
	if not os.path.exists(file_path):
		urllib.request.urlretrieve(url,file_path)
	if not os.path.exists("data/"+unzipped_name):
		shutil.unpack_archive(file_path,"data/")
		if os.path.exists("data/"+unzipped_name[:-4]+".rar"):
			print("+"*100)
			print("WARNING: will need unrar installed, to unzip a .rar file. You might want to run 'sudo apt-get install unrar'")
			print("+"*100)
			subprocess.run(["unrar", "e", f"{'data/'+unzipped_name[:-4]+'.rar'}"])
			subprocess.run(["mv","pd_speech_features.csv","data/pd_speech_features.csv"])

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

def get_csv_data(unzipped_name, ninstances, nfeatures, separator = ',', nheaders = 1, target = -1, normalize = False, startcol = 0):
	time_hand = time.time()
	file_path = "data/"+unzipped_name
	f = open(file_path,'r')
	data = np.zeros((ninstances,nfeatures)) # each line is an instance, each column a pixel
	classes = np.zeros((ninstances)) # the correct class of each instance
	j = -nheaders
	for line in f.readlines():
		# First line is header
		if j < 0:
			j = j+1
			continue
		info = line.split(separator)
		# This dataset sometimes has commas inside of the data... Fixing that here:
		if unzipped_name == "Grisoni_et_al_2016_EnvInt88.csv" and info[0][0] == '"':
			info = info[1:]
		info = info[startcol:]
		# Ignoring invalid lines for seventh dataset:
		if not (len(info) > 1 and info[1].replace('.','',1).isdigit()) or (len(info) > 10 and info[9] == ''):
			continue
		# Ignoring label:
		if target != -1 and target != 0:
			data[j] = np.concatenate((info[:target],info[target+1:]))
		elif target == -1:
			data[j] = info[:-1]
		else:
			data[j] = info[1:]
		classes[j] = info[target]
		j = j+1
	print(f"Found {j} valid lines, expected {ninstances}.")
	f.close()
	print(f"Took {time.time()-time_hand} seconds to prepare {unzipped_name} Data.")
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
		("https://archive.ics.uci.edu/static/public/186/wine+quality.zip","wine+quality.zip","winequality-white.csv"),
		("https://archive.ics.uci.edu/static/public/445/absenteeism+at+work.zip","Absenteeism_at_work.zip","Absenteeism_at_work.csv"),
		("https://archive.ics.uci.edu/static/public/176/blood+transfusion+service+center.zip","blood+transfusion+service+center.zip","transfusion.data"),
		("https://archive.ics.uci.edu/static/public/470/parkinson+s+disease+classification.zip","parkinson+s+disease+classification.zip","pd_speech_features.csv"),
		("https://archive.ics.uci.edu/static/public/475/audit+data.zip","audit+data.zip","audit_data/trial.csv"),
		("https://archive.ics.uci.edu/static/public/510/qsar+bioconcentration+classes+dataset.zip","qsar+bioconcentration+classes+dataset.zip","Grisoni_et_al_2016_EnvInt88.csv"),
		("https://archive.ics.uci.edu/static/public/522/south+german+credit.zip","south+german+credit.zip","SouthGermanCredit.asc"),
		("https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip","banknote+authentication.zip","data_banknote_authentication.txt")
	]
	i = 1
	for url, name, unzipped_name in urls_names:
		time_i = time.time()
		download_and_unzip(url,name,unzipped_name)
		print(f"Took {time.time()-time_i} seconds to download and unzip dataset {i}: {unzipped_name}")
		i = i+1
	print("Will now prepare all datasets...")
	ans = [
		get_Handwritten_Digit_data(urls_names[0][2]),
		get_csv_data(urls_names[1][2],1599,11,separator=';'),
		get_csv_data(urls_names[2][2],4898,11,separator=';'),
		get_csv_data(urls_names[3][2],740,20,separator=';'),
		get_csv_data(urls_names[4][2],748,4),
		get_csv_data(urls_names[5][2],756,754, nheaders=2,target=1),
		get_csv_data(urls_names[6][2],772,17),
		get_csv_data(urls_names[7][2],779,10,target=-2,startcol=3),
		get_csv_data(urls_names[8][2],1000,20,separator=' '),
		get_csv_data(urls_names[9][2],1372,4,nheaders=0)
	]
	print(f"Took {time.time()-time_data} seconds to prepare all datasets")
	return ans

if __name__ == "__main__":
	datasets = get_all_datasets()
	print(f"We have {len(datasets)} datasets.")
