import csv
from pre_process import get_nframe_video, split_subj_, sort_dataFile_list_
import pathlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy import  signal
import json
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

def get_HR_path(dataDir):
	return sorted(list(dataDir.rglob("*/source2/gt_HR.csv")))

def get_wave_path(dataDir):
	return sorted(list(dataDir.rglob("*/source2/wave.csv")))

def get_HR(path, dataset):
	if dataset == "VIPL":
		first = True
		data = []
		with open(path,"r") as f:
			for line in f.readlines():
				if first:
					first = False
					continue
				else:
					data.append(int(line.strip()))
	elif dataset == "PURE":
		data = []
		with open(path, "r") as json_file:
			json_data = json.load(json_file)
			for p in json_data['/FullPackage']:
				data.append(p['Value']['pulseRate'])

	return np.mean(data)

def get_wave_HR(path, fs, dataset):
	if dataset =="VIPL":
		if path.suffix == ".csv":
			first = True
			data = []
			with open(path,"r") as f:
				for line in f.readlines():
					if first:
						first = False
						continue
					else:
						data.append(int(line.strip()))
			wave = np.array(data)
		elif path.suffix == ".hdf5":
			f1 = h5py.File(path, 'r')
			wave = np.array(f1['pulse'])

	elif dataset == "PURE":
		pulse = []
		with open(path, "r") as json_file:
			json_data = json.load(json_file)
			for p in json_data['/FullPackage']:
				pulse.append(p['Value']['waveform'])
		wave = np.array(pulse)

	#calculating HR from wave
	N = 30 * fs
	pulse_fft = np.expand_dims(wave, 0)
	f, pxx = signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
	fmask = np.argwhere((f >= 0.75) & (f <= 3))  # regular Heart beat are 0.75*60 and 2.5*60
	frange = np.take(f, fmask)
	HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60
	return HR

def main(dataset):

	if dataset == "VIPL":
		dataDir = "/home/bhargav/vipl/hdf5"
		original = False
		if original:
			outputPath = "/home/bhargav/MTTS-CAN/code/vipl-HR-HR-OR.csv"
			gtHRPaths = get_HR_path(pathlib.Path(dataDir).parent / 'data')
			gtwavePaths = get_wave_path(pathlib.Path(dataDir).parent / 'data')
			fs = 62
		else:
			outputPath = "/home/bhargav/MTTS-CAN/code/vipl-HR-HR-Sampled.csv"
			subTrain, subDev, subTest = split_subj_("/home/bhargav/vipl/data/", "VIPL")
			path_of_video_tr = sort_dataFile_list_(dataDir, subTrain, dataset, trainMode=True)
			path_of_video_test = sort_dataFile_list_(dataDir, subTest, dataset, trainMode=True)
			path_of_video_val = sort_dataFile_list_(dataDir, subDev, dataset, trainMode=True)
			paths = path_of_video_val + path_of_video_test + path_of_video_tr
			gtHRPaths = get_HR_path(pathlib.Path(dataDir).parent / 'data')
			gtwavePaths = sorted(paths)
			for path in gtHRPaths:
				filename = "/home/bhargav/vipl/hdf5/" + "-".join(str(path.parent).split("/")[-3:]) + ".hdf5"
				if filename not in gtwavePaths:
					print("Removing", path)
					gtHRPaths.remove(path)

			fs = 30

	elif dataset=="PURE":
		dataDir = "/home/bhargav/PURE/"
		subTrain, subDev, subTest = split_subj_("/home/bhargav/PURE/", "PURE")
		paths = list(pathlib.Path(dataDir).rglob("*.json"))
		outputPath = "/home/bhargav/MTTS-CAN/code/PURE-HR-HR.csv"
		fs = 60
		gtHRPaths = []
		gtwavePaths = []
		gtHRPaths = paths
		gtwavePaths = paths

	with open(outputPath,"w") as csvfile:
		filewriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow(["Name HR","Name wave","HR-GT","HR"])
		gt = []
		wave = []
		for i in tqdm(range(len(gtwavePaths))):
			print(gtHRPaths[i],gtwavePaths[i])
			gtHR = get_HR(gtHRPaths[i], dataset)
			waveHR = get_wave_HR(pathlib.Path(gtwavePaths[i]),fs, dataset)
			gt.append(gtHR)
			wave.append(waveHR)
			filewriter.writerow([gtHRPaths[i],gtwavePaths[i],gtHR,waveHR])
		mae = mean_absolute_error(gt, wave)
		print("MAE", mae)

if __name__=="__main__":
	main(dataset="PURE")