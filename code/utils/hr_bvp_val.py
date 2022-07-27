
import numpy as np
from pre_process import get_nframe_video, split_subj_, sort_dataFile_list_
import matplotlib.pyplot as plt
import h5py
import csv
from scipy import signal
import json
from scipy.stats import pearsonr, normaltest
def save_fig(paths,dataset):
	if dataset == "VIPL":
		for path in paths:
			f1 = h5py.File(path, 'r')
			dXsub = np.array(f1["data"])
			dysub = np.array(f1['pulse'])
			dzsub = np.array(f1['original'])
			f1.close()
			if any(np.isnan(dysub)):
				print(path,"Has an error")
				raise NotImplementedError
			else:
				print(path,"All good boi")
			fig, ax = plt.subplots(3, 1)
			ax[0].imshow(dXsub[0, :, :, -3:])
			ax[1].plot(dysub)
			ax[2].plot(dzsub)
			plt.savefig("/home/bhargav/MTTS-CAN/code/data/" + path.strip(".hdf5").split("/")[-1] + ".jpg")
			plt.close(fig)
	elif dataset=="COHFACE":
		for path in paths:
			print(path)
			f1 = h5py.File(path, 'r')
			dXsub = np.array(f1["data"])
			dysub = np.array(f1['pulse'])
			f1.close()
			if any(np.isnan(dysub)):
				print(path,"Has an error")
				raise NotImplementedError
			else:
				print(path,"All good boi")
			fig, ax = plt.subplots(2,1)
			ax[0].imshow(dXsub[0,:,:,-3:])
			ax[1].plot(dysubt)
			plt.savefig("/home/bhargav/MTTS-CAN/code/cohface/"+"_".join(path.strip(".hdf5").split("/")[3:])+".jpg")
			plt.close(fig)

def bvp_hr(paths, dataset):
	if dataset == "COHFACE":
		fs = 256
		hr_total = []
		bvp_total = []
		with open("/code/cohface.csv", "w") as csvfile:
			filewriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
			filewriter.writerow(["Name","bvp_average","Max","Min","STD","HR"])
			for path in paths:
				original_dataPath = path.strip("_dataFile.hdf5") + "data.hdf5"
				print(original_dataPath)
				f1 = h5py.File(original_dataPath, 'r')
				pulse = np.array(f1['pulse'])
				f1.close()
				N = 30 * fs
				pulse_fft = np.expand_dims(pulse, 0)
				f, pxx = signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
				fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
				frange = np.take(f, fmask)
				HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60
				BVP_mean = np.mean(pulse)
				BVP_max = np.max(pulse)
				BVP_min = np.min(pulse)
				BVP_std = np.std(pulse)
				hr_total.append(HR)
				bvp_total.append(BVP_mean)
				filewriter.writerow([original_dataPath,BVP_mean,BVP_max,BVP_min,BVP_std, HR])
		_, p = normaltest(hr_total)
		print("HR",p)
		_, p = normaltest(bvp_total)
		print("BVP",p)
		plt.scatter(hr_total, bvp_total)
		corr, _ = pearsonr(hr_total, bvp_total)
		plt.xlabel("HR")
		plt.ylabel("BVP value")
		plt.text(max(hr_total)//2,max(bvp_total),"Pearson Corr"+str(corr))
		plt.savefig("/home/bhargav/MTTS-CAN/code/cohface/" + "overall" + ".png")
		plt.title("Cohface dataset")
		plt.close()
		plt.hist(hr_total)
		plt.show()
		plt.title("HR-hist")
		plt.hist(bvp_total)
		plt.show()
	elif dataset=="PURE":
		fs = 60
		hr_total = []
		bvp_total = []
		with open("/code/PURE.csv", "w") as csvfile:
			filewriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
			filewriter.writerow(["Name","bvp_average","Max","Min","STD","HR_calculate","Hr_measure"])
			for path in paths:
				pulse = []
				hr = []
				original_dataPath = path.removesuffix("_dataFile.hdf5") + ".json"
				print(original_dataPath)
				with open(original_dataPath) as json_file:
					json_data = json.load(json_file)
					for p in json_data['/FullPackage']:
						pulse.append(p['Value']['waveform'])
						hr.append(p['Value']['pulseRate'])
				pulse = np.array(pulse)
				hr = np.array(hr)
				N = 30 * fs
				pulse_fft = np.expand_dims(pulse, 0)
				f, pxx = signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
				fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
				frange = np.take(f, fmask)
				HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60
				BVP_mean = np.mean(pulse)
				BVP_max = np.max(pulse)
				BVP_min = np.min(pulse)
				BVP_std = np.std(pulse)
				hr_mean = np.mean(hr)
				hr_total.append(HR)
				bvp_total.append(BVP_mean)
				filewriter.writerow([original_dataPath.split("/")[-1].strip(".json"),BVP_mean,BVP_max,BVP_min,BVP_std, HR, hr_mean])
				# plt.scatter(hr,pulse)
				# plt.xlabel("HR")
				# plt.ylabel("BVP value")
				# plt.savefig("/home/bhargav/MTTS-CAN/code/correlation/PURE"+ f"{original_dataPath.split('/')[-1].removesuffix('.json')}"+".png")
				# plt.close()
				# f.writelines()
		_, p = normaltest(hr_total)
		print("HR",p)
		_, p = normaltest(bvp_total)
		print("BVP",p)
		plt.scatter(hr_total,bvp_total)
		corr,_ = pearsonr(hr_total, bvp_total)
		plt.xlabel("HR")
		plt.ylabel("BVP value")
		plt.text(max(hr_total)//2,max(bvp_total),"Pearson Corr"+str(corr))
		plt.savefig("/home/bhargav/MTTS-CAN/code/correlation/PURE/"+"overall"+".png")
		plt.close()
		plt.hist(hr_total)
		plt.show()
		plt.title("HR-hist")
		plt.hist(bvp_total)
		plt.title("BVP-hist")
		plt.show()

def main(dataset, data_dir):
	if dataset == "VIPL":
		pass
	elif dataset == "PURE":
		subTrain, subDev, subTest = split_subj_("/home/bhargav/PURE/", "PURE")
		path_of_video_tr = sort_dataFile_list_(data_dir, subTrain, dataset, trainMode=True)
		path_of_video_test = sort_dataFile_list_(data_dir, subTest, dataset, trainMode=True)
		path_of_video_val = sort_dataFile_list_(data_dir, subDev, dataset, trainMode=True)
		paths = path_of_video_val + path_of_video_test + path_of_video_tr
	elif dataset=="COHFACE":
		subTrain, subDev, subTest = split_subj_("/home/bhargav/cohface/", "COHFACE")
		path_of_video_tr = sort_dataFile_list_(data_dir, subTrain, dataset, trainMode=True)
		path_of_video_test = sort_dataFile_list_(data_dir, subTest, dataset, trainMode=True)
		path_of_video_val = sort_dataFile_list_(data_dir, subDev, dataset, trainMode=True)
		paths = path_of_video_val + path_of_video_test + path_of_video_tr
	bvp_hr(paths, dataset)





if __name__=="__main__":
	# dataset = "VIPL"
	dataset = "COHFACE"
	# dataset = "PURE"
	if dataset == "COHFACE":
		data_dir = "/home/bhargav/cohface/"
	elif dataset == "VIPL":
		data_dir = "/home/bhargav/vipl/hdf5/"
	elif dataset == "PURE":
		data_dir = "/home/bhargav/PURE/"
	main(dataset, data_dir)
