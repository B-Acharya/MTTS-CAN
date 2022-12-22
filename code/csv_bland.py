import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np

def load_pickle(path):
	with open(path, "br") as f:
		data = pickle.load(f)
	return data

def plot_bland(bpm_gt,bpm,dataset, model, savePath ):
	hr = np.concatenate((np.array(bpm_gt).reshape(-1, 1), bpm.reshape(-1, 1)), axis=1)
	averages_NR = np.mean(hr, axis=1)
	diff_NR = np.diff(hr, axis=1)
	diff_NR_mean, diff_NR_std = np.mean(diff_NR), np.std(diff_NR)
	upper_limit_NR = diff_NR_mean + 1.96 * diff_NR_std
	lower_limit_NR = diff_NR_mean - 1.96 * diff_NR_std

	x_value = np.max(averages_NR)

	plt.scatter(averages_NR, diff_NR)
	plt.hlines(upper_limit_NR, min(averages_NR), max(averages_NR), colors="red", linestyle="dashed", label="+1.96SD")
	plt.hlines(lower_limit_NR, min(averages_NR), max(averages_NR), colors="red", linestyle="dashed", label="-1.96SD")
	plt.hlines(diff_NR_mean, min(averages_NR), max(averages_NR), colors="Blue", linestyle="solid", label="Mean")
	plt.text(x_value, upper_limit_NR + 1, "+1.96SD")
	plt.text(x_value, upper_limit_NR - 1, f"{upper_limit_NR:.2f}")
	plt.text(x_value, lower_limit_NR + 1, "+1.96SD")
	plt.text(x_value, lower_limit_NR - 1, f"{lower_limit_NR:.2f}")
	plt.text(x_value, diff_NR_mean + 1, "Mean")
	plt.text(x_value, diff_NR_mean - 1, f"{diff_NR_mean:.2f}")
	plt.title(dataset+"-"+model)
	plt.xlabel("Average of the estimated HR and Ground truth")
	plt.ylabel("Difference between estimated HR and ground truth HR")
	plt.savefig(savePath + model + "bland.png")
	plt.show()

def read_csv(path):
	df = pd.read_excel(path,engine='openpyxl')
	df = df.drop(0)
	bpm = df['HR_predicted'].values
	if any(np.isnan(bpm)):
		print("Dropping a row, please check the data there is a row with NaN's")
		bpm = bpm[~np.isnan(bpm)]

	bpm_GT = df['HR_GT'].values
	if any(np.isnan(bpm_GT)):
		print("Dropping a row, please check the data there is a row with NaN's")
		bpm_GT = bpm_GT[~np.isnan(bpm_GT)]

	return bpm, bpm_GT


def plot_scatter(bpm, bpm_gt, dataset,model,savePath):
	min_bpm = min(bpm)
	max_bpm = max(bpm)
	plt.scatter(bpm,bpm_gt)
	plt.plot([min_bpm,max_bpm],[min_bpm,max_bpm],"r")
	plt.xlabel("HR from Video")
	plt.ylabel("HR ground Truth")
	plt.title(dataset+"-"+model)
	plt.savefig(savePath+model+".png")
	plt.show()

def main(path, model, dataset, savePath):
	if dataset == "PPG":
		sheets = []
		df_sheets = pd.ExcelFile(path)
		for sheet in df_sheets.sheet_names:
			sheets.append(df_sheets.parse(sheet))

		for i, sheet in enumerate(sheets):
			if i == 0:
				columns = sheet.columns
			else:
				sheet.columns = columns
				sheet = sheet.drop(0)
				bpm = sheet['HR_predicted'].values
				if any(np.isnan(bpm)):
					print("Dropping a row, please check the data there is a row with NaN's")
					bpm = bpm[~np.isnan(bpm)]

				bpm_GT = sheet['HR_GT'].values
				if any(np.isnan(bpm_GT)):
					print("Dropping a row, please check the data there is a row with NaN's")
					bpm_GT = bpm_GT[~np.isnan(bpm_GT)]
				model = sheet["Model"][1] + sheet["Trained ON "][1]
				plot_scatter(bpm, bpm_GT, dataset, model,savePath)
				plot_bland(bpm, bpm_GT, dataset, model, savePath)
				mae = mean_absolute_error(bpm,bpm_GT)
				rmse = mean_squared_error(bpm,bpm_GT,squared=False)
				print(model, "Mae:",mae, "RMSE:",rmse)
	else:
		bpm, bpm_gt = read_csv(path)
		plot_scatter(bpm, bpm_gt, dataset, model, savePath)
		plot_bland(bpm, bpm_gt, dataset, model, savePath)
		mae = mean_absolute_error(bpm,bpm_gt)
		rmse = mean_squared_error(bpm,bpm_gt,squared=False)

	return mae, rmse


if __name__=="__main__":
	# dataset = "VIPL"
	# dataset = "PURE"
	# dataset = "cohface"
	# dataset = "PUREVIPL"
	dataset = "PPG"
	if dataset == "VIPL":
		paths = [("/home/bhargav/vipl/predictions/Presentation/CAN_3D.xlsx","3D"),
			 ("/home/bhargav/vipl/predictions/Presentation/TS_CAN.xlsx","TS"),
			 ("/home/bhargav/vipl/predictions/Presentation/Hybrid_CAN.xlsx","HY")]
		# paths = [("/home/bhargav/vipl/predictions/PURE-VIPL-CROSS//Hybrid_CAN.xlsx", "HY")]
		savePath = "/home/bhargav/vipl/predictions/Presentation/"
	elif dataset == "PURE":
		paths = [("/home/bhargav/PURE/predictions/Presentation/CAN_3D.xlsx","3D"),
				 ("/home/bhargav/PURE/predictions/Presentation/TS_CAN.xlsx","TS"),
				 ("/home/bhargav/PURE/predictions/Presentation/Hybrid_CAN.xlsx","HY")]
		savePath = "/home/bhargav/PURE/predictions/Presentation/"
	elif dataset == "cohface":
		paths = [("/home/bhargav/cohface_dataset/predictions/Presentation/CAN_3D.xlsx","3D"),
				 ("/home/bhargav/cohface_dataset/predictions/Presentation/TS_CAN.xlsx","TS"),
				 ("/home/bhargav/cohface_dataset/predictions/Presentation/Hybrid_CAN.xlsx","HY")]
		savePath = "/home/bhargav/cohface_dataset/predictions/Presentation/"

	elif dataset == "PUREVIPL":
		paths = [("/home/bhargav/vipl/predictionsPURE-VIPL-CROSS/Hybrid_CAN.xlsx", "HY")]

	elif dataset == "PPG":
		paths = [("/home/bhargav/processed/collected-dataset/finalEvaluation.xlsx", "all")]
		savePath = "/home/bhargav/processed/collected-dataset/"

	for path, model in paths:
		print(path.split("/")[-1].removesuffix(".xlsx"))
		mae , rmse = main(path,model,dataset, savePath)
		print("Mae:",mae, "RMSE:",rmse)