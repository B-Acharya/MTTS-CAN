import argparse
import logging
import itertools

import json
import os

import numpy as np
import scipy.io
import tensorflow as tf

from data_generator import DataGenerator
from pre_process import get_nframe_video, split_subj_, sort_dataFile_list_
from sklearn.metrics import mean_absolute_error, mean_squared_error

from inference_preprocess import preprocess_raw_video, detrend
from scipy.signal import butter
import h5py
from sklearn.preprocessing import MinMaxScaler
from utils import BVPsignal
from scipy import signal
from pathlib import Path

def dummy_estimate(path_videos, dataset):
    """
    Return the dummy heartrate estiamte and mean bvp estimate
    """
    hr_estimates = []
    bvp_mean = []
    if dataset == "PURE":
        fs = 30
    elif dataset == "COHFACE":
        fs = 20.0
    elif dataset == "VIPL":
        fs = 20.0
    elif dataset == "PPG":
        fs = 1000.0
    for video_path in path_videos:
        logging.info(f"Running on video {video_path}")
        if dataset=="COHFACE":
            truth_path = video_path.replace(".avi","_dataFile.hdf5")
        elif dataset=="PURE":
            truth_path = video_path
        elif dataset == "VIPL":
            truth_path = video_path
        elif dataset == "PPG":
            truth_path = video_path

        gound_truth_file = h5py.File(truth_path, "r")
        if dataset == "PPG":
            pulse_truth = np.array(gound_truth_file["bvp"])   ### range ground truth from 0 to 1
        else:
            pulse_truth = np.array(gound_truth_file["pulse"])   ### range ground truth from 0 to 1
        bvp_mean.append(np.mean(pulse_truth))
        gound_truth_file.close()

        # pulse_truth = detrend(np.cumsum(pulse_truth), 100)
        # [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        # pulse_truth = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_truth))

        #calculation of HR from the entire video
        N = 30 * fs
        pulse_fft = np.expand_dims(pulse_truth, 0)
        f, pxx = signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
        fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        frange = np.take(f, fmask)
        HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60
        logging.info(f"HR:{HR}")
        hr_estimates.append(HR)

    return np.mean(hr_estimates), np.mean(bvp_mean)
    #added for PPG dataset
    # return np.mean(hr_estimates), 0

def evaluate(path_of_video, dummyEst,dummyBVP, dataset):
    HR_estimates = []
    BVP_estimates = []
    for video_path in path_of_video:
        logging.info(f"Running on video {video_path}")
        if dataset=="COHFACE":
            truth_path = video_path.replace(".avi","_dataFile.hdf5")
            # truth_path_original = truth_path.replace("_dataFile.hdf5",".hdf5")
            truth_path_original = video_path
            fs = 20
        elif dataset=="PURE":
            truth_path = video_path
            truth_path_original = video_path
            fs = 30
        elif dataset == "VIPL":
            truth_path = video_path
            truth_path_original = video_path
            fs = 30
        elif dataset == "PPG":
            truth_path = video_path
            truth_path_original = video_path
            fs = 1000

        gound_truth_file = h5py.File(truth_path_original, "r")
        if dataset == "PPG":
            pulse_truth = np.array(gound_truth_file["bvp"])   ### range ground truth from 0 to 1
        else:
            pulse_truth = np.array(gound_truth_file["pulse"])   ### range ground truth from 0 to 1
        gound_truth_file.close()

        #calculation of HR from the entire video
        N = 30 * fs
        pulse_fft = np.expand_dims(pulse_truth, 0)
        f, pxx = signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
        fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        frange = np.take(f, fmask)
        HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60

        logging.info("HR:", HR)
        HR_estimates.append(HR)

        gound_truth_file = h5py.File(truth_path, "r")
        if dataset == "PPG":
            pulse_truth = np.array(gound_truth_file["bvp"])   ### range ground truth from 0 to 1
        else:
            pulse_truth = np.array(gound_truth_file["pulse"])   ### range ground truth from 0 to 1
        bpm_pred = np.ones(pulse_truth.shape)*dummyBVP
        mae_bvp = mean_absolute_error(pulse_truth, bpm_pred)
        BVP_estimates.append(mae_bvp)

    gtBPM = np.ones(len(HR_estimates))*dummyEst
    mae = mean_absolute_error(HR_estimates, gtBPM)
    rmse = mean_squared_error(HR_estimates, gtBPM, squared=False)
    logging.info(f"MAE, {mae}")
    logging.info(f"MAE bvp, {mae_bvp}")
    return mae, rmse, BVP_estimates

if __name__ == "__main__":

    # dataset = "PURE"
    dataset = "PPG"
    # dataset = "COHFACE"
    if dataset == "PURE":
        logging.basicConfig(filename='Dummy-Evalutation-pure.log', level=logging.INFO)
        logging.info('Started')
        datadir = "/home/bhargav/PURE/"
        split_path = "/home/bhargav/PURE/"
    elif dataset == "VIPL":
        logging.basicConfig(filename='Dummy-Evalutation-vipl.log', level=logging.INFO)
        logging.info('Started')
        datadir = "/home/bhargav/vipl/hdf5/"
        split_path = "/home/bhargav/vipl/data/"
    elif dataset == "COHFACE":
        logging.basicConfig(filename='Dummy-Evalutation-cohface.log', level=logging.INFO)
        logging.info('Started')
        datadir = "/home/bhargav/cohface/"
        split_path = "/home/bhargav/cohface/"
    elif dataset == "PPG":
        logging.basicConfig(filename='Dummy-Evalutation-PPG.log', level=logging.INFO)
        logging.info('Started')
        datadir = "/home/bhargav/processed"
        split_path = "/home/bhargav/processed/"

    if dataset != "PPG":
        subTrain, subDev, subTest = split_subj_(split_path, dataset)

        path_of_video_train = sort_dataFile_list_(datadir, subTrain, dataset, trainMode=True)
        path_of_video_dev = sort_dataFile_list_(datadir, subDev, dataset, trainMode=True)
        path_of_video_test = sort_dataFile_list_(datadir, subTest, dataset, trainMode=True)
        logging.info(f"list of all the videos f{subTest,path_of_video_dev}")
        logging.info(f"list of all the videos f{subTest,path_of_video_test}")

    else:
        path = Path("/home/bhargav/processed/")
        path_of_video_train = list(map(str,path.rglob("*.hdf5")))
        path_of_video_test  = list(map(str,path.rglob("*.hdf5")))
        logging.info(f"list of all the videos f{path_of_video_train}")
        logging.info(f"list of all the videos f{path_of_video_test}")

    #Calculate the dummy estimate
    dummy_mean, bvp_mean = dummy_estimate(path_of_video_train, dataset)
    print("dummy mean ", dummy_mean)
    logging.info(f"Dummy mean : {dummy_mean}")
    #test set mae evaluation
    if dataset!="PPG":
        errors_HR, rmse_HR, errors_bvp = evaluate(path_of_video_dev,dummy_mean, bvp_mean, dataset)
        logging.info(f"******************************************************")
        logging.info(f"DevSet")
        logging.info(f"******************************************************")
        logging.info(f"Mean error {errors_HR}, Mean loss {np.mean(np.array(errors_bvp))}")
        logging.info(f"RMSE error {rmse_HR}")
        np.save("./dev_dummy_errors.npy", errors_HR)
        np.save("./dev_dummy_errorsbvp.npy", np.mean(np.array(errors_bvp)))
        logging.info(f"errors {errors_HR}")

    logging.info(f"******************************************************")
    logging.info(f"TestSet")
    logging.info(f"******************************************************")
    errors_HR, rmse_HR, errors_bvp = evaluate(path_of_video_test,dummy_mean, bvp_mean,dataset)
    logging.info(f"Mean error {errors_HR}, Mean loss {np.mean(np.array(errors_bvp))}")
    logging.info(f"RMSE error {rmse_HR}")
    np.save("./dummy_errors.npy", errors_HR)
    np.save("./dummy_errorsbvp.npy", np.mean(np.array(errors_bvp)))
    logging.info(f"errors {errors_HR, errors_bvp}")
    logging.info('Finished')
