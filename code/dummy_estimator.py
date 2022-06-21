
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
from sklearn.metrics import mean_absolute_error

from inference_preprocess import preprocess_raw_video, detrend
from scipy.signal import butter
import h5py
import heartpy as hp
from sklearn.preprocessing import MinMaxScaler
from utils import BVPsignal 

def dummy_estimate(path_videos):
    """
    Return the dummy heartrate estiamte and mean bvp estimate
    """
    mms = MinMaxScaler()
    hr_estimates = []
    bvp_mean = []
    fs = 20.0
    for video_path in path_videos:
        logging.info(f"Running on video {video_path}")
        #path to the processed signal(MinMaxScaler)
        truth_path = video_path.replace(".avi","_dataFile.hdf5")
        #path to the original signal from the dataset
        truth_path_original = truth_path.replace("_dataFile.hdf5",".hdf5")
        print(truth_path_original)
        print(truth_path)
       
        gound_truth_file = h5py.File(truth_path, "r")
        pulse_truth = np.array(gound_truth_file["pulse"])   ### range ground truth from 0 to 1
        bvp_mean.append(np.mean(pulse_truth))
        gound_truth_file.close()

        gound_truth_file = h5py.File(truth_path_original, "r")
        bvp_signal = np.array(gound_truth_file['pulse'])
        print(bvp_signal.shape)
        bvp = BVPsignal(bvp_signal.reshape(1,len(bvp_signal)),256)
        gtBPM, timeGT = bvp.getBPM(winsize=10)
        #bpm_truth = bpm_truth[~np.isnan(bpm_truth)]
        gound_truth_file.close()
        hr_estimates.extend(gtBPM)
        logging.info(f'{gtBPM}')
    
    return np.mean(hr_estimates), np.mean(bvp_mean)

def evaluate(path_of_video, dummyEst,dummyBVP):
    error_HR = []
    error_BVP = []
    for video_path in path_of_video:

        logging.info(f"Running on video {video_path}")
        truth_path = video_path.replace(".avi","_dataFile.hdf5")
        truth_path_original = truth_path.replace("_dataFile.hdf5",".hdf5")
        print(truth_path_original)
        print(truth_path)
        gound_truth_file = h5py.File(truth_path_original, "r")
        bvp_signal = np.array(gound_truth_file['pulse'])
        gound_truth_file.close()

        bvp = BVPsignal(bvp_signal.reshape(1,len(bvp_signal)),256)
        gtBPM, timeGT = bvp.getBPM(winsize=10)

        #bpm_truth = bpm_truth[~np.isnan(bpm_truth)]
        bpm_pred = np.ones(gtBPM.shape)*dummyEst
        mae = mean_absolute_error(bpm_pred, gtBPM)
        logging.info(f"MAE, {mae}")
        error_HR.append(mae)
        gound_truth_file.close()

        gound_truth_file = h5py.File(truth_path, "r")
        pulse_truth = np.array(gound_truth_file["pulse"])   ### range ground truth from 0 to 1
        bpm_pred = np.ones(pulse_truth.shape)*dummyBVP
        mae_bvp = mean_absolute_error(pulse_truth, bpm_pred)
        error_BVP.append(mae_bvp)

    return error_HR, error_BVP
        
if __name__ == "__main__":

    logging.basicConfig(filename='Dummy-Evalutation.log', level=logging.INFO)
    logging.info('Started')
    subTrain, subDev, subTest = split_subj_("/work/data/bacharya/cohface/", "COHFACE")
    path_of_video_train = sort_dataFile_list_("/work/data/bacharya/cohface/", subTrain, "COHFACE", trainMode=True) 

    #Calculate the dummy estimate
    dummy_mean, bvp_mean = dummy_estimate(path_of_video_train)    
    print("dummy mean ", dummy_mean)
    logging.info(f"Dummy mean : {dummy_mean}")
    #test set mae evaluation
    path_of_video_dev = sort_dataFile_list_("/work/data/bacharya/cohface/", subDev, "COHFACE", trainMode=True)
    path_of_video_test = sort_dataFile_list_("/work/data/bacharya/cohface/", subTest, "COHFACE", trainMode=True)
    logging.info(f"list of all the videos f{subTest,path_of_video_dev}")
    logging.info(f"list of all the videos f{subTest,path_of_video_test}")
    errors_HR, errors_bvp = evaluate(path_of_video_dev,dummy_mean, bvp_mean)
    logging.info(f"******************************************************"")
    logging.info(f"DevSet")
    logging.info(f"******************************************************"")
    logging.info(f"Mean error {np.mean(np.array(errors_HR))}, Mean loss {np.mean(np.array(errors_bvp))}")
    np.save("./dev_dummy_errors.npy", np.mean(np.array(errors_HR)))
    np.save("./dev_dummy_errorsbvp.npy", np.mean(np.array(errors_bvp)))
    logging.info(f"errors {errors_HR}")

    errors_HR, errors_bvp = evaluate(path_of_video_dev,dummy_mean, bvp_mean)
    logging.info(f"Mean error {np.mean(np.array(errors_HR))}, Mean loss {np.mean(np.array(errors_bvp))}")
    np.save("./dummy_errors.npy", np.mean(np.array(errors_HR)))
    np.save("./dummy_errorsbvp.npy", np.mean(np.array(errors_bvp)))
    logging.info(f"errors {errors_HR, errors_bvp}")
    logging.info('Finished')
