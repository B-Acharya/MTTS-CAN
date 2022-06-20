
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

def dummy_estimate(path_videos):
    mms = MinMaxScaler()
    hr_estimates = []
    fs = 20.0
    for video_path in path_videos:
        logging.info(f"Running on video {video_path}")
        truth_path = video_path.replace(".avi","_dataFile.hdf5")
        gound_truth_file = h5py.File(truth_path, "r")
        pulse_truth = gound_truth_file["pulse"]   ### range ground truth from 0 to 1
        pulse_truth = detrend(np.cumsum(pulse_truth), 100)
        [b_pulse_tr, a_pulse_tr] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_truth = scipy.signal.filtfilt(b_pulse_tr, a_pulse_tr, np.double(pulse_truth))
        pulse_truth = np.array(mms.fit_transform(pulse_truth.reshape(-1,1))).flatten()
        working_data_truth, measures_truth = hp.process_segmentwise(pulse_truth, sample_rate=fs, segment_width = 10, segment_overlap = 0.90)
        bpm_truth = np.array(measures_truth['bpm'])
        bpm_truth = bpm_truth[~np.isnan(bpm_truth)]
        hr_estimates.extend(bpm_truth)
        logging.info(f'{bpm_truth}')
    
    return np.mean(hr_estimates)
def evaluate(path_of_video, dummyEst):
    mms = MinMaxScaler()
    fs = 20.0
    error = []
    for video_path in path_of_video:
        logging.info(f"Running on video {video_path}")

        truth_path = video_path.replace(".avi","_dataFile.hdf5")
        gound_truth_file = h5py.File(truth_path, "r")
        pulse_truth = gound_truth_file["pulse"]   ### range ground truth from 0 to 1
        pulse_truth = detrend(np.cumsum(pulse_truth), 100)
        [b_pulse_tr, a_pulse_tr] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_truth = scipy.signal.filtfilt(b_pulse_tr, a_pulse_tr, np.double(pulse_truth))
        pulse_truth = np.array(mms.fit_transform(pulse_truth.reshape(-1,1))).flatten()

        working_data_truth, measures_truth = hp.process_segmentwise(pulse_truth, sample_rate=fs, segment_width = 10, segment_overlap = 0.90)
        bpm_truth = np.array(measures_truth['bpm'])
        bpm_truth = bpm_truth[~np.isnan(bpm_truth)]
        bpm_pred = np.ones(bpm_truth.shape)*dummyEst
        mae = mean_absolute_error(bpm_pred, bpm_truth)
        logging.info(f"MAE, {mae}")
        error.append(mae)
    return error
        
if __name__ == "__main__":

    logging.basicConfig(filename='Dummy-Evalutation.log', level=logging.INFO)
    logging.info('Started')
    subTrain, subDev, subTest = split_subj_("/work/data/bacharya/cohface/", "COHFACE")
    path_of_video_train = sort_dataFile_list_("/work/data/bacharya/cohface/", subTrain, "COHFACE", trainMode=True) 

    #Calculate the dummy estimate
    dummy_mean = dummy_estimate(path_of_video_train)    
    print("dummy mean ", dummy_mean)
    logging.info(f"Dummy mean : {dummy_mean}")
    #test set mae evaluation
    path_of_video_tr = sort_dataFile_list_("/work/data/bacharya/cohface/", subTest, "COHFACE", trainMode=True)
    logging.info(f"list of all the videos f{subTest,path_of_video_tr}")
    errors = evaluate(path_of_video_tr,dummy_mean)
    logging.info(f"Mean error {np.mean(np.array(errors))}")
    np.save("./dummy_errors.npy", np.mean(np.array(errors)))
    logging.info(f"errors {errors}")
    logging.info('Finished')
