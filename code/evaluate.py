import argparse
import logging
import itertools

import json
import os

import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
from tensorflow import keras

from data_generator import DataGenerator
from model import HeartBeat, CAN, CAN_3D, Hybrid_CAN, TS_CAN, MTTS_CAN, \
    MT_Hybrid_CAN, MT_CAN_3D, MT_CAN
from pre_process import get_nframe_video, split_subj_, sort_dataFile_list_
from sklearn.metrics import mean_absolute_error

from inference_preprocess import preprocess_raw_video, detrend
from scipy.signal import butter
import h5py
import heartpy as hp
from sklearn.preprocessing import MinMaxScaler

def get_frame_sum(list_vid, maxLen_Video):
    frames_sum = 0
    counter = 0
    for vid in list_vid:
        hf = h5py.File(vid, 'r')
        shape = hf['data'].shape
        # if shape[0] > maxLen_Video:
        #   frames_sum += maxLen_Video
        # else: 
        frames_sum += shape[0]
        counter += 1
    return frames_sum

def prepare_MTTS_CAN(video, maxlen):
    frame_depth = 10
    dim_0 , dim_1 = (36,36)
    maxLen_Video = maxlen 
    sum_frames_batch = get_frame_sum(list_video_temp, maxLen_Video)
    data = np.zeros((sum_frames_batch, dim_0, dim_1, 6), dtype=np.float32)
    label_y = np.zeros((sum_frames_batch, 1), dtype=np.float32)
    num_window = int(sum_frames_batch/ frame_depth)
    index_counter = 0
    for index, temp_path in enumerate(list_video_temp):
        f1 = h5py.File(temp_path, 'r')
        dXsub = np.array(f1['data'])
        dysub = np.array(f1['pulse'])
        current_nframe = dXsub.shape[0]
        data[index_counter:index_counter+current_nframe, :, :, :] = dXsub
        label_y[index_counter:index_counter+current_nframe, 0] = dysub # data BVP
        index_counter += current_nframe
    motion_data = data[:, :, :, :3]
    apperance_data = data[:, :, :, -3:]
    max_data = num_window*frame_depth
    motion_data = motion_data[0:max_data, :, :, :]
    apperance_data = apperance_data[0:max_data, :, :, :]
    label_y = label_y[0:max_data, 0]
    label_r = label_r[0:max_data, 0]
    apperance_data = np.reshape(apperance_data, (num_window, frame_depth, dim_0, dim_1, 3))
    apperance_data = np.average(apperance_data, axis=1)
    apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], frame_depth, axis=1)
    apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
                                                         apperance_data.shape[2], apperance_data.shape[3],
                                                         apperance_data.shape[4]))
    output = (motion_data, apperance_data)
    label = label_y
    return output, label


def evaluate(model_path, test_set, model_name):
    mms = MinMaxScaler()
    model_ckpt = model_path
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    batch_size = 100
    model_ckpt = './mtts_can.hdf5'
    #cohface
    fs = 20 
    error = []
    for video_path in test_set:
        logging.info(f"Running on video {video_path}")
        print(video_path)
        sample_data_path = video_path
        
        #MTTS-CAN
        if model_name=="MTTS-CAN":
            dXsub = preprocess_raw_video(sample_data_path, dim=36)
            print('dXsub shape', dXsub.shape)

            dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
            dXsub = dXsub[:dXsub_len, :, :, :]

            model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
            model.load_weights(model_ckpt)

            yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

            pulse_pred = yptest[0]
            pulse_pred = detrend(np.cumsum(pulse_pred), 100)
            [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
            pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
            pulse_pred = np.array(mms.fit_transform(pulse_pred.reshape(-1,1))).flatten()

        #groud truth signal
        truth_path = video_path.replace(".avi","_dataFile.hdf5")
        gound_truth_file = h5py.File(truth_path, "r")
        pulse_truth = gound_truth_file["pulse"]   ### range ground truth from 0 to 1
        pulse_truth = pulse_truth[0:dXsub_len]
        pulse_truth = detrend(np.cumsum(pulse_truth), 100)
        [b_pulse_tr, a_pulse_tr] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_truth = scipy.signal.filtfilt(b_pulse_tr, a_pulse_tr, np.double(pulse_truth))
        pulse_truth = np.array(mms.fit_transform(pulse_truth.reshape(-1,1))).flatten()

        if len(pulse_pred) > len(pulse_truth):
            pulse_pred = pulse_pred[:len(pulse_truth)]
        elif len(pulse_pred) < len(pulse_truth):
            pulse_truth = pulse_truth[:len(pulse_pred)] 

        working_data_pred, measures_pred = hp.process_segmentwise(pulse_pred, sample_rate=fs, segment_width = 10, segment_overlap = 0.90)
        working_data_truth, measures_truth = hp.process_segmentwise(pulse_truth, sample_rate=fs, segment_width = 10, segment_overlap = 0.90)
        bpm_pred = np.array(measures_pred['bpm'])
        bpm_truth = np.array(measures_truth['bpm'])
        logging.info(f'{bpm_pred}')
        logging.info(f'Nan present {np.isnan(bpm_pred).any()}')
        logging.info(f'Nan present {np.isnan(bpm_truth).any()}')
        logging.info(f'{bpm_pred[np.isnan(bpm_truth)]}')
        logging.info(f'{bpm_truth[np.isnan(bpm_truth)]}')
        bpm_pred, bpm_truth = bpm_pred[~np.isnan(bpm_pred)], bpm_truth[~np.isnan(bpm_pred)] 
        bpm_pred, bpm_truth = bpm_pred[~np.isnan(bpm_truth)], bpm_truth[~np.isnan(bpm_truth)] 
        mae = mean_absolute_error(bpm_pred, bpm_truth)
        logging.info(f"MAE, {mae}")
        error.append(mae)
        gound_truth_file.close()

    return error

if __name__ == "__main__":

    logging.basicConfig(filename='Evalutation.log', level=logging.INFO)
    logging.info('Started')
    subTrain, subDev, subTest = split_subj_("/work/data/bacharya/cohface/", "COHFACE")
    path_of_video_tr = sort_dataFile_list_("/work/data/bacharya/cohface/", subTest, "COHFACE", trainMode=False)
    logging.info(f"list of all the videos f{subTest,path_of_video_tr}")
    errors = evaluate("model",path_of_video_tr ,"MTTS_CAN")
    logging.info(f"Mean error {np.mean(np.array(errors))}")
    np.save("./errors.npy", np.mean(np.array(errors)))
    logging.info(f"errors {errors}")
    logging.info('Finished')
