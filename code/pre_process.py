import glob
import os

import itertools
import h5py
import numpy as np
import scipy.io


def get_nframe_video(path):
    temp_f1 = h5py.File(path, 'r')
    temp_dysub = np.array(temp_f1["data"])
    nframe_per_video = temp_dysub.shape[0]
    return nframe_per_video


def get_nframe_video_val(path):
    temp_f1 = scipy.io.loadmat(path)
    temp_dXsub = np.array(temp_f1["dXsub"])
    nframe_per_video = temp_dXsub.shape[0]
    return nframe_per_video


def split_subj(data_dir, cv_split, subNum):
    f3 = h5py.File(data_dir + '/M.mat', 'r')
    M = np.transpose(np.array(f3["M"])).astype(np.bool)
    subTrain = subNum[~M[:, cv_split]].tolist()
    subTest = subNum[M[:, cv_split]].tolist()
    return subTrain, subTest


def take_last_ele(ele):
    ele = ele.split('.')[0][-2:]
    try:
        return int(ele[-2:])
    except ValueError:
        return int(ele[-1:])


def sort_video_list(data_dir, taskList, subTrain):
    final = []
    for p in subTrain:
        for t in taskList:
            x = glob.glob(os.path.join(data_dir, 'P' + str(p) + 'T' + str(t) + 'VideoB2*.mat'))
            x = sorted(x)
            x = sorted(x, key=take_last_ele)
            final.append(x)
    return final

def sort_video_list_(data_dir, taskList, subTrain, database_name):
    final = []
    if database_name == "COHFACE":
            for p in subTrain:
                for t in taskList:
                    x = glob.glob(os.path.join(data_dir, str(p), str(t), 'data.avi'))
                    x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                    final.append(x)
    else: 
        print("not implemented yet.")
    return final

def get_video_list(path):
    participants = []
    with open(path,"r") as f:
        for line in f.readlines():
            participants.append(line.split("/")[0])
    return participants 

def split_subj_(data_dir, database): # trennen der Daten innerhalb 1 Subjekts...
    if database == "COHFACE":
        #subTrain = np.array(range(1, 33)).tolist()# 33)).tolist()
        subTrain = get_video_list(data_dir+"protocols/split/train.txt") 
        subDev = get_video_list(data_dir+"protocols/split/dev.txt") 
        subTest = get_video_list(data_dir+"protocols/split/test.txt") 
    else:
        print("This Database isn't implemented yet.")
    return subTrain, subDev, subTest

def dataFile_COHFACE(data_dir, taskList, subTrain, train):
    final = []
    if train:
        for p in subTrain:
            for t in taskList:
                x = glob.glob(os.path.join(data_dir, str(p), str(t), '*dataFile.hdf5'))
                x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                final.append(x)
    else:
         for p in subTrain:
            for t in taskList:
                x = glob.glob(os.path.join(data_dir, str(p), str(t), '*data.avi'))
                x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                final.append(x)
    return final

def sort_dataFile_list_(data_dir, subTrain, database_name, trainMode):
    if database_name == "COHFACE":
        taskList = [0, 1, 2, 3]
        final = dataFile_COHFACE(data_dir, taskList, subTrain, trainMode)
        final = list(itertools.chain(*final))
    #TODO add pure
    else: 
        print("not implemented yet.")
    return final
