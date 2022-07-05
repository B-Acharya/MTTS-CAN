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
    #        for t in taskList:
            x = glob.glob(os.path.join(data_dir, str(p), 'data.avi'))
            x = sorted(x)
            #x = sorted(x, key=take_last_ele)
            final.append(x)
    if database_name == "PURE":
        for p in subTrain:
            x = data_dir+"-".join(p.split("/")[:2])
            #x = sorted(x)
            # x = sorted(x, key=take_last_ele)
            print(x)
            final.append([x])
    else:
        print("not implemented yet.")
    return final

def get_video_list(path, database):
    participants = []
    if database=="COHFACE":
        with open(path,"r") as f:
            for line in f.readlines():
                participants.append(line.strip()[:-4])
    elif database=="PURE":
        with open(path, "r") as f:
            for line in f.readlines():
                participants.append(line.strip()[:-4])
    elif database=="VIPL":
        print(path, path)
        with open(path, "r") as f:
            for line in f.readlines():
                participants.append("p"+"-".join(line.strip().split("/")))
    return participants

def split_subj_(data_dir, database): # trennen der Daten innerhalb 1 Subjekts...
    if database == "COHFACE":
        #subTrain = np.array(range(1, 33)).tolist()# 33)).tolist()
        #subTrain = get_video_list(data_dir+"protocols/split/train.txt", database)
        #subDev = get_video_list(data_dir+"protocols/split/dev.txt", database)
        #subTest = get_video_list(data_dir+"protocols/split/test.txt", database)
        subTrain = get_video_list(data_dir+"protocols/all/train.txt", database)
        subDev = get_video_list(data_dir+"protocols/all/dev.txt", database)
        subTest = get_video_list(data_dir+"protocols/all/test.txt", database)
    elif database == "PURE":
        subTrain = get_video_list(data_dir+"train.csv", database)
        subDev = get_video_list(data_dir+"dev.csv", database)
        subTest = get_video_list(data_dir+ "test.csv", database)
    elif database =="VIPL":
        subTrain = get_video_list(data_dir+"train.csv", database)
        subDev = get_video_list(data_dir+"dev.csv", database)
        subTest = get_video_list(data_dir+ "test.csv", database)
    else:
        print("This Database isn't implemented yet.")
    return subTrain, subDev, subTest

def dataFile_COHFACE(data_dir, subTrain, train):
    final = []
    if train:
        for p in subTrain:
            #x = glob.glob(os.path.join(data_dir, str(p),  '*dataFile.hdf5'))
            x = glob.glob(os.path.join(data_dir, str(p),  '*dataFile_NONR.hdf5'))
            x = sorted(x)
            #x = sorted(x, key=take_last_ele)
            final.append(x)
    else:
         for p in subTrain:
             x = glob.glob(os.path.join(data_dir, str(p),  '*.avi'))
             x = sorted(x)
             #x = sorted(x, key=take_last_ele)
             final.append(x)
    return final

def dataFile_PURE(data_dir, subTrain, train):
    final = []
    if train:
        for p in subTrain:
            x = glob.glob(os.path.join(data_dir, "-".join(str(p)[:-1].split("/"))+'_dataFile.hdf5'))
            x = sorted(x)
            #x = sorted(x, key=take_last_ele)
            final.append(x)
    else:
         for p in subTrain:
             x = glob.glob(os.path.join(data_dir, str(p),  '*.avi'))
             x = sorted(x)
             #x = sorted(x, key=take_last_ele)
             final.append(x)
    return final

def dataFile_VIPL(data_dir, subTrain, train):
    final = []
    if train:
        for p in subTrain:
            print(os.path.join(data_dir, str(p)+'.hdf5'))
            x = glob.glob(os.path.join(data_dir, str(p)+'.hdf5'))
            x = sorted(x)
            print("dataFile_VIPL",x)
            #x = sorted(x, key=take_last_ele)
            final.append(x)
    else:
         for p in subTrain:
             x = glob.glob(os.path.join(data_dir, str(p),  '*.avi'))
             x = sorted(x)
             #x = sorted(x, key=take_last_ele)
             final.append(x)
    return final

def sort_dataFile_list_(data_dir, subTrain, database_name, trainMode):
    if database_name == "COHFACE":
        final = dataFile_COHFACE(data_dir, subTrain, trainMode)
        final = list(itertools.chain(*final))
    elif database_name =="PURE":
        final = dataFile_PURE(data_dir, subTrain, trainMode) 
        final = list(itertools.chain(*final))
    elif database_name == "VIPL":
        final = dataFile_VIPL(data_dir, subTrain, trainMode) 
        final = list(itertools.chain(*final))
        
    else: 
        print("not implemented yet.")
    return final
