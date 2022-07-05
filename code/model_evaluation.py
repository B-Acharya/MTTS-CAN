from aifc import Error
import numpy as np
import scipy.io
import xlsxwriter
from model import CAN, CAN_3D,TS_CAN, Hybrid_CAN,MTTS_CAN
import h5py
import os
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video_, detrend
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import scipy.stats as sc
from glob import glob
from scipy import signal
from pre_process import get_nframe_video, split_subj_, sort_video_list_, sort_dataFile_list_
from utils import BVPsignal , RMSEerror, MAEerror
import pickle


import heartpy as hp

'''Code adapted from https://raw.githubusercontent.com/KISMED-TUDa/rPPG-CANs/main/code/final_evaluation.py'''

def write_header(worksheet):
    header = ['Database','p', 'MAE', 'RMSE']
    for index in range(len(header)):
        worksheet.write(0,index, header[index])

def prepare_3D_CAN(dXsub):
    frame_depth = 10
    num_window = int(dXsub.shape[0]) - frame_depth + 1
    tempX = np.array([dXsub[f:f + frame_depth, :, :, :] # (491, 10, 36, 36 ,6) (169, 10, 36, 36, 6)
                    for f in range(num_window)])
    tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
    tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
    return tempX

def prepare_Hybrid_CAN(dXsub):
    frame_depth = 10
    num_window = int(dXsub.shape[0]) - frame_depth + 1
    tempX = np.array([dXsub[f:f + frame_depth, :, :, :] # (169, 10, 36, 36, 6)
                        for f in range(num_window)])
    tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
    tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
    motion_data = tempX[:, :, :, :, :3]
    apperance_data = np.average(tempX[:, :, :, :, -3:], axis=-2)
    return motion_data, apperance_data

def predict_vitals(worksheet, test_name, model_name, video_path, path_results, model_checkpoint_path,database_name):
    mms = MinMaxScaler()
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    batch_size = 100

    #### initialize Model ######
    model_checkpoint = model_checkpoint_path 
    
    if model_name == "TS_CAN":
        model = TS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    elif model_name == "CAN_3D":
        model = CAN_3D(frame_depth, 32, 64, (img_rows, img_cols, frame_depth, 3))
    elif model_name == "CAN":
        model = CAN(32, 64, (img_rows, img_cols, 3))
    elif model_name == "Hybrid_CAN":
        model = Hybrid_CAN(frame_depth, 32, 64, (img_rows, img_cols, frame_depth, 3),
                            (img_rows, img_cols, 3))
    elif model_name == "MTTS_CAN":
        model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    else: 
        raise NotImplementedError

    model.load_weights(model_checkpoint)
    ###### load video Data #######
    counter_video = 1
    old_database = "COH"
    list_bpm = {} 
    for sample_data_path in video_path:
        print("path:  ",sample_data_path)
        if database_name=="COHFACE":
            if sample_data_path[-4:] == ".avi":
                print("processing data")
                dXsub, fs = preprocess_raw_video_(sample_data_path, dim=36)
                print('dXsub shape', dXsub.shape, "fs: ", fs)
        
            dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
            dXsub = dXsub[:dXsub_len, :, :, :]
        elif database_name=="PURE":
            data = h5py.File(sample_data_path, "r")
            dXsub = np.array(data["data"])
            dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
            dXsub = dXsub[:dXsub_len, :, :, :]
            fs = 30
        else:
            print("Database not implemented")

        
        if model_name == "CAN_3D":
            dXsub = prepare_3D_CAN(dXsub)
            dXsub_len = (dXsub.shape[0] // (frame_depth))  * (frame_depth)
            dXsub = dXsub[:dXsub_len, :, :, :,:]
            yptest = model.predict((dXsub[:, :, :,: , :3], dXsub[:, :, :, : , -3:]), verbose=1)
        elif model_name == "Hybrid_CAN":
            dXsub1, dXsub2 = prepare_Hybrid_CAN(dXsub)
            dXsub_len1 = (dXsub1.shape[0] // (frame_depth*10))  * (frame_depth*10)
            dXsub1 = dXsub1[:dXsub_len1, :, :, :, :]
            dXsub_len2 = (dXsub2.shape[0] // (frame_depth*10))  * (frame_depth*10)
            dXsub2 = dXsub2[:dXsub_len2, :, :, :]
            yptest = model.predict((dXsub1, dXsub2), verbose=1)
        elif model_name == "MTTS_CAN":
            dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
            dXsub = dXsub[:dXsub_len, :, :, :]

            yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

        else:
            yptest = model((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), training=False)
        
        if model_name == "CAN_3D" or model_name == "Hybrid_CAN":
            pulse_pred = yptest[:,0]
        elif model_name == "MTTS_CAN":
            pulse_pred = yptest[0]
            print(pulse_pred.shape)
            
        elif model_name != "PTS_CAN" and model_name != "PPTS_CAN":
            pulse_pred = yptest
            
        else:
            pulse_pred = yptest[0]
        
        pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        [b_pulse_pred, a_pulse_pred] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_pred = scipy.signal.filtfilt(b_pulse_pred, a_pulse_pred, np.double(pulse_pred))
        pulse_pred = np.array(mms.fit_transform(pulse_pred.reshape(-1,1))).flatten()
        
        ##### ground truth data resampled  #######
        if(database_name == "COHFACE"):
            truth_path = sample_data_path.replace(".avi", ".hdf5")   # akutell für COHACE...
            truth_path_data = truth_path.replace(".hdf5", "_dataFile.hdf5")   # akutell für COHACE...
        elif database_name == "PURE":
            truth_path = sample_data_path
            truth_path_data = sample_data_path
        else:
            return print("Error in finding the ground truth signal...")
        data = h5py.File(truth_path,"r")
        pulse_truth = np.array(data["pulse"])
        data.close()
        ########### BPM ###########
        bvp = BVPsignal(pulse_pred.reshape(1,len(pulse_pred)),fs)
        BPM, times = bvp.getBPM(winsize=10)
        if database_name == "COHFACE":
            bvpGT = BVPsignal(pulse_truth.reshape(1,len(pulse_truth)),256)
            BPMGT, timesGT = bvpGT.getBPM(winsize=10)
        elif database_name =="PURE":
            bvpGT = BVPsignal(pulse_truth.reshape(1, len(pulse_truth)), 30)
            BPMGT, timesGT = bvpGT.getBPM(winsize=10)
        print(times, timesGT)
        ######## name files #############
        if(database_name=="COHFACE"):
            nameStr = str(sample_data_path).replace("data.avi", "")
            print(nameStr)
        elif database_name=="PURE":
            nameStr = str(sample_data_path).replace("_dataFile.hdf5", "")
        elif(str(sample_data_path).find("UBFC-PHYS") > 0):
            nmr = str(sample_data_path).find("UBFC-PHYS")
            nameStr = str(sample_data_path)[nmr + 12:].replace("\\", "-").replace("vid_", "").replace(".avi", "")
        elif(str(sample_data_path).find("UBFC") > 0):
            nmr = str(sample_data_path).find("UBFC")
            nameStr = str(sample_data_path)[nmr + 5:].replace("\\", "-").replace("vid.avi", "")
        elif(str(sample_data_path).find("BP4D") > 0):
            nmr = str(sample_data_path).find("BP4D")
            nameStr = str(sample_data_path)[nmr + 5:].replace("\\", "-")
        else:
            raise ValueError
        ########## Plot ##################

        ######### Metrics ##############
        # MSE:
        MAE = MAEerror(BPM.reshape(1,-1), BPMGT, timesES=times, timesGT=timesGT)
        # RMSE:
        RMSE = RMSEerror(BPM.reshape(1,-1), BPMGT, timesES=times, timesGT=timesGT)
        # Pearson correlation:
        data = h5py.File(truth_path_data,"r")
        pulse_truth_resampled = np.array(data["pulse"])
        data.close()
        if len(pulse_pred) > len(pulse_truth_resampled):
            pulse_pred = pulse_pred[:len(pulse_truth_resampled)]
        elif len(pulse_pred) < len(pulse_truth_resampled):
            pulse_truth_resampled = pulse_truth_resampled[:len(pulse_pred)]
        p = sc.pearsonr(pulse_truth_resampled, pulse_pred)

        ####### Logging #############
        if database_name != old_database:
            counter_video += 1
        worksheet.write(counter_video,0, database_name)
        worksheet.write(counter_video,1, nameStr)
        worksheet.write(counter_video,4, p[0])
        worksheet.write(counter_video,5, MAE)
        worksheet.write(counter_video,6, RMSE)
        list_bpm[sample_data_path] = [BPM, times, BPMGT, timesGT]
        counter_video += 1
        old_database = database_name
    with open(f"{database_name}_{model_name}.pkl", "wb") as output:
        pickle.dump(list_bpm, output) 
   
if __name__ == "__main__":
    #database_name = "COHFACE"
    database_name = "PURE"
    if database_name=="PURE":
        path_results = "/home/bacharya/PURE/predictions/"
        model_checkpoint = "/home/bacharya/PURE/checkpoints/short-PURE-TS_CAN-4-19/cv_0_epoch02_model.tf"
        data_dir = '/home/bacharya/PURE/'
        subTrain, subDev, subTest = split_subj_(data_dir, "PURE")
        path_of_video_test = sort_dataFile_list_(data_dir, subTest, "PURE", trainMode=True)
        save_dir = '/home/bacharya/PURE/predictions/'
        test_name = "PURE-old-split"
    elif database_name=="COHFACE":
        path_results = "/home/bacharya/cohface/predictions/"
        model_checkpoint = "/home/bacharya/cohface/checkpoints/short-cohface-CAN_3D-4-39/cv_0_epoch01_model.tf"
        data_dir = '/work/data/bacharya/cohface/'
        subTrain, subDev, subTest = split_subj_(data_dir, "COHFACE")
        path_of_video_test = sort_dataFile_list_(data_dir, subTest, "COHFACE", trainMode=False)
        save_dir = '/home/bacharya/cohface/predictions/'
        test_name = "cohface-old-split"
    print(path_of_video_test)
    print(subTest)
    model_name = "TS_CAN"
    # neuer Ordner für Tests
    os.chdir(save_dir)
    try:
        os.makedirs(str(test_name))
    except:
        print("Directory exists...")
    save_path = os.path.join(save_dir, str(test_name))
    os.chdir(save_path)
    workbook = xlsxwriter.Workbook(model_name+ ".xlsx")
    worksheet = workbook.add_worksheet("Results")
    write_header(worksheet)
    predict_vitals(worksheet, test_name, model_name, path_of_video_test, path_results, model_checkpoint , database_name)
    print("Ready with this model")
    workbook.close()
