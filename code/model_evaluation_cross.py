import pathlib
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


# import heartpy as hp

'''Code adapted from https://raw.githubusercontent.com/KISMED-TUDa/rPPG-CANs/main/code/final_evaluation.py'''

def write_header(worksheet):
    header = ['Database','name','p', 'MAE', 'RMSE', 'HR_predicted', 'HR_GT']
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

def plot_signals(pulse_pred, pulse_truth, save_path, dataset, model_name):
    fib, ax = plt.subplots(2, 1)
    ax[0].plot(pulse_pred)
    ax[1].plot(pulse_truth)
    ax[0].title.set_text("Prediction")
    ax[1].title.set_text("Groung Truth")
    if dataset =="VIPL":
        directory = pathlib.Path("/home/bhargav/MTTS-CAN/code/Predictions/vipl-pure/"+model_name+"/")
        directory.mkdir(exist_ok=True)
        save_path = save_path.stem
        plt.savefig(directory / save_path )
    elif dataset == "PURE":
        directory = pathlib.Path("/home/bacharya/MTTS-CAN/code/Predictions/PURE/"+model_name+"/")
        directory.mkdir(exist_ok=True)
        save_path = save_path.stem
        plt.savefig(directory / save_path )
    elif dataset == "COHFACE":
        directory = pathlib.Path("/home/bacharya/MTTS-CAN/code/Predictions/cohface/"+model_name+"/")
        directory.mkdir(exist_ok=True)
        save_path = save_path.parent.stem + "_"+save_path.parent.parent.stem
        plt.savefig(directory / save_path )


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
            fs = 20
        elif database_name=="PURE":
            data = h5py.File(sample_data_path, "r")
            dXsub = np.array(data["data"])
            dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
            dXsub = dXsub[:dXsub_len, :, :, :]
            fs = 30
        elif database_name=="VIPL":
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
        #overall heartrate
        N = 30 * fs
        pulse_fft = np.expand_dims(pulse_pred, 0)
        f, pxx = signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
        fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        frange = np.take(f, fmask)
        HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60

        ##### ground truth data resampled  #######
        if(database_name == "COHFACE"):
            truth_path = sample_data_path.replace(".avi", ".hdf5")   # akutell für COHACE...
            truth_path_data = truth_path.replace(".hdf5", "_dataFile.hdf5")   # akutell für COHACE...
            print(truth_path_data)
        elif database_name == "PURE":
            truth_path = sample_data_path
            truth_path_data = sample_data_path
        elif database_name == "VIPL":
            truth_path = sample_data_path
            truth_path_data = sample_data_path
        else:
            return print("Error in finding the ground truth signal...")
        data = h5py.File(truth_path,"r")
        pulse_truth = np.array(data["pulse"])
        data.close()
        plot_signals(pulse_pred,pulse_truth, pathlib.Path(sample_data_path), database_name,model_name)

        pulse_truth = detrend(np.cumsum(pulse_truth), 100)
        [b_pulse_pred, a_pulse_pred] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_truth = scipy.signal.filtfilt(b_pulse_pred, a_pulse_pred, np.double(pulse_truth))
        #pulse_truth = np.array(mms.fit_transform(pulse_truth.reshape(-1,1))).flatten()



        pulse_fft = np.expand_dims(pulse_truth, 0)
        f, pxx = signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
        fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        frange = np.take(f, fmask)
        HR_GT = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60
        ########### BPM ###########
        bvp = BVPsignal(pulse_pred.reshape(1,len(pulse_pred)),fs)
        BPM, times = bvp.getBPM(winsize=10)
        if database_name == "COHFACE":
            bvpGT = BVPsignal(pulse_truth.reshape(1,len(pulse_truth)),256)
            BPMGT, timesGT = bvpGT.getBPM(winsize=10)
        elif database_name =="PURE":
            bvpGT = BVPsignal(pulse_truth.reshape(1, len(pulse_truth)), 30)
            BPMGT, timesGT = bvpGT.getBPM(winsize=10)
        elif database_name =="VIPL":
            bvpGT = BVPsignal(pulse_truth.reshape(1, len(pulse_truth)), 30)
            BPMGT, timesGT = bvpGT.getBPM(winsize=10)
        print(times, timesGT)
        ######## name files #############
        if(database_name=="COHFACE"):
            nameStr = str(sample_data_path).replace("data.avi", "")
            print(nameStr)
        elif database_name=="PURE":
            nameStr = str(sample_data_path).replace("_dataFile.hdf5", "")
        elif database_name == "VIPL":
            nameStr = str(sample_data_path).replace(".hdf5", "")
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
        worksheet.write(counter_video,2, p[0])
        worksheet.write(counter_video,3, MAE)
        worksheet.write(counter_video,4, RMSE)
        worksheet.write(counter_video,5, HR)
        worksheet.write(counter_video,6, HR_GT)
        list_bpm[sample_data_path] = [BPM, times, BPMGT, timesGT]
        counter_video += 1
        old_database = database_name
    with open(f"{database_name}_{model_name}.pkl", "wb") as output:
        pickle.dump(list_bpm, output) 
   
if __name__ == "__main__":
    #database_name = "COHFACE"
    database_name = "VIPL"
    #database_name = "PURE"
    model_checkpoints = [
        ("Hybrid_CAN", "/home/bhargav/PURE/checkpoints/normalized-PURE-Hybrid_CAN-4-19/cv_0_epoch02_model.tf"),
        ("TS_CAN", "/home/bhargav/PURE/checkpoints/normalized-PURE-TS_CAN-4-19/cv_0_epoch02_model.tf"),
        ("CAN_3D", "/home/bhargav/PURE/checkpoints/normalized-PURE-CAN_3D-4-19/cv_0_epoch02_model.tf")]
    test_name = "PURE-VIPL-CROSS"
    if database_name=="VIPL":
        path_results = "/home/bhargav/vipl/predictions/"
        data_dir = '/home/bhargav/vipl/hdf5/'
        split_path = '/home/bhargav/vipl/data/'
        subTrain, subDev, subTest = split_subj_(split_path, "VIPL")
        path_of_video_test = sort_dataFile_list_(data_dir, subTest, "VIPL", trainMode=True)
        save_dir = '/home/bhargav/vipl/predictions/'

    for model_name, model_checkpoint in model_checkpoints:
        print(path_of_video_test)
        print(subTest)
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
