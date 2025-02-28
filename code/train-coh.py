'''
Training Script for Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
Author: Xin Liu, Daniel McDuff
'''
# %%
from __future__ import print_function

import argparse
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




os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

#for reproducibility 
#tf.config.experimental.enable_op_determinism()
seed = 100
tf.test.is_gpu_available()
tf.keras.utils.set_random_seed(seed)
list_gpu = tf.config.list_physical_devices('GPU')
print("List of gpus are", list_gpu)
tf.keras.backend.clear_session()
print(tf.__version__)
list_gpu = tf.config.list_physical_devices('GPU')


# %%
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-exp', '--exp_name', type=str,
                    help='experiment name')
parser.add_argument('-i', '--data_dir', type=str, help='Location for the dataset')
parser.add_argument('-database_name', '--database_name', type=str, help='which dataset',default="COHFACE")

parser.add_argument('-inter', '--inter_dir', type=str, help='intermediate saving location for the data')
parser.add_argument('-o', '--save_dir', type=str, default='./rPPG-checkpoints',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-a', '--nb_filters1', type=int, default=32,
                    help='number of convolutional filters to use')
parser.add_argument('-b', '--nb_filters2', type=int, default=64,
                    help='number of convolutional filters to use')
parser.add_argument('-c', '--dropout_rate1', type=float, default=0.25,
                    help='dropout rates')
parser.add_argument('-d', '--dropout_rate2', type=float, default=0.5,
                    help='dropout rates')
parser.add_argument('-l', '--lr', type=float, default=1.0,
                            help='learning rate')
parser.add_argument('-e', '--nb_dense', type=int, default=128,
                    help='number of dense units')
parser.add_argument('-f', '--cv_split', type=int, default=0,
                    help='cv_split')
parser.add_argument('-g', '--nb_epoch', type=int, default=2,
                    help='nb_epoch')
parser.add_argument('-t', '--nb_task', type=int, default=12,
                    help='nb_task')
parser.add_argument('-fd', '--frame_depth', type=int, default=10,
                    help='frame_depth for CAN_3D, TS_CAN, Hybrid_CAN')
parser.add_argument('-temp', '--temporal', type=str, default='Hybrid_CAN',
                    help='CAN, MT_CAN, CAN_3D, MT_CAN_3D, Hybrid_CAN, \
                    MT_Hybrid_CAN, TS_CAN, MTTS_CAN ')
parser.add_argument('-save', '--save_all', type=int, default=1,
                    help='save all or not')
parser.add_argument('-resp', '--respiration', type=int, default=0,
                    help='train with resp or not')
parser.add_argument('-m', '--manual', type=int, default=0,
                    help='train with smaller epocs and save and restart training')
parser.add_argument('-init', '--initial', type=int, default=0,
                    help='first training instance')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# %% Training
def train(args, subTrain, subTest, cv_split, img_rows=36, img_cols=36):
    print('================================')
    print('Train...')
    print('subTrain', subTrain)
    print('subTest', subTest)

    input_shape = (img_rows, img_cols, 3)

    path_of_video_tr = subTrain
    path_of_video_test = subTest 


    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    path_of_video_tr = sort_dataFile_list_(args.data_dir, subTrain, args.database_name, trainMode=True)
    path_of_video_test = sort_dataFile_list_(args.data_dir, subTest, args.database_name, trainMode=True)

    print('sample path: ', path_of_video_tr[0])
    nframe_per_video = get_nframe_video(path_of_video_tr[0])
    print('Trian Length: ', len(path_of_video_tr))
    print('Test Length: ', len(path_of_video_test))
    print('nframe_per_video', nframe_per_video)

    print('Train length', len(path_of_video_tr))
    print('Test length', len(path_of_video_test))
    if len(list_gpu) > 1:
        print("Using MultiWorkerMirroredStrategy")
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else: 
        print("Using MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        #added Bhargav
        if strategy.num_replicas_in_sync == 4:
            print("Using 4 GPUs for training")
            if args.temporal == 'CAN' or args.temporal == 'MT_CAN':
                args.batch_size = 16
            elif args.temporal == 'CAN_3D' or args.temporal == 'MT_CAN_3D':
                args.batch_size = 2
            elif args.temporal == 'TS_CAN' or args.temporal == 'MTTS_CAN':
                args.batch_size = 12
            elif args.temporal == 'Hybrid_CAN' or args.temporal == 'MT_Hybrid_CAN':
                args.batch_size = 2
            else:
                raise ValueError('Unsupported Model Type!')
        elif strategy.num_replicas_in_sync == 8:
            print('Using 8 GPUs for training!')
            args.batch_size = 1
        elif strategy.num_replicas_in_sync == 2:
            args.batch_size = 4
        else:
            raise Exception('Only supporting 4 GPUs or 8 GPUs now. Please adjust learning rate in the training script!')
        if args.initial:
            if args.temporal == 'CAN':
                print('Using CAN!')
                model = CAN(args.nb_filters1, args.nb_filters2, input_shape, dropout_rate1=args.dropout_rate1,
                            dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
            elif args.temporal == 'MT_CAN':
                print('Using MT_CAN!')
                model = MT_CAN(args.nb_filters1, args.nb_filters2, input_shape, dropout_rate1=args.dropout_rate1,
                               dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
            elif args.temporal == 'CAN_3D':
                print('Using CAN_3D!')
                input_shape = (img_rows, img_cols, args.frame_depth, 3)
                model = CAN_3D(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                               dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
            elif args.temporal == 'MT_CAN_3D':
                print('Using MT_CAN_3D!')
                input_shape = (img_rows, img_cols, args.frame_depth, 3)
                model = MT_CAN_3D(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                                  dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                                  nb_dense=args.nb_dense)
            elif args.temporal == 'TS_CAN':
                print('Using TS_CAN!')
                input_shape = (img_rows, img_cols, 3)
                model = TS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                               dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
            elif args.temporal == 'MTTS_CAN':
                print('Using MTTS_CAN!')
                input_shape = (img_rows, img_cols, 3)
                model = MTTS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                                 dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
            elif args.temporal == 'Hybrid_CAN':
                print('Using Hybrid_CAN!')
                input_shape_motion = (img_rows, img_cols, args.frame_depth, 3)
                input_shape_app = (img_rows, img_cols, 3)
                model = Hybrid_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape_motion,
                                   input_shape_app,
                                   dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                                   nb_dense=args.nb_dense)
            elif args.temporal == 'MT_Hybrid_CAN':
                print('Using MT_Hybrid_CAN!')
                input_shape_motion = (img_rows, img_cols, args.frame_depth, 3)
                input_shape_app = (img_rows, img_cols, 3)
                model = MT_Hybrid_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape_motion,
                                      input_shape_app,
                                      dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                                      nb_dense=args.nb_dense)
            else:
                raise ValueError('Unsupported Model Type!')
        else:
            print("Loading old model")
            model = keras.models.load_model(args.inter_dir)
        #optimizer = tf.python.keras.optimizers.Adadelta(learning_rate=args.lr)
        if args.initial:
            optimizer = Adadelta(learning_rate=args.lr)
            if args.temporal == 'MTTS_CAN' or args.temporal == 'MT_Hybrid_CAN' or args.temporal == 'MT_CAN_3D' or \
                    args.temporal == 'MT_CAN':
                losses = {"output_1": "mean_squared_error", "output_2": "mean_squared_error"}
                loss_weights = {"output_1": 1.0, "output_2": 1.0}
                model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)
            else:
                model.compile(loss='mean_squared_error', optimizer=optimizer)
        else:
            print("model.compile skipped as we are using old model")
            pass
        print('learning rate: ', args.lr)

        # %% Create data genener
        training_generator = DataGenerator(path_of_video_tr, nframe_per_video, (img_rows, img_cols),
                                           batch_size=args.batch_size, frame_depth=args.frame_depth,
                                           temporal=args.temporal, respiration=args.respiration, shuffle = True, dataset="COHFACE")
        validation_generator = DataGenerator(path_of_video_test, nframe_per_video, (img_rows, img_cols),
                                             batch_size=args.batch_size, frame_depth=args.frame_depth,
                                             temporal=args.temporal, respiration=args.respiration, shuffle= True,dataset="COHFACE")
        # %%  Checkpoint Folders
        checkpoint_folder = str(os.path.join(args.save_dir, args.exp_name))
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        cv_split_path = str(os.path.join(checkpoint_folder, "cv_" + str(cv_split)))

	#loading form a checkpoint to resume training
        if args.manual:
            print("Manual training mode")
            print(f"setting number of epocs to {args.nb_epoch}")
            
        #Changed from .hdf5 to .tf
        # %% Callbacks
        if args.save_all == 1:
            save_best_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=cv_split_path + "_epoch{epoch:02d}_model.tf",
                save_best_only=False, verbose=1)
        else:
            save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cv_split_path + "_last_model.tf",
                                                                    save_best_only=False, verbose=1)
        csv_logger = tf.keras.callbacks.CSVLogger(filename=cv_split_path + '_train_loss_log.csv')
        hb_callback = HeartBeat(training_generator, validation_generator, args, str(cv_split), checkpoint_folder)
        print("***********************")
        print(model.summary())

        # %% Model Training and Saving Results
        history = model.fit(x=training_generator, validation_data=validation_generator, epochs=args.nb_epoch, verbose=1,
                            shuffle=True, callbacks=[csv_logger, save_best_callback, hb_callback], validation_freq=1)
        if args.manual:
            print('****************************************')
            print("saving the model")
            print('****************************************')
            model.save(args.inter_dir,save_format='tf')

        val_loss_history = history.history['val_loss']
        val_loss = np.array(val_loss_history)
        np.savetxt((cv_split_path + '_val_loss_log.csv'), val_loss, delimiter=",")

        score = model.evaluate_generator(generator=validation_generator, verbose=1)

        print('****************************************')
        if args.temporal == 'MTTS_CAN' or args.temporal == 'MT_Hybrid_CAN' or args.temporal == 'MT_CAN_3D' \
                or args.temporal == 'MT_CAN':
            print('Average Test Score: ', score[0])
            print('PPG Test Score: ', score[1])
            print('Respiration Test Score: ', score[2])
        else:
            print('Test score:', score)
        print('****************************************')
        print('Start saving predicitions from the last epoch')

        #training_generator = DataGenerator(path_of_video_tr, nframe_per_video, (img_rows, img_cols),
        #                                   batch_size=args.batch_size, frame_depth=args.frame_depth,
        #                                   temporal=args.temporal, respiration=args.respiration, shuffle=False)

        validation_generator = DataGenerator(path_of_video_test, nframe_per_video, (img_rows, img_cols),
                                             batch_size=args.batch_size, frame_depth=args.frame_depth,
                                             temporal=args.temporal, respiration=args.respiration, shuffle=False, dataset="COHFACE")

        #yptrain = model.predict(training_generator, verbose=1)
        #scipy.io.savemat(checkpoint_folder + '/yptrain_best_' + '_cv' + str(cv_split) + '.mat',
        #                 mdict={'yptrain': yptrain})
        yptest = model.predict(validation_generator, verbose=1)
        scipy.io.savemat(checkpoint_folder + '/yptest_best_' + '_cv' + str(cv_split) + '.mat',
                         mdict={'yptest': yptest})

        print('Finish saving the results from the last epoch')


# %% Training
def get_video_list(path,basepath):
    videoPaths = []
    with open(path,"r") as f:
        for line in f.readlines():
            #TODO: update the cohface directory to have - and not _
            videoPaths.append(basepath+ "_".join(line.strip("\n").strip("data").strip("/").split("/"))+".hdf5")
    return videoPaths

print('Using Split ', str(args.cv_split))
subTrain, subDev, subTest = split_subj_("/work/data/bacharya/cohface/", "COHFACE")
train(args, subTrain, subDev, args.cv_split)
