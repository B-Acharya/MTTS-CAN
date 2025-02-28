'''
Data Generator for Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
Author: Xin Liu
'''

import math

import h5py
import numpy as np
from tensorflow import keras
from pre_process import get_nframe_video, split_subj_, sort_dataFile_list_

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


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths_of_videos, maxlen, dim, batch_size=32, frame_depth=10,
                 shuffle=True, temporal=True, respiration=0, dataset=None):
        self.dim = dim
        self.batch_size = batch_size
        self.paths_of_videos = paths_of_videos
        self.nframe_per_video = maxlen 
        self.shuffle = shuffle
        self.maxLen_Video = maxlen
        self.temporal = temporal
        self.frame_depth = frame_depth
        self.respiration = respiration
        self.on_epoch_end()
        self.dataset = dataset

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.paths_of_videos) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.paths_of_videos[k] for k in indexes]
        #print(list_IDs_temp)
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths_of_videos))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_video_temp):
        'Generates data containing batch_size samples'

        if self.respiration == 1:
            label_key = "drsub"
        else:
            label_key = 'dysub'

        if self.temporal == 'CAN_3D':
            sum_frames_batch = get_frame_sum_3D_Hybrid(list_video_temp, self.maxLen_Video)
            data = np.zeros((sum_frames_batch, self.dim[0], self.dim[1],self.frame_depth,  6), dtype=np.float32)
            label = np.zeros((sum_frames_batch, self.frame_depth), dtype=np.float32)
            index_counter = 0
            for index, temp_path in enumerate(list_video_temp):
                f1 = h5py.File(temp_path, 'r')
                dXsub = np.array(f1['data'])
                dysub = np.array(f1['pulse'])
                # if dXsub.shape[0] > self.maxLen_Video: # only 30 sek videos
                #   dXsub = dXsub[0:self.maxLen_Video, :,:,:]
                #   dysub = dysub[0:self.maxLen_Video]
                num_window = int(dXsub.shape[0]) -(self.frame_depth+1)  
                tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (491, 10, 36, 36 ,6) (169, 10, 36, 36, 6)
                                  for f in range(num_window)])
                tempY = np.array([dysub[f:f + self.frame_depth] #(491,10,1) - (169, 10, 1)
                                  for f in range(num_window)])
                tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
                tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
                tempY = np.reshape(tempY, (num_window, self.frame_depth)) # (169, 10)
                data[index_counter: index_counter + num_window, :, :, :, :] = tempX
                label[index_counter: index_counter + num_window, :] = tempY
                index_counter += num_window

            motion_data = data[:, :, :, :, :3]
            apperance_data = data[:, :, :, :, -3:]
            max_data = num_window*self.frame_depth
            motion_data = motion_data[0:max_data, :, :, :]
            apperance_data = apperance_data[0:max_data, :, :, :]
            label = label[0:max_data, :]
            output = (motion_data, apperance_data)

       # if self.temporal == 'CAN_3D':
       #     num_window = self.nframe_per_video - (self.frame_depth + 1)
       #     data = np.zeros((num_window*len(list_video_temp), self.dim[0], self.dim[1], self.frame_depth, 6),
       #                     dtype=np.float32)
       #     label = np.zeros((num_window*len(list_video_temp), self.frame_depth), dtype=np.float32)
       #     for index, temp_path in enumerate(list_video_temp):
       #         f1 = h5py.File(temp_path, 'r')
       #         #dXsub = np.transpose(np.array(f1["dXsub"]))
       #         dXsub = np.array(f1["dXsub"])
       #         dysub = np.array(f1[label_key])
       #         tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (169, 10, 36, 36, 6)
       #                           for f in range(num_window)])
       #         tempY = np.array([dysub[f:f + self.frame_depth] # (169, 10, 1)
       #                           for f in range(num_window)])
       #         tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
       #         tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
       #         tempY = np.reshape(tempY, (num_window, self.frame_depth)) # (169, 10)
       #         data[index*num_window:(index+1)*num_window, :, :, :, :] = tempX
       #         label[index*num_window:(index+1)*num_window, :] = tempY
       #     output = (data[:, :, :, :, :3], data[:, :, :, :, -3:])
        elif self.temporal == 'MT_CAN_3D':
            num_window = self.nframe_per_video - (self.frame_depth + 1)
            data = np.zeros((num_window*len(list_video_temp), self.dim[0], self.dim[1], self.frame_depth, 6),
                            dtype=np.float32)
            label_y = np.zeros((num_window*len(list_video_temp), self.frame_depth), dtype=np.float32)
            label_r = np.zeros((num_window * len(list_video_temp), self.frame_depth), dtype=np.float32)
            for index, temp_path in enumerate(list_video_temp):
                f1 = h5py.File(temp_path, 'r')
                #TODO:why the trasform was in place ?
                #dXsub = np.transpose(np.array(f1["dXsub"]))
                dXsub = np.array(f1["dXsub"])
                drsub = np.array(f1['drsub'])
                dysub = np.array(f1['dysub'])
                tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (169, 10, 36, 36, 6)
                                  for f in range(num_window)])
                tempY_y = np.array([dysub[f:f + self.frame_depth] # (169, 10, 1)
                                  for f in range(num_window)])
                tempY_r = np.array([drsub[f:f + self.frame_depth] # (169, 10, 1)
                                  for f in range(num_window)])
                tempX = np.swapaxes(tempX, 1, 3) # (1VIPL69, 36, 36, 10, 6)
                tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
                tempY_y = np.reshape(tempY_y, (num_window, self.frame_depth)) # (169, 10)
                tempY_r = np.reshape(tempY_r, (num_window, self.frame_depth))  # (169, 10)
                data[index*num_window:(index+1)*num_window, :, :, :, :] = tempX
                label_y[index*num_window:(index+1)*num_window, :] = tempY_y
                label_r[index * num_window:(index + 1) * num_window, :] = tempY_r
            output = (data[:, :, :, :, :3], data[:, :, :, :, -3:])
            label = (label_y, label_r)
        elif self.temporal == 'CAN':
            data = np.zeros((self.nframe_per_video * len(list_video_temp), self.dim[0], self.dim[1], 6), dtype=np.float32)
            label = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
            for index, temp_path in enumerate(list_video_temp):
                f1 = h5py.File(temp_path, 'r')
                dXsub = np.transpose(np.array(f1["dXsub"])) #dRsub for respiration
                dysub = np.array(f1[label_key])
                data[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :, :, :] = dXsub
                label[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :] = dysub
            output = (data[:, :, :, :3], data[:, :, :, -3:])
        elif self.temporal == 'MT_CAN':
            data = np.zeros((self.nframe_per_video * len(list_video_temp), self.dim[0], self.dim[1], 6),
                            dtype=np.float32)
            label_y = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
            label_r = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
            for index, temp_path in enumerate(list_video_temp):
                f1 = h5py.File(temp_path, 'r')
                dXsub = np.transpose(np.array(f1["dXsub"]))  # dRsub for respiration
                drsub = np.array(f1['drsub'])
                dysub = np.array(f1['dysub'])
                data[index * self.nframe_per_video:(index + 1) * self.nframe_per_video, :, :, :] = dXsub
                label_y[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :] = dysub
                label_r[index * self.nframe_per_video:(index + 1) * self.nframe_per_video, :] = drsub
            output = (data[:, :, :, :3], data[:, :, :, -3:])
            label = (label_y, label_r)
        #elif self.temporal == 'TS_CAN':
        #    sum_frames_batch = get_frame_sum(list_video_temp, self.maxLen_Video)
        #    data = np.zeros((sum_frames_batch, self.dim[0], self.dim[1], 6), dtype=np.float32)
        #    label = np.zeros((sum_frames_batch, 1), dtype=np.float32)
        #    num_window = int(sum_frames_batch/ self.frame_depth)
        #    index_counter = 0
        #    for index, temp_path in enumerate(list_video_temp):
        #        f1 = h5py.File(temp_path, 'r')
        #        dXsub = np.array(f1['data'])
        #        dysub = np.array(f1['pulse'])
        #        # if dXsub.shape[0] > self.maxLen_Video: # only 1 min videos
        #        #   current_nframe = self.maxLen_Video
        #        #   dXsub = dXsub[0:self.maxLen_Video, :,:,:]
        #        #   dysub = dysub[0:self.maxLen_Video]
        #        # else:
        #        current_nframe = dXsub.shape[0]
        #        data[index_counter:index_counter+current_nframe, :, :, :] = dXsub
        #        label[index_counter:index_counter+current_nframe, 0] = dysub # data BVP
        #        index_counter += current_nframe
        #    motion_data = data[:, :, :, :3]
        #    apperance_data = data[:, :, :, -3:]
        #    
        #    if num_window % 2 == 1:
        #        num_window = num_window - 1
        #        max_data = num_window * self.frame_depth
        #    else: 
        #        max_data = num_window*self.frame_depth
        #    motion_data = motion_data[0:max_data, :, :, :]
        #    apperance_data = apperance_data[0:max_data, :, :, :]
        #    label = label[0:max_data, 0]
        #    apperance_data = np.reshape(apperance_data, (num_window, self.frame_depth, self.dim[0], self.dim[1], 3))
        #    apperance_data = np.average(apperance_data, axis=1)
        #    apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
        #    apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
        #                                                 apperance_data.shape[2], apperance_data.shape[3],
        #                                                 apperance_data.shape[4]))
        #    output = (motion_data, apperance_data)

        elif self.temporal == 'TS_CAN':
            #     data_len = 1800
            # elif self.dataset =="VIPL":
            maxLen_Video = get_max_frame(list_video_temp)
            sum_frames_batch = get_frame_sum(list_video_temp, maxLen_Video)
            data = np.zeros((sum_frames_batch, self.dim[0], self.dim[1], 6), dtype=np.float32)
            label = np.zeros((sum_frames_batch, 1), dtype=np.float32)
            num_window = int(sum_frames_batch / self.frame_depth)
            index_counter = 0
            for index, temp_path in enumerate(list_video_temp):
                f1 = h5py.File(temp_path, 'r')
                dXsub = np.array(f1['data'])
                dysub = np.array(f1['pulse'])
                # if dXsub.shape[0] > self.maxLen_Video: # only 1 min videos
               	#   current_nframe = self.maxLen_Video
               	#   dXsub = dXsub[0:self.maxLen_Video, :,:,:]
               	#   dysub = dysub[0:self.maxLen_Video]
               	# else:
               	current_nframe = dXsub.shape[0]
                data[index_counter:index_counter + current_nframe, :, :, :] = dXsub
                label[index_counter:index_counter + current_nframe, 0] = dysub  # data BVP
                index_counter += current_nframe
            motion_data = data[:, :, :, :3]
            apperance_data = data[:, :, :, -3:]

            if num_window % 2 == 1:
                num_window = num_window - 1
                max_data = num_window * self.frame_depth
            else:
                max_data = num_window * self.frame_depth
            motion_data = motion_data[0:max_data, :, :, :]
            apperance_data = apperance_data[0:max_data, :, :, :]
            label = label[0:max_data, 0]
            apperance_data = np.reshape(apperance_data, (num_window, self.frame_depth, self.dim[0], self.dim[1], 3))
            apperance_data = np.average(apperance_data, axis=1)
            apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
            apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
               											 apperance_data.shape[2], apperance_data.shape[3],
               											 apperance_data.shape[4]))
            output = (motion_data, apperance_data)
            # print("Here", list_video_temp)
            # data = np.zeros((data_len* len(list_video_temp), self.dim[0], self.dim[1], 6), dtype=np.float32)
            # label = np.zeros((data_len* len(list_video_temp), 1), dtype=np.float32)
            # num_window = int(data_len/ self.frame_depth) * len(list_video_temp)
            # for index, temp_path in enumerate(list_video_temp):
            #     print(temp_path)
            #     f1 = h5py.File(temp_path, 'r')
            #     dXsub = np.transpose(np.array(f1["dXsub"])) #dRsub for respiration
                # dXsub = np.array(f1["data"])[0:data_len,:,:,:] #dRsub for respiration
                # dysub = np.array(f1['pulse'])[0:data_len].reshape(-1,1)
                # data[index*data_len:(index+1)*data_len, :, :, :] = dXsub
                # reshape added to make data processing easier check later for correctness
                # label[index*data_len:(index+1)*data_len, :] = dysub.reshape(-1,1)
                # f1.close()
            # motion_data = data[:, :, :, :3]
            # apperance_data = data[:, :, :, -3:]
            # apperance_data = np.reshape(apperance_data, (num_window, self.frame_depth, self.dim[0], self.dim[1], 3))
            # apperance_data = np.average(apperance_data, axis=1)
            # apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
            # apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
            #                                              apperance_data.shape[2], apperance_data.shape[3],
            #                                              apperance_data.shape[4]))
            # output = (motion_data, apperance_data)
        #elif self.temporal == 'MTTS_CAN':
        #    data = np.zeros((self.nframe_per_video * len(list_video_temp), self.dim[0], self.dim[1], 6), dtype=np.float32)
        #    label_y = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
        #    label_r = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
        #    num_window = int(self.nframe_per_video / self.frame_depth) * len(list_video_temp)
        #    for index, temp_path in enumerate(list_video_temp):
        #        f1 = h5py.File(temp_path, 'r')
        #        dXsub = np.array(f1["dXsub"])
#       #          dXsub = np.transpose(np.array(f1["dXsub"])) #dRsub for respiration
#       #          drsub = np.array(f1['drsub'])
        #        dysub = np.array(f1['dysub']).reshape(-1,1)
        #        data[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :, :, :] = dXsub
        #        label_y[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :] = dysub
#       #          label_r[index * self.nframe_per_video:(index + 1) * self.nframe_per_video, :] = drsub
        #    motion_data = data[:, :, :, :3]
        #    apperance_data = data[:, :, :, -3:]
        #    apperance_data = np.reshape(apperance_data, (num_window, self.frame_depth, self.dim[0], self.dim[1], 3))
        #    apperance_data = np.average(apperance_data, axis=1)
        #    apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
        #    apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
        #                                                 apperance_data.shape[2], apperance_data.shape[3],
        #                                                 apperance_data.shape[4]))
        #    output = (motion_data, apperance_data)
        #    label = label_y
        #    #label = (label_y, label_r)
        elif self.temporal == 'MT_Hybrid_CAN':
            num_window = self.nframe_per_video - (self.frame_depth + 1)
            data = np.zeros((num_window*len(list_video_temp), self.dim[0], self.dim[1], self.frame_depth, 6),
                            dtype=np.float32)
            label_y = np.zeros((num_window*len(list_video_temp), self.frame_depth), dtype=np.float32)
            label_r = np.zeros((num_window * len(list_video_temp), self.frame_depth), dtype=np.float32)
            for index, temp_path in enumerate(list_video_temp):
                f1 = h5py.File(temp_path, 'r')
                dXsub = np.transpose(np.array(f1["dXsub"]))
                drsub = np.array(f1['drsub'])
                dysub = np.array(f1['dysub'])
                tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (169, 10, 36, 36, 6)
                                  for f in range(num_window)])
                tempY_y = np.array([dysub[f:f + self.frame_depth] # (169, 10, 1)
                                  for f in range(num_window)])
                tempY_r = np.array([drsub[f:f + self.frame_depth] # (169, 10, 1)
                                  for f in range(num_window)])
                tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
                tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
                tempY_y = np.reshape(tempY_y, (num_window, self.frame_depth)) # (169, 10)
                tempY_r = np.reshape(tempY_r, (num_window, self.frame_depth))  # (169, 10)
                data[index*num_window:(index+1)*num_window, :, :, :, :] = tempX
                label_y[index*num_window:(index+1)*num_window, :] = tempY_y
                label_r[index * num_window:(index + 1) * num_window, :] = tempY_r
            motion_data = data[:, :, :, :, :3]
            apperance_data = np.average(data[:, :, :, :, -3:], axis=-2)
            output = (motion_data, apperance_data)
            label = (label_y, label_r)
        elif self.temporal == 'Hybrid_CAN':
            print(list_video_temp)
            nframe_per_video = get_min_frame(list_video_temp)
            num_window = nframe_per_video - (self.frame_depth + 1)
            data = np.zeros((num_window*len(list_video_temp), self.dim[0], self.dim[1], self.frame_depth, 6),
                            dtype=np.float32)
            label = np.zeros((num_window*len(list_video_temp), self.frame_depth), dtype=np.float32)
            for index, temp_path in enumerate(list_video_temp):
                f1 = h5py.File(temp_path, 'r')
                #dXsub = np.transpose(np.array(f1["dXsub"]))
                dXsub = np.array(f1["data"])
                dysub = np.array(f1['pulse'])
                tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (169, 10, 36, 36, 6)
                                  for f in range(num_window)])
                tempY = np.array([dysub[f:f + self.frame_depth] # (169, 10, 1)
                                  for f in range(num_window)])
                tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
                tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
                tempY = np.reshape(tempY, (num_window, self.frame_depth)) # (169, 10)
                data[index*num_window:(index+1)*num_window, :, :, :, :] = tempX
                label[index*num_window:(index+1)*num_window, :] = tempY
            motion_data = data[:, :, :, :, :3]
            apperance_data = np.average(data[:, :, :, :, -3:], axis=-2)
            output = (motion_data, apperance_data)
        else:
            raise ValueError('Unsupported Model!')

        return output, label
def get_max_frame(video):
    maxlen = 0
    for video_path in video:
        print(video_path)
        hf = h5py.File(video_path, 'r')
        nframe_per_video = np.array(hf['data']).shape[0]
        hf.close()
        if maxlen <  nframe_per_video:
            maxlen = nframe_per_video
        else:
            pass
    return maxlen


def get_min_frame(video):
    minlen = float('inf')
    for video_path in video:
        hf = h5py.File(video_path, 'r')
        nframe_per_video = np.array(hf['data']).shape[0]
        print(video_path, nframe_per_video)
        hf.close()
        if minlen > nframe_per_video:
            minlen = nframe_per_video
        else:
            pass
    print("Min length per batch", minlen)
    return minlen

def get_frame_sum_3D_Hybrid(list_vid, maxLen_Video):
    frames_sum = 0
    counter = 0
    print(list_vid)
    for vid in list_vid:
        hf = h5py.File(vid, 'r')
        shape = hf['data'].shape
        # if shape[0] > maxLen_Video:
        #   frames_sum += maxLen_Video - 9
        # else: 
        frames_sum += shape[0] - 9
        counter += 1
    return frames_sum
