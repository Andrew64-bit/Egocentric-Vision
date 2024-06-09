import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
from .actionNet_emg_record import ActionNetVideoEmgRecord
from .actionNet_emg_rgb_record import ActionNetVideoEmgRgbRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import numpy as np
import torch
import pickle


#-------------------------------------------------------only RGB-------------------------------------------------------------------------#


class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        # A single EpicVideoRecord is a single clip,
        # example of tup is :
        '''
            (0, uid                        13744
            participant_id               P08
            video_id                  P08_09
            narration          get mocha pot
            start_timestamp      00:00:02.61
            stop_timestamp       00:00:03.61
            start_frame                  156
            stop_frame                   216
            verb                         get
            verb_class                     0
            Name: 0, dtype: object)
        '''
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat


        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################

        num_frames_record = record.num_frames[modality] #100
        num_frames_per_clip = self.num_frames_per_clip[modality] #20
        num_clips = self.num_clips #8
        stride = self.stride

        # offset da sommare ad ogni index per centrare la clip
        centroid_offset = num_frames_record / num_clips / 2 - ( num_frames_per_clip * stride / 2 )
        segment_dim = num_frames_record / num_clips

        all_indices = []

        # --- dense sampling ---
        if self.dense_sampling[modality]:
            for _ in range(num_clips):
                # prende un punto centrale randomico assicurandosi un certo offset dall'inizio e dalla fine
                central_point = np.random.randint(centroid_offset, num_frames_record-centroid_offset)
                # starting index of the clip, 0 in case of negative values
                start_idx = max(0, central_point - num_frames_per_clip * stride / 2)
                indices = [(idx * stride + start_idx) % num_frames_record for idx in range(num_frames_per_clip)]
                indices.sort()
                all_indices += indices            
            
        # --- uniform sampling ---
        else:
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                frame_idx = np.multiply(np.arange(self.num_frames_per_clip[modality]), average_duration) + \
                            np.random.randint(average_duration, size=self.num_frames_per_clip[modality])
                all_indices = np.tile(frame_idx, self.num_clips)
            else:
                all_indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))

        return all_indices

        # logger.info("----------------------")
        # logger.info(f"num_frames : {num_frames_record}, indeces : {all_indices}")
        # logger.info("----------------------")
        

    def _get_val_indices(self, record, modality):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        
        num_frames_record = record.num_frames[modality] #100
        num_frames_per_clip = self.num_frames_per_clip[modality] #20
        num_clips = self.num_clips #8
        stride = self.stride

        # offset da sommare ad ogni index per centrare la clip
        centroid_offset = num_frames_record / num_clips / 2 - ( num_frames_per_clip * stride / 2 )
        segment_dim = num_frames_record / num_clips

        all_indices = []

        # --- dense sampling ---
        if self.dense_sampling[modality]:
            #logger.info("___________________________________________________DENSE SAMPLING___________________________________________________")
            # prende il max index possibile per far stare la clip
            max_idx = max(0, num_frames_record - segment_dim)
            
            # indici iniziali di ogni segmento centrati al centroide del segmento
            clips_start_idx = np.linspace(0, max_idx, num=num_clips, dtype=float)
            segment_dim = clips_start_idx[1]
            # caso in cui i segmenti non si overlappano, centralizza il segmento
            if segment_dim > num_frames_per_clip*stride:
                clips_start_idx += centroid_offset

            for start_idx in clips_start_idx:
                indices = [(idx * stride +start_idx) % num_frames_record for idx in range(num_frames_per_clip)]
                indices.sort()
                all_indices += indices

        # --- uniform sampling ---
        else:
            #logger.info("___________________________________________________UNIFORM SAMPLING___________________________________________________")

            # prende il max index possibile per far stare la clip
            max_idx = max(0, num_frames_record - segment_dim)
            # indici iniziali di ogni segmento centrati al centroide del segmento
            clips_start_idx = np.linspace(0, max_idx, num=num_clips, dtype=int)
            segment_dim = clips_start_idx[1]
            for start_idx in clips_start_idx:
                all_indices += np.linspace(start_idx, start_idx + segment_dim, num=num_frames_per_clip, dtype=int).tolist()



        # logger.info("----------------------")
        # logger.info(f"num_frames : {num_frames_record}, indeces : {all_indices}")
        # logger.info("----------------------")
        
        return all_indices

    # item si riferisce alla singola clip
    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]
        # record is :
        '''
            (0, uid                        13744
            participant_id               P08
            video_id                  P08_09
            narration          get mocha pot
            start_timestamp      00:00:02.61
            stop_timestamp       00:00:03.61
            start_frame                  156
            stop_frame                   216
            verb                         get
            verb_class                     0
            Name: 0, dtype: object)
        '''

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)
        
       
        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img
    
        #logger.info(f"------------------------------segment_indices(size = {len(segment_indices['RGB'])})------------------------------\n{segment_indices}\n------------------------------frames(size = {frames['RGB'].shape})------------------------------\n{frames['RGB'][0]}")


        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)


class FeaturesDataset(data.Dataset):

    def __init__(self, features_file, mode='train'):

        # Carica le feature dal file pickle
        features = pd.read_pickle(features_file+"_"+mode+".pkl")

        # Estrai le feature e le etichette
        list_of_features = [np.array(feature['features_RGB']) for feature in features["features"]]
        labels = [feature['label'] for feature in features["features"]]
        # Converti list_of_features in un array NumPy
        list_of_features = np.array(list_of_features)
        labels = np.array(labels)

        self.mode = mode
        self.features = list_of_features
        self.labels = labels

    def __len__(self):
        return len(self.features)
            
    def __getitem__(self,idx):
        return self.features[idx], self.labels[idx]
    
class FeaturesExtendedDataset(data.Dataset):

    def __init__(self, features_file, mode='train'):

        # Carica le feature dal file pickle
        features = pd.read_pickle(features_file+"_"+mode+".pkl")

        # Estrai le feature e le etichette
        list_of_features = [np.array(f) for feature in features["features"] for f in feature['features_RGB']]
        labels = [feature['label'] for feature in features["features"]]
        labels_extended = [label for label in labels for _ in range(5)]
        # Converti list_of_features in un array NumPy
        list_of_features = np.array(list_of_features)
        labels = np.array(labels_extended)


        self.mode = mode
        self.features = list_of_features
        self.labels = labels

    def __len__(self):
        return len(self.features)
            
    def __getitem__(self,idx):
        return self.features[idx], self.labels[idx]
    
class FeaturesExtendedEMGDataset(data.Dataset):
    def __init__(self, features_file):
        # Carica i dati dal file pickle
        with open(features_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        # Restituisce il numero di campioni nel dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # Restituisce il campione (features e labels) alla posizione data
        sample = self.data[idx]
        features = sample['features']
        labels = sample['labels']
        return {'features': features, 'labels': labels}
        
#-------------------------------------------------------only EMG-------------------------------------------------------------------------#


class ActionNetEmgDataset(data.Dataset, ABC):
    def __init__(self, mode, num_frames_per_clip, num_clips, dense_sampling, annotations_dir, data_dir, stride, transform=None, **kwargs):

        """
        mode: str (train, test/val)
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        """
        self.mode = mode  # 'train', 'val' or 'test'
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = stride
        self.transform = transform  # pipeline of transforms
        self.additional_info = False
        if self.mode == "train":
            pickle_name = "ActionNet_train.pkl"
        else:
            pickle_name = "ActionNet_test.pkl"

        list_action = pd.read_pickle(os.path.join(annotations_dir, pickle_name))
        #logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        # A single EpicVideoRecord is a single clip,
        # example of tup is :
        file = ['S00_2.pkl' ,'S01_1.pkl', 'S02_2.pkl', 'S02_3.pkl', 'S02_4.pkl', 'S03_1.pkl', 'S03_2.pkl', 'S04_1.pkl', 'S05_2.pkl', 'S06_1.pkl', 'S06_2.pkl', 'S07_1.pkl', 'S08_1.pkl', 'S09_2.pkl']
        self.video_list = []
        for f in file:
            subject = list_action[list_action['file'] == f]
            index = np.array(subject['index']) - 1
            file_path = os.path.join(data_dir, f)
            df = pd.read_pickle(file_path)
            #print(index)
            df_idx = df.iloc[index]
            for (_,action) in df_idx.iterrows():
                descr = action['description']
                label = action['description_class']
                #start = action['start_time']
                #end = action['end_time']
                for el in action['emg_data']:
                    sample = ActionNetVideoEmgRecord(descr,label, el)
                    self.video_list.append(sample)



        #mappa_filtrata =list(zip(self.list_file["narration"][283:287],self.list_file["verb"][283:287],self.list_file["verb_class"][283:287]))
        #logger.info(mappa_filtrata)


    def _get_train_indices(self, record):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################

        num_frames_record = record.size #100
        num_frames_per_clip = self.num_frames_per_clip #20
        num_clips = self.num_clips #5
        stride = self.stride

        # offset da sommare ad ogni index per centrare la clip
        centroid_offset = num_frames_record / num_clips / 2 - ( num_frames_per_clip * stride / 2 )
        segment_dim = num_frames_record / num_clips

        all_indices = []

        # --- dense sampling ---
        if self.dense_sampling:
            for _ in range(num_clips):
                # prende un punto centrale randomico assicurandosi un certo offset dall'inizio e dalla fine
                central_point = np.random.randint(centroid_offset, num_frames_record-centroid_offset)
                # starting index of the clip, 0 in case of negative values
                start_idx = max(0, central_point - num_frames_per_clip * stride / 2)
                indices = [(idx * stride + start_idx) % num_frames_record for idx in range(num_frames_per_clip)]
                indices.sort()
                all_indices += indices            
            
        # --- uniform sampling ---
        else:
            average_duration = num_frames_record // self.num_frames_per_clip
            if average_duration > 0:
                frame_idx = np.multiply(np.arange(self.num_frames_per_clip), average_duration) + \
                            np.random.randint(average_duration, size=self.num_frames_per_clip)
                all_indices = np.tile(frame_idx, self.num_clips)
            else:
                all_indices = np.zeros((self.num_frames_per_clip * self.num_clips,))

        return all_indices

        # logger.info("----------------------")
        # logger.info(f"num_frames : {num_frames_record}, indeces : {all_indices}")
        # logger.info("----------------------")


    def _get_val_indices(self, record):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        
        num_frames_record = record.size #100
        num_frames_per_clip = self.num_frames_per_clip #20
        num_clips = self.num_clips #5
        stride = self.stride

        # offset da sommare ad ogni index per centrare la clip
        centroid_offset = num_frames_record / num_clips / 2 - ( num_frames_per_clip * stride / 2 )
        segment_dim = num_frames_record / num_clips

        all_indices = []

        # --- dense sampling ---
        if self.dense_sampling:
            #logger.info("___________________________________________________DENSE SAMPLING___________________________________________________")
            # prende il max index possibile per far stare la clip
            max_idx = max(0, num_frames_record - segment_dim)
            
            # indici iniziali di ogni segmento centrati al centroide del segmento
            clips_start_idx = np.linspace(0, max_idx, num=num_clips, dtype=float)
            segment_dim = clips_start_idx[1]
            # caso in cui i segmenti non si overlappano, centralizza il segmento
            if segment_dim > num_frames_per_clip*stride:
                clips_start_idx += centroid_offset

            for start_idx in clips_start_idx:
                indices = [(idx * stride +start_idx) % num_frames_record for idx in range(num_frames_per_clip)]
                indices.sort()
                all_indices += indices

        # --- uniform sampling ---
        else:
            #logger.info("___________________________________________________UNIFORM SAMPLING___________________________________________________")

            # prende il max index possibile per far stare la clip
            max_idx = max(0, num_frames_record - segment_dim)
            # indici iniziali di ogni segmento centrati al centroide del segmento
            clips_start_idx = np.linspace(0, max_idx, num=num_clips, dtype=int)
            segment_dim = clips_start_idx[1]
            for start_idx in clips_start_idx:
                all_indices += np.linspace(start_idx, start_idx + segment_dim, num=num_frames_per_clip, dtype=int).tolist()



        # logger.info("----------------------")
        # logger.info(f"num_frames : {num_frames_record}, indeces : {all_indices}")
        # logger.info("----------------------")
        
        return all_indices

    # item si riferisce alla singola clip
    def __getitem__(self, index):

        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]
        if self.mode == "train":
            # here the training indexes are obtained with some randomization
            segment_indices = self._get_train_indices(record)
        else:
            # here the testing indexes are obtained with no randomization, i.e., centered
            segment_indices = self._get_val_indices(record)

        #print(segment_indices)

        emgs, label = self.get(record, segment_indices)
        emgs = np.array(emgs)
        #logger.info(f"------------------------------segment_indices(size = {len(segment_indices['RGB'])})------------------------------\n{segment_indices}\n------------------------------frames(size = {frames['RGB'].shape})------------------------------\n{frames['RGB'][0]}")


        if self.additional_info:
            return emgs, label, record.description, record.size
        else:
            return emgs, label

    def get(self, record, indices):
        emgs = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            emg = record.get_emg(p)
            emgs.extend(emg)
        # finally, all the transformations are applied
        # process_data = self.transform[modality](emgs)
        return emgs, record.label


    def __len__(self):
        return len(self.video_list)
    



#-------------------------------------------------------EMG + RGB-------------------------------------------------------------------------#



class ActionNetEmgRgbDataset(data.Dataset, ABC):
    def __init__(self, mode, num_frames_per_clip, num_clips, dense_sampling, annotations_dir, data_dir, frame_dir, stride, transform=None, **kwargs):

        """
        mode: str (train, test/val)
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        """
        self.mode = mode  # 'train', 'val' or 'test'
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = stride
        self.transform = transform  # pipeline of transforms
        self.additional_info = False
        self.frame_dir = frame_dir
        if self.mode == "train":
            pickle_name = "ActionNet_train.pkl"
        else:
            pickle_name = "ActionNet_test.pkl"

        list_action = pd.read_pickle(os.path.join(annotations_dir, pickle_name))
        #logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        # A single EpicVideoRecord is a single clip,
        # example of tup is :
        file = ['S00_2.pkl' ,'S01_1.pkl', 'S02_2.pkl', 'S02_3.pkl', 'S02_4.pkl', 'S03_1.pkl', 'S03_2.pkl', 'S04_1.pkl', 'S05_2.pkl', 'S06_1.pkl', 'S06_2.pkl', 'S07_1.pkl', 'S08_1.pkl', 'S09_2.pkl']
        self.video_list = []
        for f in file:
            subject = list_action[list_action['file'] == f]
            index = np.array(subject['index']) - 1
            file_path = os.path.join(data_dir, f)
            df = pd.read_pickle(file_path)
            #print(index)
            df_idx = df.iloc[index]
            for (_,action) in df_idx.iterrows():
                descr = action['description']
                label = action['description_class']
                #start = action['start_time']
                #end = action['end_time']
                for el in action['emg_data']:
                    sample = ActionNetVideoEmgRecord(descr,label, el)
                    self.video_list.append(sample)

        rgb = pd.read_pickle(os.path.join(self.frame_dir, "timestamps.pkl"))
        self.video_list = [sample.combine_emg_rgb(rgb) for sample in self.video_list]


        #mappa_filtrata =list(zip(self.list_file["narration"][283:287],self.list_file["verb"][283:287],self.list_file["verb_class"][283:287]))
        #logger.info(mappa_filtrata)


    def _get_train_indices(self, record):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################

        num_frames_record = record.size #100
        num_frames_per_clip = self.num_frames_per_clip #20
        num_clips = self.num_clips #5
        stride = self.stride

        # offset da sommare ad ogni index per centrare la clip
        centroid_offset = num_frames_record / num_clips / 2 - ( num_frames_per_clip * stride / 2 )
        segment_dim = num_frames_record / num_clips

        all_indices = []

        # --- dense sampling ---
        if self.dense_sampling:
            for _ in range(num_clips):
                # prende un punto centrale randomico assicurandosi un certo offset dall'inizio e dalla fine
                central_point = np.random.randint(centroid_offset, num_frames_record-centroid_offset)
                # starting index of the clip, 0 in case of negative values
                start_idx = max(0, central_point - num_frames_per_clip * stride / 2)
                indices = [(idx * stride + start_idx) % num_frames_record for idx in range(num_frames_per_clip)]
                indices.sort()
                all_indices += indices            
            
        # --- uniform sampling ---
        else:
            average_duration = num_frames_record // self.num_frames_per_clip
            if average_duration > 0:
                frame_idx = np.multiply(np.arange(self.num_frames_per_clip), average_duration) + \
                            np.random.randint(average_duration, size=self.num_frames_per_clip)
                all_indices = np.tile(frame_idx, self.num_clips)
            else:
                all_indices = np.zeros((self.num_frames_per_clip * self.num_clips,))

        return all_indices

        # logger.info("----------------------")
        # logger.info(f"num_frames : {num_frames_record}, indeces : {all_indices}")
        # logger.info("----------------------")


    def _get_val_indices(self, record):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        
        num_frames_record = record.size #100
        num_frames_per_clip = self.num_frames_per_clip #20
        num_clips = self.num_clips #5
        stride = self.stride

        # offset da sommare ad ogni index per centrare la clip
        centroid_offset = num_frames_record / num_clips / 2 - ( num_frames_per_clip * stride / 2 )
        segment_dim = num_frames_record / num_clips

        all_indices = []

        # --- dense sampling ---
        if self.dense_sampling:
            #logger.info("___________________________________________________DENSE SAMPLING___________________________________________________")
            # prende il max index possibile per far stare la clip
            max_idx = max(0, num_frames_record - segment_dim)
            
            # indici iniziali di ogni segmento centrati al centroide del segmento
            clips_start_idx = np.linspace(0, max_idx, num=num_clips, dtype=float)
            segment_dim = clips_start_idx[1]
            # caso in cui i segmenti non si overlappano, centralizza il segmento
            if segment_dim > num_frames_per_clip*stride:
                clips_start_idx += centroid_offset

            for start_idx in clips_start_idx:
                indices = [(idx * stride +start_idx) % num_frames_record for idx in range(num_frames_per_clip)]
                indices.sort()
                all_indices += indices

        # --- uniform sampling ---
        else:
            #logger.info("___________________________________________________UNIFORM SAMPLING___________________________________________________")

            # prende il max index possibile per far stare la clip
            max_idx = max(0, num_frames_record - segment_dim)
            # indici iniziali di ogni segmento centrati al centroide del segmento
            clips_start_idx = np.linspace(0, max_idx, num=num_clips, dtype=int)
            segment_dim = clips_start_idx[1]
            for start_idx in clips_start_idx:
                all_indices += np.linspace(start_idx, start_idx + segment_dim, num=num_frames_per_clip, dtype=int).tolist()



        # logger.info("----------------------")
        # logger.info(f"num_frames : {num_frames_record}, indeces : {all_indices}")
        # logger.info("----------------------")
        
        return all_indices

    # item si riferisce alla singola clip
    def __getitem__(self, index):

        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]
        if self.mode == "train":
            # here the training indexes are obtained with some randomization
            segment_indices = self._get_train_indices(record)
        else:
            # here the testing indexes are obtained with no randomization, i.e., centered
            segment_indices = self._get_val_indices(record)

        #print(segment_indices)

        emgs, images, label = self.get(record, segment_indices)
        emgs = np.array(emgs)
        #logger.info(f"------------------------------segment_indices(size = {len(segment_indices['RGB'])})------------------------------\n{segment_indices}\n------------------------------frames(size = {frames['RGB'].shape})------------------------------\n{frames['RGB'][0]}")


        if self.additional_info:
            return emgs, images, label, record.description, record.size
        else:
            return emgs, images, label

    def get(self, record, indices):
        emgs = list()
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            emg, frame_name = record.get_emg_rgb(p)
            emgs.extend(emg)
                
            frame = self._load_data(frame_name)
            images.extend(frame)
        # finally, all the transformations are applied
        # process_data = self.transform[modality](emgs)
        process_data = self.transform(images)
        return emgs, process_data, record.label

    def _load_data(self, frame_name):
        data_path = self.frame_dir
        # here the offset for the starting index of the sample is added
        idx_untrimmed = frame_name
        formatted_str = "frame_%010d.jpg" % idx_untrimmed

        try:
            img = Image.open(os.path.join(data_path, formatted_str)) \
                .convert('RGB')
        except FileNotFoundError:
            print("Img not found")
            raise FileNotFoundError
        return [img]

        
    def __len__(self):
        return len(self.video_list)