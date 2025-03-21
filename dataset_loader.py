import torch
import torch.utils.data as data
import os
import numpy as np
import utils 

class Drone_Anomaly(data.DataLoader):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list','DA-i3d-{}.list'.format(self.mode))
        
        self.list = list(open(split_path, 'r'))
        # self.list = []
        # for line in split_file:
        #     self.list.append(line.split())
        # split_file.close()
        
        if self.mode == "Train":
            index_n = [i for i, item in enumerate(self.list) if 'label_0' in item]
            index_a = [i for i, item in enumerate(self.list) if 'label_1' in item]

            if is_normal is True:
                # self.list = self.list[8100:]
                self.list = [self.list[i] for i in index_n]
                
            elif is_normal is False:
                # self.list = self.list[:8100]
                self.list = [self.list[i] for i in index_a]

            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.list=[]
    

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        
        # if self.mode == "Test":
        #     data,label,name = self.get_data(index)
        #     return data,label,name
        # else:
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_info = self.list[index].strip('\n')
        vid_name = vid_info.split("/")[-1]
        video_feature = np.load(vid_info).astype(np.float32) 
        video_feature = video_feature.transpose(1, 0, 2) # (10, 37, 2048)
        # breakpoint() 

        if "label_0" in vid_name:
            label = 0
        else:
            label = 1
        
        if self.mode == "Train":
            # new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            # r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = int)
            # for i in range(self.num_segments):
            #     if r[i] != r[i+1]:
            #         new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
            #     else:
            #         new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
            # video_feature = new_feat

            divided_features = []
            for feature in video_feature:
                # breakpoint()
                # feature.shape = (34, 2048)
                feature = utils.process_feat(feature, 32)   # divide a video into 32 segments (T=32)
                divided_features.append(feature)
            
            divided_features = np.array(divided_features, dtype=np.float32)  # (10, 32, 2048)
            video_feature = divided_features
            # print(f'After split into snippets: shape {video_feature.shape}')
            

        if self.mode == "Test":
            return video_feature, label    
        else:
            return video_feature, label  


class UCF_crime(data.DataLoader):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list','UCF_{}.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.list = []
        for line in split_file:
            self.list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.list = self.list[8100:]
            elif is_normal is False:
                self.list = self.list[:8100]
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.list=[]
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data,label,name = self.get_data(index)
            return data,label,name
        else:
            data,label = self.get_data(index)
            return data,label

    def get_data(self, index):
        vid_info = self.list[index][0]  
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)   

        if "Normal" in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = np.int)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label



class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.modal=modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        if self.modal == 'all':
            self.feature_path = []
            if self.mode == "Train":
                for _modal in ['RGB', 'Flow']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
            else:
                for _modal in ['RGBTest', 'FlowTest']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
        else:
            self.feature_path = os.path.join(self.data_path, modal)
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.list = []
        for line in split_file:
            self.list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.list = self.list[9525:]
            elif is_normal is False:
                self.list = self.list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.list=[]
        
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        data,label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.list[index][0]
        label=0
        if "_label_A" not in vid_name:
            label=1  
        video_feature = np.load(os.path.join(self.feature_path[0],
                                vid_name )).astype(np.float32)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments,self.len_feature)).astype(np.float32)
            sample_index = utils.random_perturb(video_feature.shape[0],self.num_segments)
            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
        return video_feature, label    
