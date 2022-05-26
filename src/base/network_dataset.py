from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import os
import torch
import numpy as np

from tqdm import tqdm



def read_file(path):
  '''read hdfs files from disk 
  , split logs  ;=> split label,
  format is  
  22 5 5 5 11 9 11 9 26 26 11 9 26\t1

  return "22 5 5 5 11 9 11 9 26 26 11 9 26" , 1
  '''
  datas = []
  edg = []

  with open(path,'r') as f:
    for line in f:
      # LDAP remove id
      id = line.split()[0]
      line = line.replace(id+" ","",1)

      tmp = line.split("\t")
      # if don't give a label
      if len(tmp)<2:
          tmp.append(0)
      
      a = len(tmp[0].split(" "))
      #if a < 100:
      datas.append((tmp[0],int(tmp[1])))
      
    return datas

def logkey_to_int_graph(logkey, model):

    arr =np.zeros(64)

    keys = logkey.strip().split(" ")
   
    for key in keys:
        arr += model.wv[key]
    
    arr = arr/len(keys)
    
    graph = torch.tensor(arr, dtype=torch.float32)
    

    return graph
     



        

class networkseqLogsDataset(Dataset):
    """
    LogDataset class for log dataset which preprocess by Drain3

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """


    def __init__(self, root: str, dataset_name: str, file_name: str, network_model, train=True, random_state=None):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        self.root = root
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        self.file_name = file_name
        self.network_model= network_model

        # load data  len = 64
        #train_file = read_file(root+'/'+dataset_name+'/hdfs_train_labl')
        #train_file = read_file(root+'/'+dataset_name+'/hdfs_train_labl_replace')
        #test_file = read_file(root+'/'+dataset_name+'/hdfs_test_labl')
        #test_file = read_file(root+'/'+dataset_name+'/hdfs_test_labl_replace')

        # hdfs toy data
        # train_file = read_file(root+'/'+dataset_name+'/train.txt')
        # test_file = read_file(root+'/'+dataset_name+'/vaild.txt')

        # ldap don't forget to remove id 
        #train_file  = read_file(root+'/'+dataset_name+'/logkeys_drains_label_id_0_1')
        #train_file = read_file(root+'/'+dataset_name+'/logkeys_drains_label_id_0_1_replace')
        #test_file = read_file(root+'/'+dataset_name+'/logkeys_cross_359_01mP_m_conn_n_rmdate_with_id_0_1')
        #test_file = read_file(root+'/'+dataset_name+'/logkeys_cross_1022add80_01mP_m_conn_n_rmdate_with_id_0_1')
        #test_file = read_file(root+'/'+dataset_name+'/logkeys_cross_1022add80_01mP_m_conn_n_rmdate_with_id_0_1_replace')

        # 
        #X_test = [ logkey_to_int_graph(X) for X, y in test_file]

    
    def init_file(self):
        if self.train:
            train_file  = read_file(self.root+'/'+self.dataset_name+'/'+self.file_name)
        else:
            test_file = read_file(self.root+'/'+self.dataset_name+'/'+self.file_name)
        

        if self.train:
            X_train = [ logkey_to_int_graph(X, self.network_model) for X, y in train_file]
            y_train = [ y for X, y in train_file]
            
            self.data = X_train
            #self.data = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            X_test = []
            for X, y in tqdm(test_file):
                X_test.append(logkey_to_int_graph(X, self.network_model))
            y_test = [y for X, y in test_file]
             
            self.data = X_test 
            #self.data = torch.tensor(X_test, dtype=torch.float32) 
            self.targets = torch.tensor(y_test, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), self.semi_targets[index]

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)

    def replace_target(self, normal):
        tmp_arr = []
        for i in self.targets:
            if i in normal:
                tmp_arr.append(0)
            else:
                tmp_arr.append(1)
        self.targets = tmp_arr
        
