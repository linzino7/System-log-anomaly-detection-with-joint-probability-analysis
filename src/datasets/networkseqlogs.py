from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.network_dataset import networkseqLogsDataset
from .preprocessing import create_semisupervised_setting

import torch

import networkx as nx
from node2vec import Node2Vec

def get_edge(path, rm_id=True):
  edg = []

  with open(path,'r') as f:
    for line in f:
      # LDAP remove id
      if rm_id:
        id = line.split()[0]
        line = line.replace(id+" ","",1)

      tmp = line.split("\t")

      # edge
      keys = tmp[0]
      for idx in range(0,len(keys)-1):
            f = keys[idx]
            t = keys[idx+1]
            pair = (f,t)
            if pair not in edg:
                edg.append(pair)

    return  edg

def init_edge(train,test,rm_id=True):
    train_edge = get_edge(train, rm_id)
    test_edge = get_edge(test, rm_id)

    return train_edge+test_edge


class networkseqLogsSADDataset(networkseqLogsDataset):
    ''' build log data loader'''

    def __init__(self, root: str, dataset_name: str, n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None):
        super().__init__(root, dataset_name, '' , '') # LogDataset __init__
        
        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,) #(1,0)
        self.outlier_classes = (1,)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = (1,)

       

        # HDFS
        train_filename = 'hdfs_train_labl'
        test_filename = 'hdfs_test_labl'

        #train_filename  = 'logkeys_drains_label_id_0_1'
        #train_file = read_file(root+'/'+dataset_name+'/logkeys_drains_label_id_0_1_replace')
        #test_filename = 'logkeys_cross_359_01mP_m_conn_n_rmdate_with_id_0_1'
        #test_filename = 'logkeys_cross_1022add80_01mP_m_conn_n_rmdate_with_id_0_1'
        #test_file = read_file(root+'/'+dataset_name+'/logkeys_cross_1022add80_01mP_m_conn_n_rmdate_with_id_0_1_replace')
        
        length = 64

        #init graph
        G=nx.Graph()
        for i in range(0,length):
            G.add_node(str(i))
        
        edg = init_edge(root+'/'+dataset_name+'/'+train_filename,
                        root+'/'+dataset_name+'/'+test_filename, rm_id=True)
        G.add_edges_from(edg)

        # init node2vec
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=1)  # Use temp_folder for big graphs
        model = node2vec.fit(window=3, min_count=1, batch_words=4)

        # Get train set
        train_set = networkseqLogsDataset(root=self.root, dataset_name=dataset_name, file_name = train_filename,
                                          network_model=model, train=True, random_state=random_state)
        train_set.init_file()

        print('normal_classes:' ,self.normal_classes)
        print('outlier_classes:' ,self.outlier_classes)
        print('known_outlier_classes:' ,self.known_outlier_classes)
        print('ratio_known_normal:',ratio_known_normal)
        print('ratio_known_outlier:',ratio_known_outlier)
        print('ratio_pollution:',ratio_pollution)

        # Create semi-supervised setting
        idx, _ , semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        
        print('len of idx:' , len(idx))
        # print('idx:' , idx)
        p = 0 ; z = 0;n = 0;
        for i in semi_targets:
            if i == 1 : 
                p+=1
            elif i == -1:
                n+=1
            else:
                z+=1
        print('semi: labeled_nor None labeled_Outer -', p, z, n )
        print(' new semi_targets :',len(semi_targets))
        print('train_set.semi_targets',len(train_set.semi_targets))
        

        train_set.semi_targets[idx] = torch.tensor(semi_targets, dtype=torch.int64)  # set respective semi-supervised labels
        train_set.replace_target(self.normal_classes)

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = networkseqLogsDataset(root=self.root, dataset_name=dataset_name, file_name = test_filename,
                                              network_model=model, train=False, random_state=random_state)
        self.test_set.init_file()
        self.test_set.replace_target(self.normal_classes)


    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
