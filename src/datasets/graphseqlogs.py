from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.graphseq_dataset import graphseqLogsDataset
from .preprocessing import create_semisupervised_setting

import torch

class graphseqLogsSADDataset(graphseqLogsDataset):
    ''' build log data loader'''

    def __init__(self, root: str, dataset_name: str, net_name:str, n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None):
        super().__init__(root, dataset_name, net_name) # LogDataset __init__

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,) #(1,0)
        self.outlier_classes = (1,)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = (1,)

        # Get train set
        train_set = graphseqLogsDataset(root=self.root, dataset_name=dataset_name, net_name=net_name, train=True, random_state=random_state)

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
        self.test_set = graphseqLogsDataset(root=self.root, dataset_name=dataset_name, net_name=net_name, train=False, random_state=random_state)
        self.test_set.replace_target(self.normal_classes)


    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
