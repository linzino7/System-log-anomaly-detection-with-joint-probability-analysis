System-log-anomaly-detection-with-join-histogram-analysis
========

![Overview of Our method](https://raw.githubusercontent.com/linzino7/System-log-anomaly-detection-with-join-histogram-analysis/main/overview.PNG)


System anomaly detection is a critical problem to construct a secure and stable information system. Most anomaly detection methods detect anomalous activities through analyzing numerous system logs recorded during system running. In recent years, several deep learning-based methods have been proposed for system log anomaly detection. However, those methods may incorrectly detect an anomaly from a single event log with long length. Therefore, we proposed a method to convert long logs into shorter representations via the joint histogram analysis. Anomalous activities then can be recognized for long logs with multiple-input multiple-output Autoencoder. We also use different datasets of various systems to demonstrate that the proposed method produced a superior performance than the previous deep learning-based methods.

# Dataset
### HDFS 
HDFS original log data: https://github.com/logpai/loghub/tree/master/HDFS

### BGL
BGL original log data: https://github.com/logpai/loghub/tree/master/BGL 

### LDAP
Due to the Non-disclosure agreement, we could not public dataset. However, we offer preprocessed log key sequences.

### Preprocessed data 
we used same Preprocessing method with LogBERT in HDFS and BGL datasets.

Preprocessed data in this work can download by [here](https://drive.google.com/file/d/1S9REkg2aONADkz9Vv-TqLrKxaMjCAgO1/view?usp=sharing).


# Experiment
## environment
* CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz
* RAM: 64G
* GPU: Nvidia Tesla P100-PCIE
* Other: python 3.6.9 and pytorch 1.7.1


## Training
### Our Method
#### LDAP
environment result show in [here](https://github.com/linzino7/System-log-anomaly-detection-with-join-histogram-analysis/blob/main/log/DeepSAD/LDAP_MIMO_conv_mlp/log.txt)

```
mkdir log/DeepSAD/LDAP_MIMO_conv_mlp
python3 src/main.py LDAP LDAP_MIMO_conv_mlp log/DeepSAD/LDAP_MIMO_conv_mlp data --lr 0.001  --n_epochs 30 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True   --ae_lr 0.001 --ae_n_epochs 100 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --n_known_outlier_classes 1  --ratio_known_normal 0.01 --seed 2
```

#### HDFS
```
mkdir log/DeepSAD/HDFS_MIMO_conv_mlp
python3 src/main.py HDFS HDFS_MIMO_conv_mlp log/DeepSAD/HDFS_MIMO_conv_mlp data --lr 0.001  --n_epochs 30 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True   --ae_lr 0.001 --ae_n_epochs 100 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --n_known_outlier_classes 1  --ratio_known_normal 0.01 --seed 0
```

#### BGL
```
mkdir log/DeepSAD/BGL_MIMO_conv_mlp
python3 src/main.py BGL BGL_MIMO_conv_mlp log/DeepSAD/BGL_MIMO_conv_mlp data --lr 0.001  --n_epochs 50 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True   --ae_lr 0.001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --n_known_outlier_classes 1  --ratio_known_normal 0.01 --seed 1
```


### Deep SAD
#### LDAP
```
mkdir log/DeepSAD/LDAP_top64_MLP
python3 src/main.py LDAP LDAP_mlp log/DeepSAD/LDAP_top64_MLP data --lr 0.001  --n_epochs 100 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True   --ae_lr 0.001 --ae_n_epochs 130 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --n_known_outlier_classes 1  --ratio_known_normal 0.01  --seed 0
```

#### HDFS
```
mkdir log/DeepSAD/HDFS_mlp
python3 src/main.py HDFS HDFS_mlp log/DeepSAD/HDFS_mlp data --lr 0.0001  --n_epochs 200 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True   --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --n_known_outlier_classes 1  --ratio_known_normal 0.01 --seed 0
```

#### BGL
```
mkdir log/DeepSAD/BGL_top64_MLP
python3 src/main.py BGL BGL_mlp log/DeepSAD/BGL_top64_MLP data --lr 0.001  --n_epochs 100 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True   --ae_lr 0.001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --n_known_outlier_classes 1  --ratio_known_normal 0.01 --seed 0
```

### DeepLog
Our result implemented by https://github.com/wuyifan18/DeepLog

### LogBERT
Our result implemented by https://github.com/HelenGuohx/logbert

## Evaluation
python3 src/measueAUC.py [log_folder_path]

```
python3 src/measueAUC.py log/DeepSAD/LDAP_MIMO_conv_mlp
```

# Reference
This is repository modifying some part of the original Deep SAD [code](https://github.com/lukasruff/Deep-SAD-PyTorch).


[1] Min Du, Feifei Li, Guineng Zheng, and Vivek Srikumar. Deeplog: Anomaly detection and169
diagnosis from system logs through deep learning. In Proceedings of the 2017 ACM SIGSAC170
conference on computer and communications security, pages 1285–1298, 2

[2] Lukas Ruff, Robert A Vandermeulen, Nico Görnitz, Alexander Binder, Emmanuel Müller,194
Klaus-Robert Müller, and Marius Kloft. Deep semi-supervised anomaly detection. arXiv195
preprint arXiv:1906.02694, 201

[3] Haixuan Guo, Shuhan Yuan, and Xintao Wu. Logbert: Log anomaly detection via bert. In 2021177
International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 202

### Bibtex

```
@inproceedings{du2017deeplog,
  title={Deeplog: Anomaly detection and diagnosis from system logs through deep learning},
  author={Du, Min and Li, Feifei and Zheng, Guineng and Srikumar, Vivek},
  booktitle={Proceedings of the 2017 ACM SIGSAC conference on computer and communications security},
  pages={1285--1298},
  year={2017}
}
@article{ruff2019deep,
  title={Deep semi-supervised anomaly detection},
  author={Ruff, Lukas and Vandermeulen, Robert A and G{\"o}rnitz, Nico and Binder, Alexander and M{\"u}ller, Emmanuel and M{\"u}ller, Klaus-Robert and Kloft, Marius},
  journal={arXiv preprint arXiv:1906.02694},
  year={2019}
}
@inproceedings{guo2021logbert,
  title={Logbert: Log anomaly detection via bert},
  author={Guo, Haixuan and Yuan, Shuhan and Wu, Xintao},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}

```


