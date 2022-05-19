# System-log-anomaly-detection-with-join-histogram-analysis

![Overview of Our method](https://raw.githubusercontent.com/linzino7/System-log-anomaly-detection-with-join-histogram-analysis/main/overview.PNG)


System anomaly detection is a critical problem to construct a secure and stable information system. Most anomaly detection methods detect anomalous activities through analyzing numerous system logs recorded during system running. In recent years, several deep learning-based methods have been proposed for system log anomaly detection. However, those methods may incorrectly detect an anomaly from a single event log with long length. Therefore, we proposed a method to convert long logs into shorter representations via the joint histogram analysis. Anomalous activities then can be recognized for long logs with multiple-input multiple-output Autoencoder. We also use different datasets of various systems to demonstrate that the proposed method produced a superior performance than the previous deep learning-based methods.

# Dataset

# Reference
This is repository modifying some part of the original Deep SAD [code](https://github.com/lukasruff/Deep-SAD-PyTorch).
