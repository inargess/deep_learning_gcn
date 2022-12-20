# Overview
Conventional deep learning techniques such as Long short-term memory (LSTM) and Convolutional
Neural Network (CNN) 
typically ignore the connectivity and strength
of the connectivity between electrode pairs. In this project, I proposed a GCN model that considers its connectivity and strength. Furthermore, the proposed model utilizes
CNN's spatial feature extraction power to decode the underlying neurophysiological features
from the motor imagery tasks.
# Dataset
For this work, I used PhysioNet EEG Motor Movement/Imagery dataset. This dataset includes 64-channel EEG signals collected at a sample rate of 160 Hz from 109 healthy subjects who performed six different tasks in the 14 experimental runs. These tasks are eye open, eye closed, left or right fist open and closed, imagine left or right fist open and closed, both fists or feet open and closed, and imagine both fists or feet open and closed. Subjects performed the last four tasks by following the screen display in the experimental room. 

In the first two runs, subjects performed eye-open and closed tasks for one minute each. The subjects repeatedly performed the remaining four tasks (two minutes each, including rest periods in between trials) in three different
repetitions (rep 1: run 3-6, rep 2: run 7-10, and rep 3: run 11-14). There are multiple trials in each run;
labels T0, T1, and T2 respectively correspond to rest, the left fist (in runs 3, 4, 7, 8, 11, 12) or both fists (in
runs 5, 6, 9, 10, 13, 14), and the right fist (in runs 3, 4, 7, 8, 11, 12) or both feet (in runs 5, 6, 9, 10, 13,
14). I excluded eye-open and closed tasks in this project, as they were labeled as rest (T0). It is worth noting that the subjects' demographic information is not used to avoid bias in the model.

# Model
To implement the GCN model, I used PyTorch Geometrics package, in which a GCN model takes up to three inputs: an adjacency matrix in coordinate format, a feature matrix, and an edge attribute matrix. 
- Adjacency matrix: The adjacency matrix of each trial is a fully connected matrix excluding self-connections.
- Edge attribute matrix: stores edge weights from each trial. The Spearman correlation coefficient is used to compute pairwise comparisons between electrodes.
comparison between electrodes
- Feature matrix: temporal, spectral, and connectivity features from the normalized EEG signals of the trial for each node are extracted. Resulting in $64X25$ matrix for each trial. 

Chosen GCN model contains of three Chebyshev spectral graph convolutional operators, in which each operator
contains a Chebyshev convolutional (ChebConv) layer, a Batch Normalization layer, and
a ReLU layer. The GCN has $32, 64, 128$ filters and $3, 4, 5$ filter sizes on each ChebConv layer. Lastly, the concatenated outputs from maximum global pooling and global average pooling are passed to a 256 cells dense layer followed by a Log Softmax activation. This would help to obtain the predicted probability of each task. 

The model is designed to store the best model, meaning if the model accuracy does not improve for 80 consecutive iterations, the previously trained model will be used. Model training had a mini-batch size of 32 and a learning rate of $4e^{-4}$ on the Adam optimizer. Furthermore, weight decay regularization and early stopping techniques prevent over-fitting issues. Finally, cross-entropy is
considered to be the loss function.
