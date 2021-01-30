# Introduction
This repository contains code for the paper "Perceiving More Truth: A Dilated-Block-Based Convolutional Network for Rumor Identification" By Yue Yuan, Yanli Wang, Kan Liu, available via https://pan.baidu.com/s/1nPi0_NZKBZBAu5mm3K4zwA Extraction code: scex
# Requirements
python 3.7.0  
tensorflow==1.4.0  
# How to use
## Dataset
The raw dataset is collected by Ma et al. (2016) and can bed downloaded from .zip includes nflod, resource, twitter15 and twitter16 folders. This dataset collected by Ma et al. (2018), and the raw datasets can be downloaded from [here](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0):  
Jing Ma, Wei Gao, Kam-Fai Wong. Rumor Detection on Twitter with Tree-structured Recursive Neural Networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018.  
The ind_twitter15.graph, ind_twitter15.features, ind_twitter15.poster, ind_twitter16.graph, ind_twitter16.features, and ind_twitter16.poster files are the propocessed data of user behavious graph on datasets twitter15 and twitter16, respectively.  
## Training & Testing
python main.py --dbcn --train -- # training the DBCN model, the trained model is saved in 
python Main_TD_RvNN_GCN.py # testing the TD-Hybrid model
