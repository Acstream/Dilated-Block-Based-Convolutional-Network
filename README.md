# Introduction
This repository contains code for our paper "Perceiving More Truth: A Dilated-Block-Based Convolutional Network for Rumor Identification" By Yue Yuan, Yanli Wang, Kan Liu, available via https://pan.baidu.com/s/1nPi0_NZKBZBAu5mm3K4zwA Extraction code: scex
# Requirements
python==3.6.5  
tensorflow==1.8.0  
gensim==3.8.3
numpy==1.19.4
# How to use
## Dataset
The raw dataset is collected by Ma et al. ([J. Ma, W. Gao, P. Mitra, S. Kwon, B. J. Jansen, K.-F. Wong, M. Cha, Detecting rumors from microblogs with recurrent neural networks, in: Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, 2016, pp. 3818–3824](https://www.ijcai.org/Proceedings/16/Papers/537.pdf)) and can be downloaded from [here](https://alt.qcri.org/~wgao/data/rumdect.zip). 
The raw dataset has been processed according to Section 3.2 Data prepraration in our paper and is stored in ./Data (512 topics: 307 for training, 51 for developing, 154 for testing).   
Therefore, you can use the data in ./Data directly without performing the data prepraration process described in our paper.   
## Training & Testing
We recommend you to use our trained models to test and evaluate:
python main.py --dbcn --test -- # test the DBCN model by our trained model saved in ./trained models/networks/dbcn/
python main.py --odcn --test -- # test the ODCN model by our trained model saved in ./trained models/networks/odcn/
However, if you want to train your own models, you can type: 
python main.py --dbcn --train -- # train the DBCN model, your trained model is defaultly saved in ./trained models/new networks/dbcn/
python main.py --odcn --train -- # train the ODCN model, your trained model is defaultly saved in ./trained models/new networks/odcn/
or
python main.py --dbcn --train YOURPATH # train the DBCN model, your trained model is saved in ./trained models/new networks/dbcn/
python main.py --odcn --train YOURPATH # train the ODCN model, your trained model is saved in ./trained models/new networks/odcn/
to obtain your own trained models
