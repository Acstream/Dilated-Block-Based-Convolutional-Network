# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 15:50
# @Author  : Yuan Yue (Acstream) and Wang Yanli (lilijodie)
# @Email   : yuangyue@qq.com
# @File    : main.py

import sys
import DBCN_Model
import ODCN_Model
import tensorflow as tf
import DataPreprocessor

if __name__ == '__main__':
    modelName=str(sys.argv[1])
    trainOrTest=str(sys.argv[2])
    filePath=str(sys.argv[3])

    sess = tf.Session()
    dataPreprocessor = DataPreprocessor.DataPreprocessor()
    if trainOrTest=="--train":
        if modelName=="--dbcn":
            print("Start Training DBCN!")
            dbcn=DBCN_Model.DBCN_Model()
            if filePath=="--":
                DBCN_Model.DBCN_Model.modelSavingPath="./trained models/new networks/dbcn/"
            else:
                DBCN_Model.DBCN_Model.modelSavingPath=filePath
            dbcn.modelSavingPath=filePath
            dbcn.train(sess,dataPreprocessor)
            print("DBCN Training Completed!")
        elif modelName=="--odcn":
            print("Start Training ODCN!")
            odcn = ODCN_Model.ODCN_Model()
            if filePath == "--":
                ODCN_Model.ODCN_Model.modelSavingPath = "./trained models/new networks/odcn/"
            else:
                ODCN_Model.ODCN_Model.modelSavingPath = filePath
            odcn.modelSavingPath = filePath
            odcn.train(sess, dataPreprocessor)
            print("ODCN Training Completed!")
    elif trainOrTest=="--test":
        if modelName=="--dbcn":
            print("Start Testing DBCN!")
            dbcn=DBCN_Model.DBCN_Model()
            if filePath=="--":
                DBCN_Model.DBCN_Model.modelSavingPath="./trained models/networks/dbcn/"
            else:
                DBCN_Model.DBCN_Model.modelSavingPath=filePath
            dbcn.modelSavingPath=filePath
            dbcn.test(sess,dataPreprocessor)
            print("DBCN Test Completed!")
        elif modelName=="--odcn":
            print("Start Testing ODCN!")
            odcn = ODCN_Model.ODCN_Model()
            if filePath == "--":
                ODCN_Model.ODCN_Model.modelSavingPath = "./trained models/networks/odcn/"
            else:
                ODCN_Model.ODCN_Model.modelSavingPath = filePath
            odcn.modelSavingPath = filePath
            odcn.test(sess, dataPreprocessor)
            print("ODCN Test Completed!")