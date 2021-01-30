import os
import numpy as np
from gensim.models.doc2vec import Doc2Vec


class DataPreprocessor:
    '''This class is created for process and extracted data.

    There are four  main functions in class DataPreprocessor:
    function 1. __transformDataIntoXY() is to transform text data into matrix by a trained doc2vec model.
    function 2. getTrainData() is for obtaining train data.
    function 3. getDevData() is for obtaining development data.
    function 4. getTestData() is for obtaining test data.
    '''
    para_embedding_size = 56
    group_num = 20

    def __init__(self, trainFilePath="./Data/Train/", testFilePath="./Data/Test/",
                 devFilePath="./Data/Dev/", doc2vecModelPath="./trained models/d2v/d2v.model",
                 logFilePath="./Data/log.txt"):
        '''
        Constructor of class DataPreprocessor.

        All parameters have default values so you do not need to clarify them in the constructor.

        :param trainFilePath: A `str` representing the directory that store training files.
        :param testFilePath: A `str` representing the directory that store testing files.
        :param devFilePath: A `str` representing the directory that store developing files.
        :param doc2vecModelPath: A `str` representing the path of the trained doc2vec model files.
        :param logFilePath: A `str` representing the path of the log file which is important for indicating the relationship between text (training,testing,developing) and doc2vec vectors.
        '''
        self.trainFilePath = trainFilePath
        self.testFilePath = testFilePath
        self.devFilePath = devFilePath
        self.doc2vecModelPath = doc2vecModelPath
        self.logFilePath = logFilePath

    def __transformDataIntoXY(self, dataDir):
        '''
        transform text data into input matrix with the shape of (file num, group_num, embedding size)
                             and label matrix with the shape of (file num, 2).
        :param dataDir: data directory (training data,testing data or developing data)
        :return: np.array(X) A `ndarry` representing input matrix X.
                 np.array(Y) A `ndarry` representing label matrix Y.
                 fileNameNoDict A `dict` representing the relationship between matrix and files.
        '''
        X = []
        Y = []
        model = Doc2Vec.load(self.doc2vecModelPath)
        logDict = {}
        fileNameNoDict = {}
        logFileLines = open(self.logFilePath, mode="r",
                            encoding="utf-8").readlines()
        for lineCounter in range(len(logFileLines)):
            logDict[logFileLines[lineCounter].replace("\n", "")] = lineCounter
        for root, dirs, files in os.walk(dataDir):
            fileCounter = 0
            for fileName in files:
                fileDir = dataDir + fileName
                file = open(fileDir, mode="r", encoding="utf-8")
                fileLines = file.readlines()
                if len(fileLines) >= 1:
                    flag = int(fileLines[0].split("\t")[1])
                    fileNameNoDict[fileCounter] = str(fileName) + "-" + str(flag)
                    if flag == 0:
                        Y.append(np.array([float(1), float(0)]))
                    elif flag == 1:
                        Y.append(np.array([float(0), float(1)]))
                    currentFileVectors = []
                    for lineCounter in range(len(fileLines)):
                        if fileLines[lineCounter].split("\t")[2] != "\n":
                            keyStr = fileName + "\t" + str(lineCounter)
                            index = logDict[keyStr]
                            currentFileVectors.append(model[index])
                        elif fileLines[lineCounter].split("\t")[2] == "\n":
                            currentFileVectors.append(
                                np.array([float(0.0) for i in range(DataPreprocessor.para_embedding_size)]))
                    X.append(np.array(currentFileVectors))
                else:
                    fileNameNoDict[fileCounter] = str(fileName) + "-" + str(1)
                    currentFileVectors = []
                    for i in range(DataPreprocessor.group_num):
                        currentFileVectors.append(
                            np.array([float(0.0) for i in range(DataPreprocessor.para_embedding_size)]))
                    X.append(np.array(currentFileVectors))
                    Y.append(np.array([float(0), float(1)]))
                fileCounter = fileCounter + 1
        return np.array(X), np.array(Y), fileNameNoDict

    def getTrainData(self):
        '''
        get train data.

        :return: Training data
        '''
        print("Getting Train Data!")
        trainX, trainY, trainFileNameNoDict = self.__transformDataIntoXY(self.trainFilePath)
        print("Shape:")
        print("trainX shape:" + str(trainX.shape))
        print("trainY shape:" + str(trainY.shape))
        return trainX, trainY, trainFileNameNoDict

    def getDevData(self):
        '''
        get develop data.

        :return: Developing data
        '''
        print("Getting Dev Data!")
        devX, devY, devFileNameNoDict = self.__transformDataIntoXY(self.devFilePath)
        print("Shape:")
        print("devX shape:" + str(devX.shape))
        print("devY shape:" + str(devY.shape))
        return devX, devY, devFileNameNoDict

    def getTestData(self):
        '''
        get test data.

        :return: Testing data
        '''
        print("Getting Test Data!")
        testX, testY, testFileNameNoDict = self.__transformDataIntoXY(self.testFilePath)
        print("Shape:")
        print("testX shape:" + str(testX.shape))
        print("testY shape:" + str(testY.shape))
        return testX, testY, testFileNameNoDict
