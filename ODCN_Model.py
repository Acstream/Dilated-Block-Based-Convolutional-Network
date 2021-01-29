import math
import DataPreprocessor
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ODCN_Model:
    modelSavingPath = "./trained models/networks/odcn/"

    def __init__(self, shuffle_size=20, batch_size=16,
                 kernel_width_DilatedBlock_1=3, kernel_num_DilatedBlock_1=6, dilation_rate_DilatedBlock_1=2,
                 kernel_width_DilatedBlock_2=4, kernel_num_DilatedBlock_2=4, dilation_rate_DilatedBlock_2=4,
                 output_units=2, dropout_rate=0.5,
                 initial_learning_rate=0.0005, epoch_num=250):
        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.para_embedding_size = DataPreprocessor.DataPreprocessor.para_embedding_size
        self.group_num = DataPreprocessor.DataPreprocessor.group_num
        self.kernel_width_DilatedBlock_1 = kernel_width_DilatedBlock_1
        self.kernel_num_DilatedBlock_1 = kernel_num_DilatedBlock_1
        self.dilation_rate_DilatedBlock_1 = dilation_rate_DilatedBlock_1
        self.kernel_width_DilatedBlock_2 = kernel_width_DilatedBlock_2
        self.kernel_num_DilatedBlock_2 = kernel_num_DilatedBlock_2
        self.dilation_rate_DilatedBlock_2 = dilation_rate_DilatedBlock_2
        self.output_units = output_units
        self.dropout_rate = dropout_rate
        self.initial_learning_rate = initial_learning_rate
        self.epoch_num = epoch_num

    def __init_weights(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

    def __odcnModel(self, sess):
        print("Building ODCN model!")
        inputs = tf.placeholder(tf.float32, shape=(None, self.group_num, self.para_embedding_size), name="inputs")
        labels = tf.placeholder(tf.float32, shape=(None, self.output_units), name="labels")
        dropout_training_flag = tf.placeholder(tf.bool, None, name="dropout_training_flag")
        batchnormalization_training_flag = tf.placeholder(tf.bool, None, name="batchnormalization_training_flag")

        w_dilatedblock_1 = self.__init_weights(
            shape=[self.kernel_width_DilatedBlock_1, self.para_embedding_size, 1, self.kernel_num_DilatedBlock_1],
            name="W_dilatedblock1")
        w_dilatedblock_2 = self.__init_weights(
            shape=[self.kernel_width_DilatedBlock_2, self.para_embedding_size, self.kernel_num_DilatedBlock_1,
                   self.kernel_num_DilatedBlock_2], name="W_dilatedblock2")

        with tf.name_scope("inputs"):
            inputs_reshape = tf.expand_dims(inputs, -1)
            inputs_unstack = tf.unstack(inputs_reshape, axis=2)

        with tf.name_scope("dilated_block_1"):
            convs1 = []
            w1_unstack = tf.unstack(w_dilatedblock_1, axis=1)
            for i in range(len(inputs_unstack)):
                conv1 = tf.nn.convolution(input=inputs_unstack[i], filter=w1_unstack[i], padding="VALID",
                                          dilation_rate=[self.dilation_rate_DilatedBlock_1])
                bn1 = tf.layers.batch_normalization(conv1, training=batchnormalization_training_flag)
                ac1 = tf.nn.relu(bn1)
                convs1.append(ac1)
            convres1 = tf.stack(convs1, axis=2)
            print("dilated block 1:" + str(convres1))

        with tf.name_scope("dilated_block_2"):
            convs2 = []
            convres1_unstack = tf.unstack(convres1, axis=2)
            w2_unstack = tf.unstack(w_dilatedblock_2, axis=1)
            for i in range(len(convres1_unstack)):
                conv2 = tf.nn.convolution(input=convres1_unstack[i], filter=w2_unstack[i], padding="VALID",
                                          dilation_rate=[self.dilation_rate_DilatedBlock_2])
                bn2 = tf.layers.batch_normalization(conv2, training=batchnormalization_training_flag)
                ac2 = tf.nn.relu(bn2)
                convs2.append(ac2)
            convres2 = tf.stack(convs2, axis=2)
            print("dilated block 2:" + str(convres2))

        with tf.name_scope("pool_flat_output"):
            poolres = tf.nn.max_pool(value=convres2, ksize=[1, int(convres2.shape[1]), 1, 1],
                                     strides=[1, 1, 1, 1], padding="VALID")
            print("pooling:" + str(poolres))
            flatres = slim.flatten(poolres)
            print("flat:" + str(flatres))
            dropoutres = tf.layers.dropout(inputs=flatres, rate=self.dropout_rate, training=dropout_training_flag)
            print("dropout:" + str(dropoutres))
            predictions = tf.layers.dense(inputs=dropoutres, units=self.output_units, activation=tf.nn.tanh)
            print("dense:" + str(predictions))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1)), tf.float32))
        train_optimization = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate).minimize(loss)

        return loss, acc, train_optimization, predictions, labels

    def train(self, sess, dataPreprocessor):

        trainX, trainY, trainFileNameNoDict = dataPreprocessor.getTrainData()
        devX, devY, devFileNameNoDict = dataPreprocessor.getDevData()
        testX, testY, testFileNameNoDict = dataPreprocessor.getTestData()
        trainDataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

        trainData = trainDataset.shuffle(self.shuffle_size).batch(self.batch_size).repeat()
        iterator = trainData.make_one_shot_iterator()
        next_iterator = iterator.get_next()
        iterations = math.ceil(trainX.shape[0] / self.batch_size)
        loss, acc, train_optimization, predictions, labels = self.__odcnModel(sess)
        previous_best_accuarcy = 0.0

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(self.epoch_num):
            for iteration in range(iterations):
                trainX_batch, trainY_batch = sess.run(next_iterator)
                _, trainLoss, trainAcc = sess.run([train_optimization, loss, acc],
                                                  feed_dict={"inputs:0": trainX_batch, "labels:0": trainY_batch,
                                                             "dropout_training_flag:0": True,
                                                             "batchnormalization_training_flag:0": False,
                                                             })
                testLoss, testAcc = sess.run([loss, acc],
                                             feed_dict={"inputs:0": testX, "labels:0": testY,
                                                        "dropout_training_flag:0": False,
                                                        "batchnormalization_training_flag:0": False,
                                                        })
                devLoss, devAcc = sess.run([loss, acc],
                                           feed_dict={"inputs:0": devX, "labels:0": devY,
                                                      "dropout_training_flag:0": False,
                                                      "batchnormalization_training_flag:0": False,
                                                      })
                print("Epoch:", '%03d' % (epoch + 1), "train loss=", "{:.9f}".format(trainLoss), "train acc=",
                      "{:.9f}".format(trainAcc),
                      "test loss=", "{:.9f}".format(testLoss), "test acc=", "{:.9f}".format(testAcc), "dev loss=",
                      "{:.9f}".format(devLoss), "dev acc=", "{:.9f}".format(devAcc))
                if testAcc > previous_best_accuarcy:
                    saver.save(sess, ODCN_Model.modelSavingPath + "odcn")
                    previous_best_accuarcy = testAcc
                    print("Saving current model!")

    def test(self, sess, dataPreprocessor):
        self.__odcnModel(sess)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(ODCN_Model.modelSavingPath))
        print("Loading model completed!")
        graph = tf.get_default_graph()
        testX, testY, testFileNameNoDict = dataPreprocessor.getTestData()
        feed_dict = {"inputs:0": testX, "dropout_training_flag:0": False, "batchnormalization_training_flag:0": False}
        denseOutput = graph.get_tensor_by_name("pool_flat_output/dense/Tanh:0")
        testY_Predict = sess.run(denseOutput, feed_dict)
        predict = sess.run(tf.argmax(testY_Predict, 1))
        real = sess.run(tf.argmax(testY, 1))
        TP_List = []
        FN_List = []
        TN_List = []
        FP_List = []
        for i in range(len(predict)):
            if predict[i] == real[i] and predict[i] == 1:
                TP_List.append(str(testFileNameNoDict[i]).split("-")[0])
            elif predict[i] != real[i] and predict[i] == 0 and real[i] == 1:
                FN_List.append(str(testFileNameNoDict[i]).split("-")[0])
            elif predict[i] == real[i] and predict[i] == 0:
                TN_List.append(str(testFileNameNoDict[i]).split("-")[0])
            elif predict[i] != real[i] and predict[i] == 1 and real[i] == 0:
                FP_List.append(str(testFileNameNoDict[i]).split("-")[0])
        print("TP num:" + str(len(TP_List)))
        print("FN num:" + str(len(FN_List)))
        print("TN num:" + str(len(TN_List)))
        print("FP num:" + str(len(FP_List)))
        return TP_List, FN_List, TN_List, FP_List


if __name__ == '__main__':
    sess = tf.Session()
    odcn = ODCN_Model()
    dataPreprocessor = DataPreprocessor.DataPreprocessor()
    TP_List, FN_List, TN_List, FP_List = odcn.test(sess, dataPreprocessor)
    # odcn.train(sess, dataPreprocessor)
