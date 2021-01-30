import math
import DataPreprocessor
import tensorflow as tf
import tensorflow.contrib.slim as slim


class DBCN_Model:
    '''This class is created for building, training and testing Dilated-Block-Based Convolutional Network (DBCN) from ``Perceiving More Truth:A Dilated-Block-Based Convolutional Network for Rumor Identification``.

    There are three major functions in class DBCN_Model:
    function 1. __dbcnModel() is for building the DBCN which is implemented by Tensorflow and it is used by train() and test().
    function 2. train() is for training the DBCN model.
    function 3. test() is for testing the trained DBCN model where the trained model file is stored in modelSavingPath.
    '''

    modelSavingPath = "./trained models/networks/dbcn/"

    def __init__(self, shuffle_size=20, batch_size=16,
                 kernel_width_DilatedBlock_1=3, kernel_num_DilatedBlock_1=6, dilation_rate_DilatedBlock_1=2,
                 kernel_width_DilatedBlock_2=4, kernel_num_DilatedBlock_2=4, dilation_rate_DilatedBlock_2=4,
                 kernel_width_Residual=17, kernel_num_Residual=4, output_units=2, dropout_rate=0.5,
                 initial_learning_rate=0.0005, epoch_num=250):
        '''
        Constructor of class DBCN_Model.

        All parameters have default values which follow the Section 4.2.2 Parameter Settings in ``Perceiving More Truth:A Dilated-Block-Based Convolutional Network for Rumor Identification``.
        So you do not need to clarify them in the constructor by default unless you need to change them by your thoughts.

        :param shuffle_size: An `int` representing the number of elements from training dataset that the new dataset will be randomly sampled.
        :param batch_size: An `int` representing the number of consecutive elements of shuffled training dataset to combine in a single batch.
        :param kernel_width_DilatedBlock_1: An `int` representing the width of the dilated convolutional layer in the first dilated block.
        :param kernel_num_DilatedBlock_1:  An `int` representing the out depth of the dilated convolutional layer in the first dilated block.
        :param dilation_rate_DilatedBlock_1: A single `int` representing the dilation rate of the dilated convolutional layer in the first dilated block (1-D: column-wise).
        :param kernel_width_DilatedBlock_2: An `int` representing the width of the dilated convolutional layer in the second dilated block.
        :param kernel_num_DilatedBlock_2: An `int` representing the out depth of the dilated convolutional layer in the second dilated block.
        :param dilation_rate_DilatedBlock_2: A single `int` representing the dilation rate of the dilated convolutional layer in the second dilated block (1-D: column-wise).
        :param kernel_width_Residual: An `int` representing the width of the residual convolutional layer.
        :param kernel_num_Residual: An `int` representing the out depth of the residual convolutional layer.
        :param output_units: An `int` representing the number of  output units in dense layer.
        :param dropout_rate: A `float` representing the dropout rate in dropout layer.
        :param initial_learning_rate: A `float` representing the initial learning rate of Adam optimizer during training.
        :param epoch_num: An `int` representing the number of training epoch.
        '''
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
        self.kernel_width_Residual = kernel_width_Residual
        self.kernel_num_Residual = kernel_num_Residual
        self.output_units = output_units
        self.dropout_rate = dropout_rate
        self.initial_learning_rate = initial_learning_rate
        self.epoch_num = epoch_num

    def __init_weights(self, shape, name):
        '''
        Initializer of convolutional kernel.

        Forming a convolutional kernel with initial weights by tf.truncated_normal method.

        :param shape: A `List` representing the shape of the convolutional kernel.
        :param name: A `Str` representing the name of the convolutional kernel.
        :return: A `tf.Variable` representing the convolutional kernel with initial weights.
        '''
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1),
                           name=name)  # the default value of stddev is set as 0.1

    def __dbcnModel(self, sess):
        '''
        Builder of DBCN model.

        Building the structure and related calculate operations of DBCN.

        :param sess: A `tf.Session` representing the session of the tensorflow model.
        :return: The calculate operations of loss, accuarcy, adam optimization, model predictions and real labels of the model.
        '''
        print("Building DBCN model!")

        # The following codes clarify placeholders in the model, which includes inputs, labels, dropout_training_flag and batchnormalization_training_flag.
        inputs = tf.placeholder(tf.float32, shape=(None, self.group_num, self.para_embedding_size), name="inputs")
        # inputs: the input (X) of the DBCN with the shape of (Batch,GroupNum,Embedding Size),
        labels = tf.placeholder(tf.float32, shape=(None, self.output_units), name="labels")
        # labels: the real label (Y) of inputs with the shape of (Batch,Label Size),
        dropout_training_flag = tf.placeholder(tf.bool, None, name="dropout_training_flag")
        # dropout_training_flag: the training signal of dropout layer. It will be activated if set to `tf.True` in training, and will be deactivated if set to `tf.False` in testing.
        batchnormalization_training_flag = tf.placeholder(tf.bool, None, name="batchnormalization_training_flag")
        # batchnormalization_training_flag: the training signal of batch normalization layer. It will be activated correctly if set to `tf.False` no matter in training or testing.

        # The following codes initialize the weights of convolutional kernels in the model.
        w_residual = self.__init_weights(
            shape=[self.kernel_width_Residual, self.para_embedding_size, 1, self.kernel_num_Residual],
            name="W_residual")
        # the residual convolutional kernel with the shape of (width,height,channel,out depth) where the channel is 1.
        w_dilatedblock_1 = self.__init_weights(
            shape=[self.kernel_width_DilatedBlock_1, self.para_embedding_size, 1, self.kernel_num_DilatedBlock_1],
            name="W_dilatedblock1")
        # the kernel of the first dilated convolutional layer with the shape of (width,height,channel,out depth) where the channel is 1.
        w_dilatedblock_2 = self.__init_weights(
            shape=[self.kernel_width_DilatedBlock_2, self.para_embedding_size, self.kernel_num_DilatedBlock_1,
                   self.kernel_num_DilatedBlock_2], name="W_dilatedblock2")
        # the kernel of the second dilated convolutional layer with the shape of (width,height,channel,out depth) where the channel is 1.

        # The following codes reshape and unstack the input in order to let the input adapt the operation of column-wise dilated convolutional layer.
        with tf.name_scope("inputs"):
            inputs_reshape = tf.expand_dims(inputs, -1)# reshape the inputs (expand the dimension of inputs from 3-D to 4-D (Batch,GroupNum,Embedding Size)=>(Batch,GroupNum,Embedding Size,1)).
            inputs_unstack = tf.unstack(inputs_reshape, axis=2)# unstack the inputs on the 3rd axis--Embedding Size, the shape of inputs will change to Embedding Size*(Batch,GroupNum,1)

        # The following codes send the inputs into residual convolutional layers and perform related operations.
        with tf.name_scope("residual_convolution"):
            convs = [] # for collecting the residual convolution results.
            w_unstack = tf.unstack(w_residual, axis=1)# unstack the residual convolutional kernel for column-wise convolution.
            # column-wise convolution
            for i in range(len(inputs_unstack)):
                conv = tf.nn.convolution(input=inputs_unstack[i], filter=w_unstack[i], padding="VALID")
                convs.append(conv)
            convres = tf.stack(convs, axis=2)# for stacking the residual convolution results (on the 3rd axis).
            print("residual convolution:" + str(convres))

        # The following codes send the inputs into first dilated block and perform related operations.
        with tf.name_scope("dilated_block_1"):
            convs1 = []# for collecting the first dilated block results.
            w1_unstack = tf.unstack(w_dilatedblock_1, axis=1)# unstack the kernel of the first dilated convolutional layer for column-wise dilated convolution.
            # column-wise dilated convolution, batch normalization and activation
            for i in range(len(inputs_unstack)):
                conv1 = tf.nn.convolution(input=inputs_unstack[i], filter=w1_unstack[i], padding="VALID",
                                          dilation_rate=[self.dilation_rate_DilatedBlock_1])
                bn1 = tf.layers.batch_normalization(conv1, training=batchnormalization_training_flag)
                ac1 = tf.nn.relu(bn1)
                convs1.append(ac1)
            convres1 = tf.stack(convs1, axis=2)# for stacking the first dilated block results (on the 3rd axis).
            print("dilated block 1:" + str(convres1))

        # The following codes send the first dilated block results into second dilated block and perform related operations.
        with tf.name_scope("dilated_block_2"):
            convs2 = []# for collecting the second dilated block results.
            convres1_unstack = tf.unstack(convres1, axis=2)# unstack the results of the first dilated block for column-wise dilated convolution.
            w2_unstack = tf.unstack(w_dilatedblock_2, axis=1)# unstack the kernel of the first dilated convolutional layer for column-wise dilated convolution.
            # column-wise dilated convolution, batch normalization and activation
            for i in range(len(convres1_unstack)):
                conv2 = tf.nn.convolution(input=convres1_unstack[i], filter=w2_unstack[i], padding="VALID",
                                          dilation_rate=[self.dilation_rate_DilatedBlock_2])
                bn2 = tf.layers.batch_normalization(conv2, training=batchnormalization_training_flag)
                ac2 = tf.nn.relu(bn2)
                convs2.append(ac2)
            convres2 = tf.stack(convs2, axis=2)# for stacking the second dilated block results (on the 3rd axis).
            print("dilated block 2:" + str(convres2))

        # The following codes concatenate the results of second dilated block and the results residual convolution and perform other operations.
        with tf.name_scope("concat_pool_flat_output"):
            concatres = tf.concat([convres, convres2], axis=1)# concatenation
            print("concat:" + str(concatres))
            poolres = tf.nn.max_pool(value=concatres, ksize=[1, int(convres2.shape[1]) + int(convres.shape[1]), 1, 1],
                                     strides=[1, 1, 1, 1], padding="VALID") # maxpooling
            print("pooling:" + str(poolres))
            flatres = slim.flatten(poolres)# flat
            print("flat:" + str(flatres))
            dropoutres = tf.layers.dropout(inputs=flatres, rate=self.dropout_rate, training=dropout_training_flag)# dropout
            print("dropout:" + str(dropoutres))
            predictions = tf.layers.dense(inputs=dropoutres, units=self.output_units, activation=tf.nn.tanh)# dense prediction output
            print("dense:" + str(predictions))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels))
        # get the loss between predictions and real labels
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1)), tf.float32))
        # calculate the accuarcy of the model
        train_optimization = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate).minimize(loss)
        # adam optimization

        return loss, acc, train_optimization, predictions, labels

    def train(self, sess, dataPreprocessor):
        '''
        Trainer of the DBCN model.

        Training and Saving the model with the highest test accuracy.

        :param sess: A `tf.Session` representing the session of the tensorflow model.
        :param dataPreprocessor: A ``
        :return: None
        '''
        trainX, trainY, trainFileNameNoDict = dataPreprocessor.getTrainData()
        devX, devY, devFileNameNoDict = dataPreprocessor.getDevData()
        testX, testY, testFileNameNoDict = dataPreprocessor.getTestData()
        trainDataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

        trainData = trainDataset.shuffle(self.shuffle_size).batch(self.batch_size).repeat()
        iterator = trainData.make_one_shot_iterator()
        next_iterator = iterator.get_next()
        iterations = math.ceil(trainX.shape[0] / self.batch_size)
        loss, acc, train_optimization, predictions, labels = self.__dbcnModel(sess)
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
                    saver.save(sess, DBCN_Model.modelSavingPath + "dbcn")
                    previous_best_accuarcy = testAcc
                    print("Saving current model!")

    def test(self, sess, dataPreprocessor):
        _, _, _, _, _ = self.__dbcnModel(sess)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(DBCN_Model.modelSavingPath))
        print("Loading model completed!")
        graph = tf.get_default_graph()
        testX, testY, testFileNameNoDict = dataPreprocessor.getTestData()
        feed_dict = {"inputs:0": testX, "dropout_training_flag:0": False, "batchnormalization_training_flag:0": False}
        denseOutput = graph.get_tensor_by_name("concat_pool_flat_output/dense/Tanh:0")
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
    dbcn = DBCN_Model()
    dataPreprocessor = DataPreprocessor.DataPreprocessor()
    TP_List, FN_List, TN_List, FP_List = dbcn.test(sess, dataPreprocessor)
    # dbcn.train(sess,dataPreprocessor)
