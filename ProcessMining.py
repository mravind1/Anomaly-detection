#Code for Process mining Dataset
# Dataset: Rare Event Classification in Multivariate Time Series
#Data available at following URL
# https://arxiv.org/abs/1809.10717

import scipy.io
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler


# Read data
import pandas as pd

fname = "processminer-rare-event-mts - data.csv"
df = pd.read_csv(fname)

x = df.values[:, 2:]
x = x.astype(float)

# KMeans Clustering
from sklearn.cluster import KMeans
number_of_clusters=40
model=KMeans(number_of_clusters)
x_addl=np.eye(number_of_clusters)[model.fit_predict(x)]

y=df.values[:,1]
y=y.reshape(-1,1)
y=y.astype(int)

# Chunk data
def chunk_data(data, chunck_size=500):
    nchuncks = floor(data.shape[0] / chunck_size)
    last_input = chunck_size * nchuncks

    shape = [nchuncks, chunck_size] + list(data.shape[1:])
    dataTensor = data[:last_input].reshape(shape)
    return dataTensor
# For without clustering usecase call chunk_data with x
from math import floor
xTensor=chunk_data(x_addl,500  )
yTensor=chunk_data(y,500  )
numsamples=x.shape[0]

# Anomaly detection And Split Chunks As Positive And Negative
def anomaly_detectionAndSplit(xTensor, yTensor, duplicateTrain=True):
    x_neg = []
    x_pos = []
    y_neg = []
    y_pos = []
    countofZeros = 0
    countofOnes = 0
    countofOnesInAPosChunk = []
    nchuncks = yTensor.shape[0]
    chunck_size = yTensor.shape[1]
    for i in range(nchuncks):
        countofOnesInAPosChunk.append(0)
        flag = False
        for j in range(chunck_size):
            if yTensor[i, j, 0] == 1:
                countofOnes += 1
                countofOnesInAPosChunk[-1] += 1
                flag = True
            else:
                countofZeros += 1

        if flag:

            x_pos.append(xTensor[i, :, :])
            y_pos.append(yTensor[i, :, :])

        else:

            x_neg.append(xTensor[i, :, :])
            y_neg.append(yTensor[i, :, :])
            del countofOnesInAPosChunk[-1]

    return x_pos, x_neg, y_pos, y_neg, countofOnesInAPosChunk

x_pos,x_neg,y_pos,y_neg,countofOnesInAPosChunk=anomaly_detectionAndSplit(xTensor,yTensor)

# Split into Train,Valid and Test
def test_train_split(data, validFraction=0.2, testFraction=0.2):
    trainFraction = 1 - validFraction - testFraction

    nchunks = np.asarray(data).shape[0]

    chunk_length = np.asarray(data).shape[1]
    train_cutoff = floor((1 - validFraction - testFraction) * nchunks)
    valid_cutoff = floor((1 - testFraction) * nchunks)
    test_cutoff = nchunks

    return data[:train_cutoff], data[train_cutoff:valid_cutoff], data[valid_cutoff:test_cutoff]

# Read RMT feature file extracted from input data
def prepareRMTfeatures():
    df = pd.read_csv("Processfeature.csv", header=None)

    times = df.iloc[1, :]
    standardDevTime = df.iloc[3, :]
    RMTdesclength = len(df) - 10
    max_repeats = 560

    values = np.zeros((numsamples, RMTdesclength, max_repeats))
    countRepeats = np.zeros((numsamples,)).astype('int')
    for i in range(len(times)):

        time = times[i].astype('int')
        standev = standardDevTime[i].astype('int')
        for t in range(time - 3 * standev, time + 3 * standev + 1):
            z = (t - time) / standev
            p = np.exp(-z ** 2)

            if (t >= 18398 or t < 0):
                continue
            randNum = np.random.rand(1)
            if randNum < p:
                values[time, :, countRepeats[time]] = df.iloc[10:, i]
                countRepeats[time] += 1

    values = values.reshape(numsamples, -1)
    return values

featureData=prepareRMTfeatures()


featureTensor=chunk_data(featureData,500  )
feature_pos,feature_neg,y_pos,y_neg,countofOnesInAPosChunk=anomaly_detectionAndSplit(featureTensor,yTensor)

x_neg_train, x_neg_valid,x_neg_test=test_train_split(x_neg)
x_pos_train, x_pos_valid,x_pos_test=test_train_split(x_pos)
y_neg_train, y_neg_valid,y_neg_test=test_train_split(y_neg)
y_pos_train, y_pos_valid,y_pos_test=test_train_split(y_pos)
feature_neg_train, feature_neg_valid,feature_neg_test=test_train_split(feature_neg)
feature_pos_train, feature_pos_valid,feature_pos_test=test_train_split(feature_pos)


def concatenate_And_shuffle(data_neg, data_pos, index=None):
    x = np.concatenate([data_neg, data_pos])
    return np.asarray(x)

#Concatenate negative and positive chunks
# Use Sampler to balance positive and negative chunks
x_train = concatenate_And_shuffle(x_neg_train[:3], x_pos_train)

y_train = concatenate_And_shuffle(y_neg_train[:3], y_pos_train)
x_valid = concatenate_And_shuffle(x_neg_valid, x_pos_valid)
y_valid = concatenate_And_shuffle(y_neg_valid, y_pos_valid)
x_test = concatenate_And_shuffle(x_neg_test, x_pos_test)
y_test = concatenate_And_shuffle(y_neg_test, y_pos_test)
feature_train = concatenate_And_shuffle(feature_neg_train[:3], feature_pos_train)
feature_valid = concatenate_And_shuffle(feature_neg_valid, feature_pos_valid)
feature_test = concatenate_And_shuffle(feature_neg_test, feature_pos_test)


#Dimensionality reduction
from sklearn.decomposition import PCA
# Target rank is 10
pcaDimension=10
pca =PCA(pcaDimension)
print(feature_train.shape)
#Get last dimension as 128
# 560 max RMT columns per timestep
flatfeatureTrain=np.swapaxes(feature_train.reshape(-1,128,560),1,2).reshape(-1,128)
flatfeatureValid=np.swapaxes(feature_valid.reshape(-1,128,560),1,2).reshape(-1,128)
flatfeatureTest=np.swapaxes(feature_test.reshape(-1,128,560),1,2).reshape(-1,128)



pca.fit(flatfeatureTrain[:,:] )

feature_train =pca.transform(flatfeatureTrain[:,:]).reshape(feature_train.shape[0],feature_train.shape[1],-1)
feature_valid =pca.transform(flatfeatureValid[:,:]).reshape(feature_valid.shape[0],feature_valid.shape[1],-1)
feature_test =pca.transform(flatfeatureTest[:,:]).reshape(feature_test.shape[0],feature_test.shape[1],-1)
del pca

####### Multi Headed Attention Unit#################################################################################
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Concatenate, Input, Reshape


# Reused code from an implementation of paper 'Attention is All you need' NIPS 2017
#https://arxiv.org/abs/1706.03762
class Attention(Layer):
    # Number of heads running in parallel, no of kinds of information to pass
    # Size per head

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head


        super(Attention, self).__init__(**kwargs)

    # Three separate weights
    def build(self, input_shape):
        # Query weight 19600*10 for one head, 19600*20 for two heads
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        # Key
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        # Values
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # Query, Key, Value
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)

        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))

        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        # softmax gets applied
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])

        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))

        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))

        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
######################################END MULTIHEAD ATTENTION #####################################
    # LSTM Model with Dual Regional Attention

    from keras.layers import Concatenate, Input, Reshape, Bidirectional
    from keras.models import Model
    from keras.layers import Input, Reshape, Concatenate, Dot, Permute, RepeatVector, Multiply, LSTM
    from keras.optimizers import Adam, SGD
    hidden_dim = 80

    seq_length = 500

    input_ = Input(shape=(seq_length, 40))

    flat = input_

    features_ = Input(shape=(seq_length, featureData.shape[-1]))
    lstm = LSTM(hidden_dim, return_sequences=True)(flat)

    weighted0 = Attention(1, 10)([features_, flat, flat])

    lstm_with_aspect = Concatenate()([lstm, weighted0])

    weighted1 = Attention(8, 10)([lstm_with_aspect, lstm, lstm])

    output = Dense(1, activation='sigmoid')(weighted1)

    model = Model(inputs=[input_, features_], outputs=output)
    optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())


    from matplotlib import pyplot

    history = model.fit([x_train, feature_train], y_train, batch_size=60,
                        epochs=10, validation_data=([x_valid, feature_valid], y_valid),
                        verbose=2, shuffle=False)

    # plot history
    pyplot.title("Training Loss & Validation Loss")
    pyplot.plot(history.history['loss'], label='training')
    pyplot.plot(history.history['val_loss'], label='Validation')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    model.layers
    from keras import backend as K
    for layer in model.layers:
        print(layer.name)

    get_wt_output = K.function([model.layers[0].input
                                   , model.layers[1].input
                                ],
                               [model.layers[6].output])
    xx = get_wt_output([x_test, feature_test])[0]

    #Confusion matrix

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    LABELS = ["0", "1"]

    y_pred = model.predict([x_test, feature_test])

    y_predrounded = np.round(y_pred).astype('int')

    y_predrounded = np.where(xx > 0.45, 1, 0)

    test_ydummy = y_test.reshape(-1)
    y_preddummy = y_predrounded.reshape(-1)

    conf_matrix = confusion_matrix(test_ydummy.astype('int'), y_preddummy)
    print(conf_matrix)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

    #F1 score calculation
    from sklearn.metrics import classification_report
    report = classification_report(test_ydummy.astype('int'), y_preddummy)


