# Code for Traffic Dataset
# Data available at following URL
# https://data.gov.uk/dataset/dc18f7d5-2669-490f-b2b5-77f27ec133ad/highways-agency-network-journey-time-and-traffic-flow-data
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
df=pd.read_csv("2011-01 AL1165A.csv")

x=df.index%96

y=df[' Travel Time']
# 96 different timestamps in a day
x=df.index%96
y=df[' Travel Time']
x=np.asarray(x)
y=np.asarray(y)
y=y.reshape(-1,1)
#numsamples is length of x which is 96
numsamples=x.shape[0]

# Process RMT features
def prepareRMTfeatures():
    df = pd.read_csv("Traveltimefeatures.csv", header=None)
    times = df.iloc[1, :]
    standardDevTime = df.iloc[3, :]
    RMTdesclength = len(df) - 10
    # 10 is the max number of features per timestep
    values = np.zeros((numsamples, RMTdesclength, 10))
    countRepeats = np.zeros((numsamples,)).astype('int')

    for i in range(len(times)):
        # we are looking at the time of the feature
        time = times[i].astype('int')
        standev = standardDevTime[i].astype('int')
        for t in range(time - 3 * standev, time + 3 * standev + 1):
            z = (t - time) / standev
            p = np.exp(-z ** 2)

            if (t >= 2976 or t < 0):
                continue
            randNum = np.random.rand(1)
            if randNum < p:
                values[t, :, countRepeats[t]] = df.iloc[10:, i]
                countRepeats[t] += 1

    values = values.reshape(numsamples, -1)
    return values

featureData=prepareRMTfeatures()

# Process data and build history
def process_data(x, y, RMTdata, seq_length):
    querytime = x[seq_length:].reshape(-1, 1)
    # y is the travel time we are predicting
    y_updated = y[seq_length:]

    prev_jt_yupdated = np.zeros((len(y_updated), seq_length))
    # timestamps corresponding to the travel times
    timestamp = np.zeros((len(y_updated), seq_length))
    RMTupdated = np.zeros((len(y_updated), seq_length, RMTdata.shape[1]))
    for i in range(seq_length):
        # prev_jt_yupdated is the list of previous travel times for a query travel time
        # list of travel  times 4 steps before we are predicting, building up history column wise

        prev_jt_yupdated[:, i] = y[i:-(seq_length - i)].reshape(-1)
        timestamp[:, i] = x[i:-(seq_length - i)].reshape(-1)
        RMTupdated[:, i, :] = RMTdata[i:-(seq_length - i), :].reshape(-1, RMTdata.shape[1])

    return prev_jt_yupdated, y_updated, timestamp, querytime, RMTupdated

seq_length = 4
xTensor,yTensor,timestamp,querytime,RMTupdated=process_data(x,y,featureData,seq_length)

def test_train_split(data, validFraction=0.2,testFraction=0.1):
    trainFraction=1-validFraction-testFraction
    nchunks=np.asarray(data).shape[0]
    chunk_length=np.asarray(data).shape[1]

    train_cutoff = floor((1-validFraction-testFraction)*nchunks )
    valid_cutoff = floor((1-testFraction)*nchunks )
    test_cutoff = nchunks


    return data[:train_cutoff], data[train_cutoff:valid_cutoff],data[valid_cutoff:test_cutoff]

from math import floor
x_train,     x_valid,     x_test     =test_train_split(xTensor)
time_train,  time_valid,  time_test  =test_train_split(timestamp)
query_train, query_valid, query_test =test_train_split(querytime)
y_train,     y_valid,     y_test     =test_train_split(yTensor)
feature_train,feature_valid,feature_test     = test_train_split(RMTupdated)

#Dimensionality Reduction

from sklearn.decomposition import PCA
# Target rank 10
pcaDimension = 10
pca = PCA(pcaDimension)

flatfeatureTrain = np.swapaxes(feature_train.reshape(-1, 128, 10), 1, 2).reshape(-1, 128)
flatfeatureValid = np.swapaxes(feature_valid.reshape(-1, 128, 10), 1, 2).reshape(-1, 128)
flatfeatureTest = np.swapaxes(feature_test.reshape(-1, 128, 10), 1, 2).reshape(-1, 128)


pca.fit(flatfeatureTrain[:, :])

# pca transform uses the coefficients to give output
feature_train = pca.transform(flatfeatureTrain[:, :]).reshape(feature_train.shape[0], feature_train.shape[1], -1)
feature_valid = pca.transform(flatfeatureValid[:, :]).reshape(feature_valid.shape[0], feature_valid.shape[1], -1)
feature_test = pca.transform(flatfeatureTest[:, :]).reshape(feature_test.shape[0], feature_test.shape[1], -1)
del pca

#Multihead Attention############################################################################################
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

##########################################END MULTIHEAD ATTENTION###########################################################
# LSTM Model with Dual Regional Attention
def build_model(heads1, heads2):
    from keras.optimizers import Adam, SGD
    from keras.models import Model
    from keras.layers import Input, Embedding, Concatenate, Dense, LSTM, Reshape, Flatten, Bidirectional
    from keras.layers import Lambda

    hidden_dim = 15
    # 3D embedding space
    embedding_size = 3
    seq_length = 4
    # Prev Travel Times
    input_ = Input(shape=(seq_length,))
    inputreshaped = Reshape((seq_length, 1))(input_)

    # Embedded Timestamps for previous timestamps and for next timestamp
    # time is represented as embedding(turns list of names to points in space)
    timestamps_ = Input(shape=(seq_length,))
    querytime_ = Input(shape=(1,))
    features_ = Input(shape=(seq_length, feature_train.shape[-1]))
    times = Concatenate()([timestamps_, querytime_])
    # plus one is for the query time
    embedding = Embedding(96, embedding_size, input_length=seq_length + 1)(times)

    # Take timestamps and query apart
    timestampEmbedding = Lambda(lambda embed: embed[:, :seq_length], output_shape=(seq_length, embedding_size))(
        embedding)
    queryEmbedding = Lambda(lambda embed: embed[:, seq_length:], output_shape=(1, embedding_size))(embedding)


    lstm = LSTM(hidden_dim, return_sequences=True)(inputreshaped)
    lstm_with_timestamps = Concatenate()([lstm, timestampEmbedding])
    features_with_timestamps = Concatenate()([features_, timestampEmbedding])
    # Query is RMT features,key is input data for previous travel times
    weighted0 = Attention(heads1, 10)([features_with_timestamps, lstm_with_timestamps, lstm])

    lstm_with_aspect = Concatenate()([lstm_with_timestamps, weighted0])

    weighted1 = Attention(heads2, 10)([queryEmbedding, lstm_with_aspect, lstm_with_aspect])



    flat = Flatten()(weighted1)
    output = Dense(1, activation=None)(flat)

    model = Model(inputs=[input_, timestamps_, querytime_, features_], outputs=output)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])
    print(model.summary())
    return model

# Call model
model = build_model(1, 8)
history = model.fit([x_train, time_train, query_train, feature_train], y_train,
                    epochs=15, validation_data=([x_valid, time_valid, query_valid, feature_valid], y_valid),
                    verbose=2, shuffle=False)

# plot history
plt.title("Training Loss & Validation Loss")
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='Validation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

err = model.predict([x_valid,time_valid,query_valid,feature_valid])-y_valid
plt.hist(err)

np.mean(np.abs(err))

np.mean(np.abs(err/y_valid * 100))

