import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import pickle

n_windows = 5
n_output = 6
# data_path = '/media/rrr/4bff43fa-41e1-46a3-8249-12574cbe3109/home/rrr/Desktop/temp/data_PICK_rnn_cnn/data_color/'
# data_path = './data_color_2_zac/'
data_path = './data_simpler_2/'

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, sample_shape=None, batch_size=1, shuffle=False):
        self.list_IDs = list_IDs
        self.total_number_of_samples = len(list_IDs)
        self.sample_shape = sample_shape
        self.batch_size = batch_size
        self.n_windows = self.sample_shape[0]
        self.batches_per_epoch =  500 #int( np.floor(self.total_number_of_samples / self.batch_size) / 200.0 )
        self.shuffle = shuffle

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        # print("-", end="")
        #print("Index: ", index)
        # print(self.shuffle)
        if self.shuffle :
            indexes = []
            for i in range(self.batch_size) :
                indexes += [ self.list_IDs[np.random.randint( self.total_number_of_samples - self.n_windows )] ]
        else :
            indexes = [self.list_IDs[k] for k in range( index, index+self.batch_size )]
        # print(index, " --> ", indexes)
        X, y = self.data_generation(indexes)
        # return self.X, self.y
        return X, y

    def on_epoch_end(self):
        # print("epoch end")
        np.random.seed()
        pass

    def data_generation(self, indexes ):
        Xims = np.empty((self.batch_size, *self.sample_shape))
        Xjjj = np.empty((self.batch_size, self.n_windows, 6))
        y = np.empty((self.batch_size, 6+10))

        for i, ID in enumerate(indexes):
            for j in range(self.n_windows) :
                Xims[i,j,] = np.load(data_path + 'img{:07}.npy'.format(ID+j)) / 255.0
                pkl_file = open(data_path + '{:07}.pkl'.format(ID + j), 'rb')
                pos = pickle.load(pkl_file)
                Xjjj[i,j,:] = np.array(pos['joints']) / 2.7

            X = (Xims, Xjjj)

            pkl_file = open(data_path + '{:07}.pkl'.format(ID + self.n_windows - 1), 'rb')
            pos0 = pickle.load(pkl_file)
            pkl_file = open(data_path + '{:07}.pkl'.format(ID + self.n_windows), 'rb')
            pos1 = pickle.load(pkl_file)
            # dict_keys(['box_pos', 'joints', 'gripper_pos', 'gripper', 'state', 'cube_pos'])

            if np.abs( pos1['joints'][0] - pos0['joints'][0] ) > 0.2 :
                y[i,] = np.array( list([0,0,0,0,0,0])  + list([pos0['gripper'] / 2.0 - 0.25]) +
                    list(np.array(pos0['cube_pos']))  +  list(np.array(pos0['gripper_pos']))  +  list(np.array(pos0['box_pos'])) )
            else :
                y[i,] = np.array( list( 4.0*np.array(pos1['joints']) - 4.0*np.array(pos0['joints']) ) + list([pos0['gripper'] / 2.0 - 0.25]) +
                    list(np.array(pos0['cube_pos']))  +  list(np.array(pos0['gripper_pos']))  +  list(np.array(pos0['box_pos'])) )


        return X, y
################## DataGenerator END #################

# tf.reset_default_graph()
if True :
    # cnn = tf.keras.models.load_model("./model_weights/001_cnn.h5")
    cnn = tf.keras.models.load_model("./c6_0.h5")
    # cnn.compile()
    # model = tf.keras.models.load_model("./model_weights/001_lstm.h5")
    model = tf.keras.models.load_model("./m6_0.h5")
    # model.compile()
else :
    cnn = tf.keras.models.Sequential()
    batch_normalization = False
    ################# WITH batch normalization ##############
    if batch_normalization :
        # cnn.add(tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(256,256,3)))
        cnn.add(tf.keras.layers.Conv2D(32, kernel_size=3 , use_bias=False,input_shape=(256,256,3)))
        cnn.add(tf.keras.layers.BatchNormalization())
        cnn.add(tf.keras.layers.Activation("relu"))
        cnn.add(tf.keras.layers.MaxPool2D())
        # 128 ^2
        # cnn.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(48, kernel_size=3 , use_bias=False))
        cnn.add(tf.keras.layers.BatchNormalization())
        cnn.add(tf.keras.layers.Activation("relu"))
        cnn.add(tf.keras.layers.MaxPool2D())
        #64 ^2
        # cnn.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(64, kernel_size=3 , use_bias=False))
        cnn.add(tf.keras.layers.BatchNormalization())
        cnn.add(tf.keras.layers.Activation("relu"))
        cnn.add(tf.keras.layers.MaxPool2D())
        #32 ^2
        # cnn.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(128, kernel_size=3 , use_bias=False))
        cnn.add(tf.keras.layers.BatchNormalization())
        cnn.add(tf.keras.layers.Activation("relu"))
        cnn.add(tf.keras.layers.MaxPool2D())
        #16 ^2
        cnn.add(tf.keras.layers.Conv2D(256, kernel_size=3 , use_bias=False))
        cnn.add(tf.keras.layers.BatchNormalization())
        cnn.add(tf.keras.layers.Activation("relu"))
        cnn.add(tf.keras.layers.MaxPool2D())
        #8 ^2
    ################# no batch normalization ##############
    else :
        cnn.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(256,256,3)))
        cnn.add(tf.keras.layers.MaxPool2D())
        # 128 ^2
        cnn.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D())
        #64 ^2
        cnn.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D())
        #32 ^2
        cnn.add(tf.keras.layers.Conv2D(196, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D())
        #16 ^2
        cnn.add(tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D())
        #8 ^2

    cnn.add(tf.keras.layers.Flatten())
    # cnn.summary()

    input_ims = tf.keras.layers.Input(shape=(n_windows,256,256,3))
    time_distribute = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda x: cnn(x)))(input_ims)
    input_jjj = tf.keras.layers.Input(shape=(n_windows,6))
    # lstm_in = tf.concat([time_distribute,input_jjj],2)
    lstm_in = tf.keras.layers.Lambda(lambda x: tf.concat(x,2) )([time_distribute,input_jjj])
    lstm_lay = tf.keras.layers.LSTM(128, return_sequences=False)(lstm_in)
    lstm_lay = tf.keras.layers.Dense(128, activation='tanh')(lstm_lay)
    output_lay = tf.keras.layers.Dense(n_output +10, activation='tanh')(lstm_lay)
    model = tf.keras.models.Model(inputs=[input_ims, input_jjj], outputs=[output_lay])


# cnn = tf.keras.models.load_model("./c4_1.h5")

opt = tf.keras.optimizers.Adam(lr=0.00002)  # they used 0.0001

model.compile(optimizer=opt, loss="mse")

# graph = tf.get_default_graph()

training_generator = DataGenerator(list(range(0,95000)), sample_shape=(n_windows,256,256,3), shuffle=True, batch_size=1)
# checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="best_weights.hdf5", monitor = 'loss', verbose=2, save_best_only=True)
history = model.fit_generator(generator=training_generator, use_multiprocessing=False, workers=1,verbose=1, epochs=100,
                              steps_per_epoch=95000, shuffle=False ) #, callbacks=[checkpointer],)

cnn.save("c6_1.h5") #### !!!!!!!!!!!!!!!!!!!!!!!!
model.save("m_1.h5")   #### !!!!!!!!!!!!!!!!!!!!!!!!
#######################################################
O = 93988 #1380 # 29500    # 30000
M = O + 300 #2380 # 29990   # 30500
DD = M-O
test_generator = DataGenerator(list(range(O,M)), sample_shape=(n_windows,256,256,3), shuffle=False, batch_size=1)
jj = np.zeros((DD,7))
jj_pred = np.zeros((DD,7))

# jjj_l = np.load("jjj.npy") / 2.7

for i in range(DD) :
    ss = test_generator.__getitem__(i)
    # print(ss[0].shape,ss[1].shape,)
    jj[i,:] = ss[1][-1,:7]
    vv = model.predict_on_batch(ss[0])
    jj_pred[i,:] = vv[0,:7]
    # print(vv.shape)

cc = ["r","b","g","c","y","k","r"]
for i in range(7) :
    plt.plot(jj[:,i], "-", label="joint {}".format(i), color=cc[i])
    plt.plot(jj_pred[:,i], ".", label="Forecast on seen", color=cc[i])
    # plt.plot(jjj_l[:M,i], ".", label="loaded j", color=cc[i])
plt.legend()
plt.show()
