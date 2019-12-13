import vrep
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import time
import pickle

# cnn = tf.keras.models.load_model("./models_weights/001_cnn.h5")
# model = tf.keras.models.load_model("./models_weights/001_lstm.h5")

cnn = tf.keras.models.load_model("./c6_1.h5")
model = tf.keras.models.load_model("./m6_1.h5")

# cnn.load_weights("./models_weights/cnn_w.h5")
# model.load_weights("./models_weights/lstm_w.h5")

print("Neural network loaded")

vrep.simxFinish(-1)
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
vrep.simxSynchronous(clientID,True)
vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)

err, cam = vrep.simxGetObjectHandle(clientID, "sensor_1", vrep.simx_opmode_blocking)
err, bvp = vrep.simxGetObjectHandle(clientID, "BaxterVacuumCup_link", vrep.simx_opmode_blocking)
err, cube = vrep.simxGetObjectHandle(clientID, "cub", vrep.simx_opmode_blocking)
err, m01 = vrep.simxGetObjectHandle(clientID, "base_cub", vrep.simx_opmode_blocking)
err, m1 = vrep.simxGetObjectHandle(clientID, "marker1", vrep.simx_opmode_blocking)
err, m2 = vrep.simxGetObjectHandle(clientID, "marker2", vrep.simx_opmode_blocking)
err, m03 = vrep.simxGetObjectHandle(clientID, "box_center", vrep.simx_opmode_blocking)
err, m3 = vrep.simxGetObjectHandle(clientID, "marker3", vrep.simx_opmode_blocking)
err, m000 = vrep.simxGetObjectHandle(clientID, "UR5_joint1", vrep.simx_opmode_blocking)
err, table = vrep.simxGetObjectHandle(clientID, "diningTable", vrep.simx_opmode_blocking)
err, cam = vrep.simxGetObjectHandle(clientID, "sensor_1", vrep.simx_opmode_blocking)
vrep.simxSetObjectParent(clientID, cube, table, 1, vrep.simx_opmode_blocking)
vrep.simxSetObjectPosition(clientID, cube, m01, [0.0,0.0,0.0], vrep.simx_opmode_blocking)

jjh = []
for i in range(6) :
    err, jh = vrep.simxGetObjectHandle(clientID, "UR5_joint{}".format(i+1), vrep.simx_opmode_blocking)
    jjh += [jh]

def get_jj() :
    jj = []
    for i in range(6) :
        err, j = vrep.simxGetJointPosition(clientID, jjh[i], vrep.simx_opmode_blocking)
        jj += [j]
    return np.array(jj)

def set_jj(jj) :
    for i in range(6) :
        err = vrep.simxSetJointPosition(clientID, jjh[i], jj[i], vrep.simx_opmode_blocking)

def grip_on() :
    vrep.simxSetObjectParent(clientID, cube, bvp, 1, vrep.simx_opmode_blocking)
    # vrep.simxSetObjectPosition(clientID, cube, bvp, [0.0,0.0,0.0], vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(clientID)

def grip_off() :
    # err, pos = vrep.simxGetObjectPosition(clientID, cube, table, vrep.simx_opmode_blocking)
    vrep.simxSetObjectParent(clientID, cube, -1, 1, vrep.simx_opmode_blocking)
    # vrep.simxSetObjectPosition(clientID, cube, table,pos, vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(clientID)



LLL = 55250
pics = np.zeros((1, 5, 256, 256, 3), dtype='float16')
jjj = np.zeros((1, 5, 6), dtype='float16')
g_bool = False

cam_pos = np.load("./data_simpler_2/kamera.npy")
II = 93988    ###1225  pro 4_2 # 1440.. blizko kostky

vrep.simxSetObjectPosition(clientID, cam, table, cam_pos[:3], vrep.simx_opmode_blocking)
vrep.simxSetObjectOrientation(clientID, cam, table, cam_pos[3:], vrep.simx_opmode_blocking)

for i in range(5) :
    pkl_file = open('./data_simpler_2/{:07}.pkl'.format(II + i), 'rb')
    pos = pickle.load(pkl_file)

    vrep.simxSetObjectPosition(clientID, cube,m000 , pos['cube_pos'], vrep.simx_opmode_blocking)
    vrep.simxSetObjectOrientation(clientID, cube,-1 , [0,0,0], vrep.simx_opmode_blocking)

    set_jj(pos["joints"])

    err, res, im = vrep.simxGetVisionSensorImage(clientID, cam, 0,  vrep.simx_opmode_blocking)
    im = np.array(im, dtype="uint8").reshape(res+[3]).astype('float16')
    pics[0,i,:,:,:] = im / 255.0
    vrep.simxSynchronousTrigger(clientID)

jjj[0,] /= 2.7
# time.sleep(5)


for l in range(LLL) :
    print(l)
    jj = get_jj()
    aa = model.predict_on_batch(  [pics,jjj] )[0]
    # print(aa)
    print("dj: ", aa[:6])
    print("gg: ", aa[6])
    # aa[9] += 0.2
    print("cube: ", aa[7:10])
    vrep.simxSetObjectPosition(clientID, m1, m000, aa[7:10], vrep.simx_opmode_blocking)
    print("gripper: ", aa[10:13])
    vrep.simxSetObjectPosition(clientID, m2, m000, aa[10:13], vrep.simx_opmode_blocking)
    print("box: ", aa[13:])
    vrep.simxSetObjectPosition(clientID, m3, m000, aa[13:], vrep.simx_opmode_blocking)

    djj = aa[:6] / 4.0
    gg = aa[6]
    if not g_bool and gg > 0.0 :
        grip_on()
        g_bool = True
    if g_bool and gg < 0.0 :
        grip_off()
        g_bool = False

    set_jj(jj + djj)

    err, res, im = vrep.simxGetVisionSensorImage(clientID, cam, 0,  vrep.simx_opmode_blocking)
    im = np.array(im, dtype="uint8").reshape(res+[3]).astype('float16')
    # print(im.shape, im.max(), im.min())

    for i in range(2, 5) :
        pics[0,i-1,:,:,:] = pics[0,i,:,:,:]
        jjj[0,i-1,:] = jjj[0,i,:]

    pics[0,4,:,:,:] = im / 255.0
    jjj[0,4,:] = ( jj + djj) / 2.7

    vrep.simxSynchronousTrigger(clientID)
    time.sleep(0.1)


vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
vrep.simxFinish(clientID)
