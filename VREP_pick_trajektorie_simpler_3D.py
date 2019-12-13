import vrep
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

data_path = "./data_00/"

print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP


if clientID!=-1:
    print ('Connected to remote API server')
    vrep.simxSynchronous(clientID,True)
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
    # err , ttt = vrep.simxGetFloatSignal(clientID, "mySimulationTime", vrep.simx_opmode_streaming)
    joint = []
    grip_num = 0

    err, pos_zero = vrep.simxGetObjectHandle(clientID, "UR5_joint1", vrep.simx_opmode_blocking)
    err, cube = vrep.simxGetObjectHandle(clientID, "cub", vrep.simx_opmode_blocking)
    err, box = vrep.simxGetObjectHandle(clientID, "box", vrep.simx_opmode_blocking)
    err, gripper = vrep.simxGetObjectHandle(clientID, "BaxterVacuumCup_link",vrep.simx_opmode_blocking)
    err, cam1 = vrep.simxGetObjectHandle(clientID, "sensor_1",vrep.simx_opmode_blocking)
    err, cam2 = vrep.simxGetObjectHandle(clientID, "sensor_2",vrep.simx_opmode_blocking)
    err, table = vrep.simxGetObjectHandle(clientID, "diningTable",vrep.simx_opmode_blocking)

    for i in range(6) :
        err, jj = vrep.simxGetObjectHandle(clientID, "UR5_joint{}".format(i+1), vrep.simx_opmode_blocking)
        joint += [jj]

    time.sleep(0.5)

    err, cam_pos = vrep.simxGetObjectPosition(clientID,cam1,table,vrep.simx_opmode_blocking)
    err, cam_or = vrep.simxGetObjectOrientation(clientID,cam1,table,vrep.simx_opmode_blocking)
    np.save( data_path + "kamera", np.array(cam_pos + cam_or) )

    err, cam_pos = vrep.simxGetObjectPosition(clientID,cam2,table,vrep.simx_opmode_blocking)
    err, cam_or = vrep.simxGetObjectOrientation(clientID,cam2,table,vrep.simx_opmode_blocking)
    np.save( data_path + "kamera", np.array(cam_pos + cam_or) )

    print("cameras positioin and orientation saved")
    # time.sleep(10000)
    count = 0
    # for ccc in range(0,100000):
    while True :

        err, oI, oF, oS, oB = vrep.simxCallScriptFunction(clientID,"sensor_1", vrep.sim_scripttype_childscript, "getData", \
            [],[],"","", vrep.simx_opmode_blocking)
        if oI == [] :
            print("\nNo oI: ")
            vrep.simxSynchronousTrigger(clientID);
            continue
        state = oI[0]
        gripper_state = oI[1]

        if state == 0 :
            grip_num += 1
        if state == 99 :
            break
        if state == 0 :
            vrep.simxSynchronousTrigger(clientID);
            continue



        print("\nCycle: ", count)
        print("Grip: ", grip_num)
        print("State, gripper:", oI) # stav 1-3 or 99
        joints = oF[:6]
        print("Joints: ", joints) # joint

        tt = vrep.simxGetLastCmdTime(clientID)
        tt_sim = oF[7]
        dt = oF[6]
        # print("Times: ", tt_sim, dt) # joint
        # print("LastCmdTime: ", tt/1000.0, "s")
        # print(oS)
        # print(type(oB)) # img

        # try :
        # img = np.array(oB).reshape(256,256,3)
        # np.save( data_path + "img{:07}".format(count),img)
        err, res, im = vrep.simxGetVisionSensorImage(clientID, cam1, 0,  vrep.simx_opmode_blocking)
        im = np.array(im, dtype="uint8").reshape(res+[3])  #.astype('float16')
        # pics[0,i,:,:,:] = im / 255.0
        np.save( data_path + "img{:07}_1".format(count),im)
        print("Image1: yes")

        err, res, im = vrep.simxGetVisionSensorImage(clientID, cam2, 0,  vrep.simx_opmode_blocking)
        im = np.array(im, dtype="uint8").reshape(res+[3])
        np.save( data_path + "img{:07}_2".format(count),im)
        print("Image2: yes")

        # except :
        #     print("Image: no")


        err, cube_pos = vrep.simxGetObjectPosition(clientID,cube,pos_zero,vrep.simx_opmode_blocking)
        # print("Cube pos: ", cube_pos)
        err, box_pos = vrep.simxGetObjectPosition(clientID,box,pos_zero,vrep.simx_opmode_blocking)
        # print("Box pos: ", box_pos)
        err, gripper_pos = vrep.simxGetObjectPosition(clientID,gripper,pos_zero,vrep.simx_opmode_blocking)
        # print("Gripper pos: ", gripper_pos)

        data = {}
        data["joints"] = joints
        data["state"] = state
        data["gripper"] = gripper_state
        data["cube_pos"] = cube_pos
        data["box_pos"] = box_pos
        data["gripper_pos"] = gripper_pos


        f = open( data_path + "{:07}.pkl".format(count),"wb")
        pickle.dump(data,f)
        f.close()

        vrep.simxSynchronousTrigger(clientID)
        # time.sleep(0.2)
        if oI[0] == 1 :
            count += 1
        if count > 1e5 :
            break

    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
