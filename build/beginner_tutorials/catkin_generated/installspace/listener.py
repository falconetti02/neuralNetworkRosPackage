import rospy
from std_msgs.msg import String
import numpy as np
import pandas as pd
import os, rospkg
 

def activ_func(inp): #leaky relU aktivation function 
    if inp < 0:
        return 0.01 * inp
    else:
        return inp

def ileriYay( newDataFrame, bias1, bias2, w1_array, w2_array, w3_array, wo_array): # forward propagation 
    #hidden layer 1

    hid = newDataFrame.dot(w1_array) # multiply input values with weights for first hidden layer
    hid1 = hid + bias1 # add bias values

    hid1 = float(activ_func(hid1)) # put the value in activation function

    #hidden layer 2
   
    hid = newDataFrame.dot(w2_array)  # multiply input values with weights for second hidden layer
    hid2 = hid + bias1 # add bias values
   
    hid2 = float(activ_func(hid2)) # put the values in activation function
       
    #hidden layer 3
    
    hid = newDataFrame.dot(w3_array) # multiply input values with weights for third hidden layer
    hid3 = hid + bias1 # add bias values
    
    hid3 = float(activ_func(hid3)) # put the values in activation function
        
    #out layer
  
    matris = np.array([hid1,hid2,hid3])

    out = matris.dot(wo_array) # multiply three hidden layer values with weights for output
    out1 = out + bias2 # add bias values
       
    final_output = float(activ_func(out1))  # put the values in activation function
    
    return final_output

def callback(data):
    data_str = data.data.split(',')
    input_data = data_str[1:5]
    # convert input_data to float array
    input_data_float = []
    for inp in input_data:
        try:
            input_data_float.append(float(inp))
        except ValueError:
            pass # ignore non-float values
    input_data = np.array(input_data_float)

    flower_name = data_str[-1]
    
    out = ileriYay(np.transpose(input_data), nbias1, nbias2, nw1_array, nw2_array, nw3_array, nwo_array)
    detect_flower(out,flower_name)

def listener():
    
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('string', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def detect_flower(out1, flower_name): # detecting flower type if it is between defined values

    if  -1 < out1 < 0.7: 
        rospy.loginfo(f"flower is Iris-setosa  {flower_name}")
    elif 0.7 < out1 < 1.6:
        rospy.loginfo(f"flower is Iris-versicolor  {flower_name}")
    elif  1.6 < out1 < 5:  
        rospy.loginfo(f"flower is Iris-virginica  {flower_name}")
    else:
        rospy.loginfo(f"cannot detect    {flower_name}")
    

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    
    w1_array = pd.read_csv(os.path.join(rospack.get_path("beginner_tutorials"), "data", "w1_array.csv"))
    w1_array = np.array(w1_array.columns)
    id = 0
    nw1_array = np.empty((4))
    for x in w1_array:
        nw1_array[id] = float(x)
        id = id + 1
    
    w2_array = pd.read_csv(os.path.join(rospack.get_path("beginner_tutorials"), "data", "w2_array.csv"))
    w2_array = np.array(w2_array.columns)
    id = 0
    nw2_array = np.empty((4))
    for x in w2_array:
        nw2_array[id] = float(x)
        id = id + 1

    w3_array = pd.read_csv(os.path.join(rospack.get_path("beginner_tutorials"), "data", "w3_array.csv"))
    w3_array = np.array(w3_array.columns)
    id = 0
    nw3_array = np.empty((4))
    for x in w3_array:
        nw3_array[id] = float(x)
        id = id + 1

    wo_array = pd.read_csv(os.path.join(rospack.get_path("beginner_tutorials"), "data", "wo_array.csv"))
    wo_array = np.array(wo_array.columns)
    id = 0
    nwo_array = np.empty((3))
    for x in wo_array:
        nwo_array[id] = float(x)
        id = id + 1

    bias1 = pd.read_csv(os.path.join(rospack.get_path("beginner_tutorials"), "data", "bias1.csv"))
    bias1 = np.array(bias1.columns)
    id = 0
    nbias1 = np.empty((1))
    for x in bias1:
        nbias1[id] = float(x)
        id = id + 1
    
    bias2 = pd.read_csv(os.path.join(rospack.get_path("beginner_tutorials"), "data", "bias2.csv"))
    bias2 = np.array(bias2.columns)
    id = 0
    nbias2 = np.empty((1))
    for x in bias2:
        nbias2[id] = float(x)
        id = id + 1
    
    listener()
