import rospy
from std_msgs.msg import String
import os, rospkg
import numpy as np
import pandas as pd

rospack = rospkg.RosPack()

def talker():
    pubStr = rospy.Publisher('string', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    # get dataframe from test dataset csv file
    # get every line in dataframe in a row and publish 
    df = pd.read_csv(os.path.join(rospack.get_path("beginner_tutorials"), "data", "test_dataset.csv"))
    df = np.array(df)
    
    for row in df:  
        row_str = ','.join(str(x) for x in row)
        rospy.loginfo(row_str)
        pubStr.publish(row_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass


