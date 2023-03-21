import rospy
from std_msgs.msg import String
import csv
import os, rospkg

rospack = rospkg.RosPack()

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    # get dataframe from test dataset csv file
    # get every line in dataframe in a row and publish 
    with open(os.path.join(rospack.get_path("beginner_tutorials"), "data", "test_dataset.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile) 
        for row in reader:  
            row_str = ','.join(row)  # Convert the row list to a string
            rospy.loginfo(row_str)
            pub.publish(row_str)
            rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass




