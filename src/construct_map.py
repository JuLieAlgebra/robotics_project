#!/usr/bin/env python2.7

from datetime import datetime, timedelta

import numpy as np

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from nav_msgs.msg import Odometry


class MappingNode:
    """ Publisher/subscriber node that handles constructing 2D map from /scan and /odom topics. """
    def __init__(self):
        # subscribers
        self.laser_sub = rospy.Subscriber('scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)

        # publisher
        self.pub = rospy.Publisher('map', LaserScan, queue_size=10)

        # processed results
        self.map = []
        self.map_size = 50
        self.odom = None

    def laser_process(self, msg):
        """ Processing laser scan data from subscriber callback message """
        print(type(msg), type(msg.header), len(msg.ranges), len(msg.intensities))

        # this is going to use laser_geometry to convert the laser_scan to pointcloud data type (still in robo frame)
        converted = msg.copy()

        # then need to rotate all of the data points to be in the right frame based on odom data


        # queue data structure so we don't endlessly accumulate points
        self.map.append(converted)
        if len(self.map) > self.map_size:
            self.map.pop(0)

        return converted

    def odom_process(self, msg):
        """ Processing odometry data from subscriber callback message """
        return msg.pose.pose

    def laser_callback(self, msg):
        # if self.odom isn't set yet, then since we started, we haven't heard an odom publication
        if self.odom:
            # rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg)
            processed_data = self.laser_process(msg)
            self.pub.publish(processed_data)

    def odom_callback(self, msg):
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg)
        processed_data = self.odom_process(msg)
        # self.odom_pub.publish(processed_data)
        self.odom = processed_data

    def run(self):
        """
        Ensuring the node continues to run & sleeps in the appropriate way. May
        be more built-out later.
        """
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('mapper')
    mapper = MappingNode()
    mapper.run()