#!/usr/bin/env python2.7

import rospy
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge, CvBridgeError


class ColorizingNode:
    def __init__(self):
        pass

    def run(self):
        """
        Ensuring the node continues to run & sleeps in the appropriate way. May
        be more built-out later.
        """
        rospy.spin()