#!/usr/bin/env python2.7

from collections import deque
import struct

import numpy as np
import rospy

from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32, Image, CameraInfo
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32

from tf.transformations import quaternion_matrix
from cv_bridge import CvBridge, CvBridgeError


class ColorizedMap:
    """
    ROS node that produces a colorized map of environment for a turtlebot3. Uses odometry, lidar data,
    and 3-channel camera to produce visualizations in RViz. See setup details in README.MD

    With computational constraints, map_size sets the number of colorized, world-frame lidar scans to be
    stores. When the map size is reached, the last one is removed.

    Tolerance sets the distance the robot needs to travel before we process another lidar scan,
    to improve the interesting-ness of the map. Otherwise we would process a new map every time and sitting
    still would make you forget everywhere you've been before.
    """
    def __init__(self, map_size=10, rate=5, tolerance=0.5):
        """
        Constructor

        Params
        ======
        map_size: number of processed lidar scans to store (each scan has ~300 points)
        rate: in hertz, the frequency of map publication
        tolerance: in meters, how far the robot should have to move before processing another scan
        """
        # configurable parameters of the class
        self.map_size = map_size
        self.rate = rate
        self.tolerance = tolerance

        # the colorized map product
        self.maps_deque = deque([])
        self.colors_deque = deque([])

        # transformation products
        self.position = None # from robot body-frame to robot world-frame
        self.rotation = None # from robot body-frame to robot world-frame

        # needed to figure out how the robot and the camera coordinate system relate
        # by reading the documentation on camera_info here: https://wiki.ros.org/image_pipeline/CameraInfo
        # you can see how the robot's xyz differs from the camera's xyz
        self.robot_to_camera = np.array([[ 0,  0,  1],
                                         [-1,  0,  0],
                                         [ 0, -1,  0]]).T

        # camera transformation products
        self.camera_matrix = None
        self.camera_height = None
        self.camera_width = None

        # to convert from ROS (encoded) Image message to processable 3D datacube
        self.bridge = CvBridge()

        # store last position to only update map if we've moved more than tolerance
        self.last = np.full((3,1), np.inf)

        # subscribers & init node
        self.map_publisher = rospy.Publisher('colorized_map', PointCloud, queue_size=10)
        rospy.init_node('colorized_map')
        rospy.Subscriber('scan', LaserScan, self.update_map)
        rospy.Subscriber('odom', Odometry, self.store_pose)
        rospy.Subscriber('camera/rgb/image_raw', Image, self.colorize)
        rospy.Subscriber('camera/rgb/camera_info', CameraInfo, self.store_camera_info)

        # we will publish on a timer set by rate
        rospy.Timer(rospy.Duration(1.0/self.rate), self.publish_map)

    def store_pose(self, msg):
        """
        Callback method for ROS subscriber. Converts odometry message including the quaternion into
        rotation matrix and offset to transform body frame coordinates to world frame.

        Params
        ======
        msg: ROS Odometry message
        """
        position = np.array(
                        [msg.pose.pose.position.x,
                         msg.pose.pose.position.y,
                         msg.pose.pose.position.z
                        ]).reshape((3,1))

        # the turtlebot3 uses quaternions, so this is to convert to a rotation matrix
        rotation = quaternion_matrix((
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ))[:3,:3] # since it returns a 4x4 matrix

        # another coordinate transformation. Need to convert robot position to lidar position (lidar pose extracted from model file)
        robot_to_lidar_pos = np.array([[-0.052], [0], [0.111]])
        self.position = position - robot_to_lidar_pos
        self.rotation = rotation

    def store_camera_info(self, msg):
        """
        Callback method for ROS subscriber. Stores transformation matrix from camera_info.
        The matrix transforms 3D body frame coordinates to 2D pixel coordinates.

        The camera_info message contains the image width & height for the *compressed* images
        being published, so the values for the raw RBG image are hardcoded.

        Params
        ======
        msg: ROS CameraInfo message
        """
        # this only needs to get set once
        if self.camera_matrix is None:
            transformation_matrix = np.array(list(msg.K), dtype=float).reshape((3,3))
            self.camera_matrix = transformation_matrix
            # this information is not actually being published in the message, the published width and height are for the compressed images
            self.camera_width = 1080  #int(msg.width)
            self.camera_height = 1920 #int(msg.height)

    def update_map(self, msg):
        """
        Callback method for ROS subscriber. Updates map.
        Converts LaserScan message to cartesian 3D point array, removing the invalid points
        then rotate those points to world_frame.

        Params
        ======
        msg: ROS LaserScan message
        """
        # since this all happens asynchronously, we could call this method before the rotation matrix was stored
        if (self.position is None) or (self.rotation is None):
            return
        # checks that the robot has moved more than our tolerance amount, otherwise skip
        # making another map
        elif np.linalg.norm(self.position - self.last) < self.tolerance:
            return
        else:
            # if we are updating the map, update our last position to reset the tolerance check
            self.last = self.position.copy()

        # start of laserscan message conversion from polar to cartesian
        ranges = np.array(msg.ranges, float)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # only use the valid data
        valids = np.isfinite(ranges)
        ranges = ranges[valids]
        angles = angles[valids]

        # laser scan points are in polar coordinates, need to convert to cartesian
        body_points = np.row_stack((
            ranges*np.cos(angles),
            ranges*np.sin(angles),
            np.zeros_like(angles)
        ))

        # rotate our laserscan points from body-frame to world-frame
        points = self.rotation.dot(body_points) + self.position
        colors = np.full_like(points, np.nan)

        self.maps_deque.append(points)
        self.colors_deque.append(colors)
        assert len(self.maps_deque) == len(self.colors_deque)

        # for computational complexity, can only store so many laser scans
        if len(self.maps_deque) > self.map_size:
            self.colors_deque.popleft()
            self.maps_deque.popleft()

    def is_in_front(self, point):
        """
        Checks if point is in front of robot or behind it by dotting robot heading
        vector (first column of rotation matrix) with point vector.
        Needs to recenter the world-frame point at the robot's current position.

        Without this check, the transformation in colorize cannot distinguish between points
        behind and in front of the robot, so the robot would colorize all points on the same
        ray through the camera image the same color (a green pillar in front of the robot would
        mean it colored the points behind it - out of view - green too).

        Params
        ======
        point: np.array of shape (3,)

        Returns
        =======
        boolean: True if in front, False if behind
        """
        return np.dot(self.rotation[:, 0], point.reshape((3,1))-self.position) > 0

    def colorize(self, msg):
        """
        Callback method for ROS subscriber. Colorizes map.

        Params
        ======
        msg: ROS Image message
        """
        # since this all happens asynchronously, we could call this method before the rotation matrix was stored
        if self.camera_matrix is None or self.rotation is None:
            return

        # converting from the (encoded) Image message data to a processable image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image = np.array(cv_image, dtype=float)

        for (i, mapp), colors in zip(enumerate(self.maps_deque), self.colors_deque):
            for (j, world_point), color in zip(enumerate(mapp.T), colors.T):
                # if point is uncolored and in front of the robot
                if np.isnan(color[0]) and self.is_in_front(world_point):
                    # camera transformation matrix goes from body-frame points to camera coordinates, need to convert our
                    # world-frame points to body-frame again.
                    body_point = np.dot(self.rotation.T, world_point.reshape((3,1)) - self.position)
                    # need to convert the body-frame points to the camera coordinate system (still in body-frame)
                    point = np.dot(self.robot_to_camera, body_point - (np.array([[0.064], [-0.047], [0.107]]) - np.array([[-0.052], [0], [0.111]])))
                    # get camera coordintes
                    u, v, w = np.dot(self.camera_matrix, point)
                    cam_x = float(u/w)
                    cam_y = float(v/w)

                    # if the point is in the image, proceed
                    if (cam_x >= 0 and cam_x < self.camera_width) and (cam_y >= 0 and cam_y < self.camera_height):
                        # get pixel coordinates in the camera image
                        x, y = int(cam_x), int(cam_y)

                        # get the color of the point based on where it intersects the image plane
                        rgb = (image[x, y, 0]/256.0, image[x, y, 1]/256.0, image[x, y, 2]/256.0)
                        colors[:, j] = rgb

    def publish_map(self, _):
        """
        At rate specified in init, publish the new map.

        Converts internal colorized map to pointcloud message, as that allows for both cartesian
        coordinates and, most importantly, setting a color channel for the points.
        """
        map_msg = PointCloud()

        map_msg.header.stamp = rospy.get_rostime()
        map_msg.header.frame_id = "odom"

        # RViz expects there to be three color channels, with each point having a value in each color
        map_msg.channels = [ChannelFloat32('r', []), ChannelFloat32('g', []), ChannelFloat32('b', [])]

        # convert our internal representation to a pointcloud message
        for (points, colors) in zip(self.maps_deque, self.colors_deque):
            for (point, color) in zip(points.T, colors.T):
                map_msg.points.append(Point32(*point))
                if np.isnan(color[0]):
                    # if the point still isn't colored when we publish, color it blue
                    color = (0.0, 0.0, 0.8)
                map_msg.channels[0].values.append(color[0])
                map_msg.channels[1].values.append(color[1])
                map_msg.channels[2].values.append(color[2])

        self.map_publisher.publish(map_msg)

    def run(self):
        """
        Ensuring the node continues to run & sleeps in the appropriate way. May
        be more built-out later.
        """
        rospy.spin()


if __name__ == '__main__':
    ColorizedMap(tolerance=0.5
        ).run()