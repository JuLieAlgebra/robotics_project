# Robotics Project

Colorizing 2D laserscan data via imagery for turtlebot3 waffle and waffle_pi models with ROS and Gazebo.

**Setup Instructions:**\
Follow instructions [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/) for installing ROS, Gazebo, and the turtlebot3 ROS package.\

Then via terminal: \
`export TURTLEBOT3_MODEL=waffle`  or `export TURTLEBOT3_MODEL=waffle_pi` \
`roslaunch turtlebot3_gazebo turtlebot3_world.launch` \
`roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch` \

To drive turtlebot3: \
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch \

**To build & launch ColorizedMap node:**
Clone this repo to wherever you store your ROS packages (usually `~/catkin_ws/src/`) via: `git clone git@github.com:JuLieAlgebra/robotics_project.git` 
Make sure to `chmod +x ~/catkin_ws/src/robotics_project/src/colormap.py`, though replace the absolute directory with your own if you have a different structure. 
If you use catkin build for ROS packages, navigate to `~/catkin_ws/src` or wherever you store your ROS packages, then `catkin build` in the directory. 
Then launch node with `rosrun robotics_project colormap.py` 

Run RViz with `rviz -d ~/catkin_ws/src/robotics_project/rviz/colored_point_cloud.rviz`, but with your own absolute path if it deviates.
<!-- **To see camera data**
In order to see raw RGB camera data from waffle turtlebot3 in RVIZ, navigate to RVIZ after it opens from last command:\
- click `add` (bottom left button in RVIZ) \
- Go to `by topic` on top right of pop-up window, then
-> `/camera` -> `/rgb` -> `/image_raw` -> `Image` \

**To see Colorized Map data**
In order to see the output colorized map in RVIZ, navigate to RVIZ after it opens from rviz launch command: \
- click `add` (bottom left button in RVIZ)\
- Go to `by topic` on top right of pop-up window, then
-> `/colorized_map` -> `PointCloud`

If the Intensity channel of the colorized_map RVIZ topic is set to anything but `RGB`, click on the drop down menu associated with the Intensity box, then set it. The robot updates its lidar map only upon moving 0.5 meters from its last position, by default, so drive around with teleop and see what you get! -->