# Robotics Project

Colorizing 2D laserscan data via imagery for turtlebot3 models.

**(Partial) Setup Instructions:**
Follow instructions [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/) for installing ROS, Gazebo, and the turtlebot3 ROS package.

Then via terminal:
`export TURTLEBOT3_MODEL=waffle`
`roslaunch turtlebot3_gazebo turtlebot3_world.launch`
`roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch`

**To see camera data**
In order to see raw RGB camera data from waffle turtlebot3 in RVIZ, navigate to RVIZ after it opens from last command:
- click `add` (bottom left button in RVIZ)
Go to `by topic`, then
-> `/camera` -> `/rgb` -> `/image_raw` -> `Image`
