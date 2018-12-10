This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: **Programming a Real Self-Driving Car**. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).
We are submitting this project as a team effort.  Here is the team composition.

[//]: # (Image References)
[image1]: ./architecture.jpg



## Team Name:  teamCarla

 
 | Num  | Name              | Email                  | Role   |
 |------|-------------------|------------------------|--------|
 | 1    | Jan Rosicky       | janrosicky@centrum.cz  | Member |
 | 2    | Komuraiah Poodari | poodari@yahoo.com      | Member |
 | 3    | Rakesh Chittineni | rakeshch.net@gmail.com | Member |
 | 4    | Santosh Bhushan   | 2sbsbsb@gmail.com      | Leader |
 
 
## System Architecture
This project is based on the architecture framework provided by Udacity as per the following system architecture diagram.

![alt text][image1]

Source: Udacity Project Lesson section 4.

### Perception Subsystem
We implemented traffic light detection by processing the camera image on need basis.  For example, processing image with minimum time gap (few milliseconds) and only when the traffic light ahead is few hundred waypoints ahead of current car position.
The objective is to optimize the processing on the platform.

We used closest distance measure between car waypoint and traffic light waypoint to identify the closest light ahead of the car. We used a pre-trained convolutional neural network model for inferencing the traffic light classification. We used two models, one each for site inferencing and simulator inferencing.

### Planning Subsystem
Planning subsystem has two nodes, viz., `waypoint loader` and `waypoint updater` nodes.  
The node `waypoint loader` loads the complete set of traffic waypoints for the entire track.
The node `waypoint updates` updates next set of waypoints ahead of the car.  We used only 50 look ahead waypoints to optimize processing.
We implemented to have the car stopped 2 waypoints ahead of the traffic light waypoint ahead, the signal warrants a stop.

### Control Subsystem
This module contains the following nodes.
The node, `DBW (Drive By Wire)` computes control inputs for actuators, `throttle`, `brake` and `steering angle` based on the desired velocities.
We implemented the main control logic in `twist_controller.py` using low pass filter, yaw_controller and PID controller given in the project. We had to play with various parameters.


### Reflection
We had to spend good amount of bandwidth on making environment setup stable, as there are quite a few options and the project needs high-end platform with GPU.  We look forward to running our implementation on real car, **Carla** and get feedback.

The rest of the sections are unmodified sections came with the project baseline.

Please use **one** of the two installation options, either native **or** docker installation. As a team some of us used native installation, some of us used docker based installation.


### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
