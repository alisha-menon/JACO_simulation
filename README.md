# jaco_learning: Jaco HRI

Control, planning, and learning system for human-robot interaction with a JACO2 7DOF robotic arm. Supports learning from physical corrections and demonstrations.

## Dependencies
* Ubuntu 14.04, ROS Indigo, OpenRAVE, Python 2.7
* or_trajopt, or_urdf, or_rviz, prpy, pr_ordata
* kinova-ros
* fcl

## Running the Adaptive Physical HRI System
### Setting up the JACO2 Robot
Turn the robot on and put it in home position by pressing and holding the center (yellow) button on the joystick.
 
In a new terminal, turn on the Kinova API by typing:
```
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2s7s300 use_urdf:=true
```
### Starting the controller and planner
To demonstrate simple path planning and control with the Jaco arm, run (in another terminal window):
```
roslaunch jaco_learning path_follower.launch
```
The launch file first reads the corresponding yaml `config/path_follower.yaml` containing all important parameters, then runs `path_follower.py`. Given a start, a goal, and other task specifications, a planner plans an optimal path, then the controller executes it. For a selection of planners and controllers, see `src/planners` (TrajOpt supported currently) and `src/controllers` (PID supported currently). The yaml file should contain parameter information to instantiate these two components.

Some important parameters for specifying the task in the yaml include:
* `start`: Jaco start configuration
* `goal`: Jaco goal configuration
* `goal_pose`: Jaco goal pose (optional)
* `T`: Time duration of the path
* `timestep`: Timestep dicretization between two consecutive waypoints on a path.
* `feat_list`: List of features the robot's internal representation contains. Options: "table" (distance to table), "coffee" (coffee cup orientation), "human" (distance to human), "laptop" (distance to laptop).
* `feat_weights`: Initial feature weights.

### Learning from physical human corrections
To demonstrate planning and control with online learning from physical human corrections, run:
```
roslaunch jaco_learning phri_inference.launch
```
The launch file first reads the corresponding yaml `config/phri_inference.yaml` containing all important parameters, then runs `phri_inference.py`. Given a start, a goal, and other task specifications, a planner plans an optimal path, and the controller executes it. A human can apply a physical correction to change the way the robot is executing the task. Depending on the learning method used, the robot learns from the human torque accordingly and updates its trajectory in real-time.

Some task-specific parameters in addition to the ones above include:
* `learner/type`: Learning method used.
  * all = update all features at once, according to A. Bajcsy* , D.P. Losey*, M.K. O'Malley, and A.D. Dragan. [Learning Robot Objectives from Physical Human Robot Interaction](http://proceedings.mlr.press/v78/bajcsy17a/bajcsy17a.pdf) Conference on Robot Learning (CoRL), 2017.
  * max = update one feature at a time, according to A. Bajcsy , D.P. Losey, M.K. O'Malley, and A.D. Dragan. [Learning from Physical Human Corrections, One Feature at a Time](https://dl.acm.org/citation.cfm?id=3171267) International Confernece on Human-Robot Interaction (HRI), 2018.
  * beta = relevance adaptive method according to our method: A. Bobu, A. Bajcsy, J. Fisac, A.D. Dragan. [Learning under Misspecified Objective Spaces](http://proceedings.mlr.press/v87/bobu18a.html) Conference on Robot Learning (CoRL), 2018.
* `save_dir`: Location for saving human data (optional). After the run, you will be prompted to save the collected data.

### Learning from physical human demonstrations
To demonstrate learning from human demonstrations, first record some demonstrations:
```
roslaunch jaco_learning demo_recorder.launch
```
The launch file first reads the corresponding yaml `config/demo_recorder.yaml` containing all important parameters, then runs `demo_recorder.py`. Given a start, the Jaco is controlled to the initial location, after which it waits for human input. Once the start is reached, the person can physically direct the arm to demonstrate how it should perform the task. The user is then shown the collected the demonstration and prompted to save it.

To perform inference from human demonstrations, run:
```
python src/demo_inference.py config/demo_inference.yaml
```
The script loads the yaml file given as argument, then performs inference. There is an option to perform inference from either recorded demonstrations or a simulated one, created using the planner. Inference is performed according to our method: A. Bobu, A. Bajcsy, J. Fisac, S. Deglurkar, and A.D. Dragan. Quantifying Hypothesis Space Misspecification in Learning from Human-Robot Demonstrations and Physical Corrections.

### References
* https://github.com/abajcsy/iact_control
* TrajOpt Planner: http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/index.html
* PID Control Reference: https://w3.cs.jmu.edu/spragunr/CS354/handouts/pid.pdf
