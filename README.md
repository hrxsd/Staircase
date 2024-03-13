# Build
Create workspace, download into **src** folder.
`catkin_make`
# Use
Initiate a network of staircase identification nodes：
*Go to the folder where **deploy.py** is located, open the terminal and type*
```
python3 deploy.py
```
Start trajectory generation related nodes：
```
roslaunch simulator test.launch
rosrun controller path2array.py
```
Launching the Local Planner and High-Level Controller
```
rosrun controller local_planner.py
rosrun controller controller_go1.py
```


