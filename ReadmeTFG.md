# Unitree Go1 - Execution Guide (Docker Environment)

Quick reference guide for connecting to the robot's configured Docker container and executing the pipeline modules.

---

## Enter the Docker Environment

Open a terminal on the robot's onboard computer and run the connection script to access the pre-configured container:

```bash
./conectar_robot
```

---

## Open a Terminal to run Rviz

```
rosrun rviz rviz
```

---

## Open a Terminal to move robot on highlevel

```
roslaunch unitree_legged_real real.launch ctrl_level:=highlevel
```

---


## Open a Terminal to run Semantic Map

```
python3.8 navegacion_semantica_pub.py
```

---

## Open a Terminal to run Behavior Tree (BT)

```
roslaunch control control.launch
```

---

## Open a Terminal to visualize the BT

```
rosrun rqt_py_trees rqt_py_trees
```

