import tdmclient.notebook
await tdmclient.notebook.start()
import cv2
import time
import numpy as np
import math
import functions as fct
from navigation import *
from obstacleDetection import *
import global_navigation as gn
from IPython.display import clear_output
import pyvisgraph as vg


# Initialization variables
thymio_connected = True
warp = True
debug_mode = True
widthImg = 1080
heightImg = 720

keep_same_map = False
ghost = False
pause = False
dt = 0.0
pTime = 0.0
my_print = []
shortest_path = []
polygons = []
goal_pos = [0, 0]
thymio_state = np.matrix([[0.0],[0.0],[0.0],[0.0],[0.0]])    # X, Y, angle, v, v_rot
measures = np.matrix([[0.0],[0.0],[0.0],[0.0],[0.0]])        # X, Y, angle, v, v_rot
measure_cam = np.matrix([[0.0],[0.0],[0.0]])                 # X, Y, angle
measure_rob = np.matrix([[0.0], [0.0]])                      # v, v_rot

# Initialize shape of the map and color tracker
matrix = fct.warp_matrix(widthImg, heightImg, "data1.txt")
tracker = fct.DetectionThymio("data2.txt")

# Initialize Kalaman filter
Kalman = fct.KalmanFilter(dt, thymio_state)

# Initialize PID regulators
PID = PID(Kp=Kp, Ki=Ki, Kd=Kd, sumError=0, subError=0, errorPrev=0, maxSumError=maxSumError)
robot = Robot(x=measure_cam[0], y=measure_cam[1], angle=measure_cam[2], state=GLOBAL)


def open_camera():
    
    # Open camera feed
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream")

    # Set width and height parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, widthImg)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, heightImg)
    
    return cap


@tdmclient.notebook.sync_var
def set_motors(speed_l, speed_r):
    global globalL, globalR
    globalL = speed_l
    globalR = speed_r


@tdmclient.notebook.sync_var
def get_motors():
    global motor_left_speed, motor_right_speed
    forwSpeed = (motor_left_speed + motor_right_speed) / 2.0
    rotSpeed = (motor_left_speed - motor_right_speed) / 2.0
    speed = [forwSpeed, rotSpeed]
    return speed


@tdmclient.notebook.sync_var
def stop_motors():
    global stop_state
    stop_state = True


@tdmclient.notebook.sync_var
def start_motors():
    global stop_state
    stop_state = False


def picture_of_the_map(cap, file_name):
    global matrix, tracker, Kalman, PID, robot, keep_same_map, thymio_connected, ghost, pause, dt, pTime, my_print, shortest_path, polygons, goal_pos, thymio_state, measures, measure_cam, measure_rob

    while True:

        # Getting a frame and warping it
        ret, frame = cap.read()
        mymap = fct.warp_image(warp, frame, matrix, (widthImg, heightImg))

        # Show text
        show = mymap.copy()
        show = fct.draw_text(show, 'Push "P" to take a new picture, "Enter" otherwise', (255, 0, 0))

        cv2.imshow("Area", show)

        k = cv2.waitKey(1)  # Enter is pressed

        if k == 13:  # Press Enter to keep the same map
            keep_same_map = True
            break
        if k == ord("p"):  # Press P to save a new map
            cv2.imwrite(file_name, mymap)
            break


def path_planning(cap, file_name):
    global matrix, tracker, Kalman, PID, robot, keep_same_map, thymio_connected, ghost, pause, dt, pTime, my_print, shortest_path, polygons, goal_pos, thymio_state, measures, measure_cam, measure_rob

    while True:

        # Getting a frame and warping it
        ret, frame = cap.read()
        mymap = fct.warp_image(warp, frame, matrix, (widthImg, heightImg))

        # Find goal point and Thymio
        goal_pos = fct.find_template('GoalTemplate.png', mymap)
        thymio_visible, measure_cam = tracker.state(mymap)

        # Show Goal
        show = mymap.copy()
        show = fct.draw_goal(show, goal_pos, (0, 255, 255))

        # Show Thymio
        if thymio_visible:
            show = fct.draw_thymio(show, measure_cam, (255, 0, 0))

        show = fct.draw_text(show, 'Push "Enter" to compute the path', (255, 0, 0))
        cv2.imshow("Area", show)

        if 13 == cv2.waitKey(1):  # Enter is pressed
            start_pos = measure_cam
            thymio_state[0] = measure_cam[0]
            thymio_state[1] = measure_cam[1]
            thymio_state[2] = measure_cam[2]
            break

    cv2.destroyAllWindows()

    # Global navigation
    start_point = vg.Point(start_pos[0], start_pos[1])
    goal_point = vg.Point(goal_pos[0], goal_pos[1])

    if not keep_same_map:

        # Map processing
        visualsProfile1()
        settingsProfile1()
        chunkmap, polys, thymiowidth = analysePicture2()

        # Save map
        string = savePoints(chunkmap, polys, thymiowidth)
        text_file = open(file_name, "w")
        text_file.write(string)
        text_file.close()

        # Path planning
        shortest_path, polygons = gn.path_planning(polys, start_point, goal_point, thymiowidth / 1.8, gridMap = chunkmap)
        shortest_path.reverse()
    else:

        # Load map
        text_file = open(file_name, "r")
        string = text_file.read()
        text_file.close()
        thymiowidth, minx, miny, maxx, maxy, polys = getPolys(string)

        # Path planning
        shortest_path, polygons = gn.path_planning(polys, start_point, goal_point, thymiowidth / 1.8, minx = minx, maxx = maxx, miny = miny, maxy = maxy)
        shortest_path.reverse()


def pre_visualisation(cap):
    global matrix, tracker, Kalman, PID, robot, keep_same_map, thymio_connected, ghost, pause, dt, pTime, my_print, shortest_path, polygons, goal_pos, thymio_state, measures, measure_cam, measure_rob

    while True:

        # Getting a frame and warping it
        ret, frame = cap.read()
        mymap = fct.warp_image(warp, frame, matrix, (widthImg, heightImg))

        # Get the position
        thymio_visible, measure_cam = tracker.state(mymap)

        # Show obstacles
        show = mymap.copy()
        show = fct.draw_obstacles(show, polygons, (0, 255, 0))

        # Show path
        show = fct.draw_path(show, shortest_path, (0, 0, 255))

        # Show Goal
        show = fct.draw_goal(show, goal_pos, (0, 255, 255))

        # Show Thymio
        if thymio_visible:
            show = fct.draw_thymio(show, measure_cam, (255, 0, 0))

        show = fct.draw_text(show, 'Push "Enter" to start', (255, 0, 0))
        cv2.imshow("Area", show)

        if 13 == cv2.waitKey(1):  # Enter is pressed
            thymio_state = np.append(measure_cam, np.matrix([[0.0], [0.0]]), axis=0)
            start_motors()
            break


def running_loop(cap, video_name):
    global matrix, tracker, Kalman, PID, robot, keep_same_map, thymio_connected, ghost, pause, dt, pTime, my_print, shortest_path, polygons, goal_pos, thymio_state, measures, measure_cam, measure_rob

    out = cv2.VideoWriter(video_name, -1, 20.0, (1080,720))
    
    if len(shortest_path) > 1:
        while True:

            # Getting a frame
            ret, frame = cap.read()
            mymap = fct.warp_image(warp, frame, matrix, (widthImg, heightImg))

            # Calculate dt
            cTime = time.time()
            dt = cTime - pTime
            pTime = cTime

            if pause:
                k = cv2.waitKey(1)
                stop_motors()

                # Used to tune the PID and debug
                robot, PID = fct.debug_mode(debug_mode, robot, PID)

                if k == ord("p"):
                    pause = False
                    start_motors()
                    dt = 0
                continue

            # Kalman's prediction step and store the print
            thymio_state = Kalman.predict(dt)
            my_print.append(thymio_state[0:2])

            # Motion control
            robot.x = thymio_state[0]
            robot.y = thymio_state[1]
            robot.angle = thymio_state[2]

            # PID controller
            PID, robot, speedL, speedR, target_point = navigation(robot, shortest_path, PID)
            set_motors(speedL, speedR)

            # Get the postition
            thymio_visible, measure_cam = tracker.state(mymap)

            # Get speed
            speed = get_motors()
            measure_rob = np.matrix([[fct.speed_pixel_s(speed[0])],
                                     [fct.rot_speed_deg_s(speed[1])]])

            measures = np.append(measure_cam, measure_rob, axis=0)

            # Kalman update step
            if ghost:
                thymio_visible = False
            Kalman.update(thymio_visible, thymio_connected, measures)

            # Drawings
            show = mymap.copy()

            # Show FPS
            show = fct.draw_fps(show, dt)

            # Show path
            show = fct.draw_path(show, shortest_path, (0, 0, 255))
            show = fct.draw_cam_statu(show, thymio_visible)

            # Show my print
            show = fct.draw_my_print(show, my_print, (0, 255, 255))

            # Show thymio position measure if visible
            if thymio_visible:
                show = fct.draw_point(show, measure_cam, (0, 255, 0))

            # Show thymio position and angle Filtered
            show = fct.draw_thymio(show, thymio_state, (255, 0, 0))

            # Show Goal
            show = fct.draw_goal(show, goal_pos, (0, 255, 255))

            # Show target point
            show = fct.draw_point(show, [target_point.x, target_point.y], (255, 0, 0))

            cv2.imshow("Area", show)
            out.write(show)

            k = cv2.waitKey(1)  # Press G to run without the cam
            if k == ord("g"):
                if ghost:
                    ghost = False
                else:
                    ghost = True

            if k == ord("p"):  # Press P to pause the run
                pause = True

            if k == ord("q") or robot.state == STOP:  # Press q to quit the run
                set_motors(0, 0)
                break
                
        cap.release()
        out.release()
        stop_motors()

