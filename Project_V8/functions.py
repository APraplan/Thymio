"""
 Library of function and classes used for the mini project
 Axel Praplan
 Lausanne 09.12.2022
 Some functions are inspired by L42project and wikipedia: Extended Kalman filter
"""
import cv2
import math
import numpy as np

# Constant
widthImg  = 1080
heightImg = 720
warp = True


"""
 Initialization routine to set a new environment up
 Input:
   name1: name of the file to save data of the size of the map 
   name2: name of the data to save the colors to track the thymio
   widthImg: width of the image
   heightImg: height of the image 
   warp: boolean, reshaping of the image or not
 Output:
   none: two text files and an image are created
"""
def initialization_data(name1, name2):
    global MOUSEX, MOUSEY, CLICK, UP, DOWN
    CLICK = UP = DOWN = False
    points = []
    colors = []

    init = input("Do you want to initialize a new setup ? (y,n)")

    if init == "y":

        cap = cv2.VideoCapture(0)
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        # Set width and height parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, widthImg)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, heightImg)

        init_map = input("Do you want to initialize a new shape for the map ? (y,n)")

        if init_map == "y":

            # Select map corners
            corner_number = 0
            text = ["Select the top left corner of the map", "Select the top right corner of the map",
                    "Select the bottom left corner of the map", "Select the bottom right corner of the map"]

            while True:

                _, frame = cap.read()
                frame = cv2.putText(frame, text[corner_number], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                    cv2.LINE_AA)
                cv2.imshow("Area", frame)
                cv2.setMouseCallback("Area", take_coordinates)

                if CLICK:
                    point = [MOUSEX, MOUSEY]
                    points.append(point)
                    corner_number = corner_number + 1
                    CLICK = False

                if corner_number == 4:
                    np.savetxt(name1, points)
                    break

                cv2.waitKey(1)

        cv2.destroyAllWindows()
        init_goal = input("Do you want to initialize a new template for the goal ? (y,n)")

        if init_goal == "y":

            _, frame = cap.read()

            matrix = warp_matrix(widthImg, heightImg, name1)

            if warp:
                mymap = cv2.warpPerspective(frame, matrix, (widthImg, heightImg))
            else:
                mymap = frame

            # Select Goal ROI
            print("Ready to select Goal template")
            area = cv2.selectROI('Area', mymap)
            template = mymap[area[1]:area[1] + area[3], area[0]:area[0] + area[2]]
            cv2.imwrite('GoalTemplate.png', template)

        cv2.destroyAllWindows()
        init_colors = input("Do you want to select new colors to track Thymio ? (y,n)")

        if init_colors == "y":

            # Select Front and back colors
            color_number = 0
            color = 0
            text = ["Select the front color", "Select the back color"]

            while True:

                _, frame = cap.read()
                
                matrix = warp_matrix(widthImg, heightImg, name1)

                if warp:
                    mymap = cv2.warpPerspective(frame, matrix, (widthImg, heightImg))
                else:
                    mymap = frame

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lo = np.array([color - 10, 100, 50])
                hi = np.array([color + 10, 255, 255])
                mask = cv2.inRange(img, lo, hi)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                cv2.imshow("mask", mask)

                mymap = cv2.putText(mymap, text[color_number], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                    cv2.LINE_AA)
                cv2.imshow("Area", mymap)

                cv2.setMouseCallback("Area", take_coordinates)

                if CLICK:
                    img = cv2.cvtColor(mymap, cv2.COLOR_BGR2HSV)
                    color = img[MOUSEY, MOUSEX][0]
                    CLICK = False

                if UP and color < 245:
                    color = color + 1
                    UP = False
                elif DOWN and color > 10:
                    color = color - 1
                    DOWN = False

                if 13 == cv2.waitKey(1):
                    colors.append(color)
                    color_number = color_number + 1

                if color_number == 2:
                    np.savetxt(name2, colors)
                    break

        cap.release()
        cv2.destroyAllWindows()
        

"""
Class extended Kalman filter with 3 states, only update with measure on the robot, update based on the camera detection or update based on camera and measures on the robot (Inspired by L42project and wiki kalman extended)
__init__:
   Input:
     dt: time interval between calls
     state: initial state of the object
   Output:
     none: initialization self parameter
 Predict:
   Input:
     dt: delta time since last call
   Output:
     self.E: state vector 1x5 predicted with pos X, pos Y, orientation, speed, rotational speed 
 Update:
   Input:
     visible: boolean, if the object is in the ligne of sight, pos X, pos Y, orientation are measured
     connected: boolean, if the object send data, speed, rotational speed are measured
     Z: measure of the state
   Output:
      self.E: state vector 1x5 updated with pos X, pos Y, orientation, speed, rotational speed 
"""
class KalmanFilter(object):
    def __init__(self, dt, state):
        self.dt = dt

        # Vecteur d'etat initial
        self.E = state

        # Jacobian
        a13 = float(-math.sin(math.radians(self.E[2])) * self.E[3] * self.dt)
        a14 = float(math.cos(math.radians(self.E[2])) * self.dt)
        a23 = float(math.cos(math.radians(self.E[2])) * self.E[3] * self.dt)
        a24 = float(math.sin(math.radians(self.E[2])) * self.dt)
        a35 = float(self.dt)

        self.F = np.matrix([[1.0, 0.0, a13, a14, 0.0],
                            [0.0, 1.0, a23, a24, 0.0],
                            [0.0, 0.0, 1.0, 0.0, a35],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])

        # Prediction error
        self.Q = np.matrix([[6.5, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 6.5, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.05, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.05]])

        # Measure error
        self.R = np.matrix([[6.5, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 6.5, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.05, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.05]])

        self.H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])

        self.P = np.eye(self.F.shape[1])

    def predict(self, dt):
        
        self.dt = dt

        # Jacobian
        a13 = float(-math.sin(math.radians(self.E[2])) * self.E[3] * self.dt)
        a14 = float(math.cos(math.radians(self.E[2])) * self.dt)
        a23 = float(math.cos(math.radians(self.E[2])) * self.E[3] * self.dt)
        a24 = float(math.sin(math.radians(self.E[2])) * self.dt)
        a35 = float(self.dt)

        self.F = np.matrix([[1.0, 0.0, a13, a14, 0.0],
                            [0.0, 1.0, a23, a24, 0.0],
                            [0.0, 0.0, 1.0, 0.0, a35],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])

        # Prediction step
        e1 = float(self.E[0] + math.cos(math.radians(self.E[2])) * self.E[3] * self.dt)
        e2 = float(self.E[1] + math.sin(math.radians(self.E[2])) * self.E[3] * self.dt)
        e3 = math.remainder(float(self.E[2] + self.E[4] * self.dt), 360)
        if e3 < 0:
            e3 = e3 + 360.0
        e4 = float(self.E[3])
        e5 = float(self.E[4])
        
        self.E = np.matrix([[e1],
                            [e2],
                            [e3],
                            [e4],
                            [e5]])

        # compute vovariance of error
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.E
    
    def update(self, visible, connected, Z):

        if visible and connected:  # Cam and Rob info

            self.H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0]])

        elif connected:  # Only rob info

            self.H = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0]])

        elif visible:  # Only cam info

            self.H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0]])
        else:
            return self.E

        # Compute Kalman gain 
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        try:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        except:
            K = [[0.0],
                 [0.0],
                 [0.0],
                 [0.0],
                 [0.0]]

        # Correction / innovation / normalization angle
        dE = Z - np.dot(self.H, self.E)
        dE[2] = math.remainder(dE[2], 360)
        self.E = self.E + np.dot(K, dE)
        self.E[2] = math.remainder(self.E[2], 360)
        if self.E[2] < 0:
            self.E[2] = self.E[2] + 360

        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P

        return self.E
    

"""
 Class detection Thymio to detect the two distinct point of colors on the map
 detection_front:
   Input:
     frame: Image to search on     
   Output:
     boolean: found or not
     pos X: position X
     pos Y: position Y 
 detection_back:
   Input:
     frame: Image to search on     
   Output:
     boolean: found or not
     pos X: position X
     pos Y: position Y 
 State:
   Input:
     frame: Image to search on     
   Output:   
     boolean: found or not
     Z: state vector 1x3 with pos X, pos Y, angle
"""
class DetectionThymio(object):
    def __init__(self, name):
        self.colors = np.loadtxt(name)

    def detection_front(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([self.colors[0] - 10, 100, 50])
        hi = np.array([self.colors[0] + 10, 255, 255])
        mask = cv2.inRange(img, lo, hi)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        #cv2.imshow("mask forward", mask)
        elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(elements) > 0:
            zone = max(elements, key=cv2.contourArea)
            area = cv2.contourArea(zone)
            ((x, y), radius) = cv2.minEnclosingCircle(zone)
            if area > 350:
                return True, x, y
        return False, 0, 0

    def detection_back(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([self.colors[1] - 10, 100, 50])
        hi = np.array([self.colors[1] + 10, 255, 255])
        mask = cv2.inRange(img, lo, hi)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        #cv2.imshow("mask back", mask)
        elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(elements) > 0:
            zone = max(elements, key=cv2.contourArea)
            area = cv2.contourArea(zone)
            ((x, y), radius) = cv2.minEnclosingCircle(zone)
            if area > 350:
                return True, x, y
        return False, 0, 0

    def state(self, frame):

        Z = np.matrix([[0.0],
                       [0.0],
                       [0.0]])

        front_found, xf, yf = self.detection_front(frame)
        back_found, xb, yb = self.detection_back(frame)

        if front_found and back_found:
            Z[0] = 2 / 7 * xf + 5 / 7 * xb
            Z[1] = 2 / 7 * yf + 5 / 7 * yb
            Z[2] = math.degrees(math.atan2(yb - yf, xb - xf))# + 180.0
            return True, Z
        else:
            return False, Z

        
"""
 Initialization of the warp matrix to reshape the map
 Input:
   widthImg: width of the image
   heightImg: height of the image
   name: name of the folder containing the data of the shape
 Output:
   matrix: warp matrix to reshape the map
"""
def warp_matrix(widthImg, heightImg, name):
    pts1 = np.loadtxt(name)
    pts1 = np.float32(pts1)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # prepare point for warp
    return cv2.getPerspectiveTransform(pts1, pts2)


"""
 Interruption routine for mouse event
 Input:
   event: type of mouse event
   x: position of the pointer X
   y: position of the pointer Y
   flags: flags of event (Wheel up, Wheel down)
   param: optional parameters
 Output:
   none: update of global variables
"""
def take_coordinates(event, x, y, flags, param):
    global MOUSEX, MOUSEY, CLICK, UP, DOWN
    if event == cv2.EVENT_LBUTTONDOWN:
        MOUSEX = x
        MOUSEY = y
        CLICK = True
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags < 0:
            DOWN = True
        else:
            UP = True

            
"""
 function to reshape the image to fill the screen
 Input:
   warp: boolean warp or keep the raw image
   frame: frame to tranform
   matrix: matrix of transforamtion, generated by warp_matrix()
   dim: dimension of the frame
 Output:
   mymap: Image with the new shape
"""
def warp_image(warp, frame, matrix, dim):
    if warp:
        mymap = cv2.warpPerspective(frame, matrix, dim)
    else:
        mymap = frame

    return mymap


"""
 funtion to apply template matching
 Input:
   name: name of the template png file
   mymap: Image to search on
 Output:
   goal_pos: vetctor 1x2 with the position X and Y of the goal 
"""
def find_template(name, mymap):
    goal_template = cv2.imread(name)
    c, w, h = goal_template.shape[::-1]
    res = cv2.matchTemplate(mymap, goal_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left_goal = max_loc
    goal_pos = [top_left_goal[0] + w / 2, top_left_goal[1] + h / 2]

    return goal_pos


"""
 function that draw a point on an image
 Input:
   show: image to draw on
   pos: position of the point 
   color: color for the point(R,G,B)
 Output:
   show: the image with the drawing   
"""
def draw_point(show, pos, color):
    cv2.circle(show, (int(pos[0]), int(pos[1])), 5, color, cv2.FILLED)

    return show


"""
 function that draw the goal with a text on it on an image
 Input:
   show: image to draw on
   goal_pos: position of the goal 
   color: color for the goal (R,G,B)
 Output:
   show: the image with the drawing   
"""
def draw_goal(show, goal_pos, color):
    
    text = "GOAL"
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    text_x = int(goal_pos[0]-textsize[0]/2)
    text_y = int(goal_pos[1]-50)
    
    cv2.circle(show, (int(goal_pos[0]), int(goal_pos[1])), 5, color, cv2.FILLED)
    cv2.putText(show, text, (text_x, text_y), font, 1, color, 2)

    return show


"""
 function that draw the thymio with his orientation
 Input:
   show: image to draw on
   thymio_state: position and orientation of the thymio
   color: color for the thymio (R,G,B)
 Output:
   show: the image with the drawing   
"""
def draw_thymio(show, thymio_state, color):
    
    end_point = (int(thymio_state[0] + 30 * math.cos(math.radians(thymio_state[2]))),
                 int(thymio_state[1] + 30 * math.sin(math.radians(thymio_state[2]))))
    cv2.arrowedLine(show, (int(thymio_state[0]), int(thymio_state[1])), end_point, color, 2)
    cv2.circle(show, (int(thymio_state[0]), int(thymio_state[1])), 5, color, cv2.FILLED)

    return show


"""
 function that draw the fps
 Input:
   show: image to draw on
   dt: delta time between frames
 Output:
   show: the image with the drawing   
"""
def draw_fps(show, dt):
    
    if dt != 0:
        fps = 1 / dt
        cv2.putText(show, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2, cv2.LINE_AA)

    return show


"""
 function that draw the path
 Input:
   show: image to draw on
   shortest_path: list of points of the path
   color: color for the path (R,G,B)
 Output:
   show: the image with the drawing   
"""
def draw_path(show, shortest_path, color):
    for i in range(len(shortest_path) - 1):
        cv2.line(show, (int(shortest_path[i].x), int(shortest_path[i].y)),
                 (int(shortest_path[i+1].x), int(shortest_path[i+1].y)), color, 2)

    return show


"""
 function that draw the path
 Input:
   show: image to draw on
   polygons: list of obstacles on the map
   color: color for the polygons (R,G,B)
 Output:
   show: the image with the drawing   
"""
def draw_obstacles(show, polygons, color):

    for i in range(len(polygons)):
        for j in range(len(polygons[i])):
            if j == len(polygons[i]) - 1:
                k = 0
            else:
                k = j + 1

            cv2.line(show, (int(polygons[i][j].x), int(polygons[i][j].y)),
                     (int(polygons[i][k].x), int(polygons[i][k].y)), color, 2)

    return show


"""
 function that draw a text
 Input:
   show: image to draw on
   text: string to draw
   color: color for the text (R,G,B)
 Output:
   show: the image with the drawing   
"""
def draw_text(show, text, color):

    font = cv2.FONT_HERSHEY_SIMPLEX

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    text_x = int((show.shape[1] - textsize[0]) / 2)
    #text_y = int((show.shape[0] + textsize[1]) / 2)
    text_y = 50

    # add text centered on image
    cv2.putText(show, text, (text_x, text_y), font, 1, color, 2)

    return show


"""
 function that draw the cam statu 
 Input:
   show: image to draw on
   thymio_visible: boolean, thymio in the field of view
 Output:
   show: the image with the drawing   
"""
def draw_cam_statu(show, thymio_visible):

    text = "No Cam"

    if not thymio_visible:

        font = cv2.FONT_HERSHEY_SIMPLEX
        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        text_x = int(show.shape[1] - textsize[0] - 50)
        text_y = 50

        # add text
        cv2.putText(show, text, (text_x, text_y), font, 1, (0, 0, 255), 2)

    return show


"""
 function that draw the prints of the thymio
 Input:
   show: image to draw on
   my_print: list of points last positions of the robot
   color: color for the path (R,G,B)
 Output:
   show: the image with the drawing   
"""
def draw_my_print(show, my_print, color):
    
    try:
        for i in range(len(my_print)):
            cv2.circle(show, (int(my_print[i][0]), int(my_print[i][1])), 2, color, cv2.FILLED)
    except:
        pass
        
    return show    


"""
  funtion that convert the speed of the thymio in pixels/s
  Input:
    speed: speed of the thymio
  Output:
    speed: speed in pixel/s
"""
def speed_pixel_s(speed):

    thymio_speed_to_mms = 0.33478260869565216
    pixel_mm = 1080.0/1420.0

    return speed * thymio_speed_to_mms * pixel_mm


"""
 funtion that convert the rotational speed of the thymio in deg/s
 Input:
   rot_speed: rotational speed of the thymio
 Output:
   rot_speed: rotational speed in deg/s
"""
def rot_speed_deg_s(rot_speed):

    thymio_speed_to_mms = 0.33478260869565216
    rayon = 52.0

    return rot_speed * thymio_speed_to_mms * 180.0 / math.pi / rayon


def debug_mode(debug_mode, robot, PID, k):
    
    if debug_mode == True:
        if k == ord("w"):
            PID.Kp = PID.Kp + 0.01
            clear_output(wait=True)
            print("PID.Kp" + str(PID.Kp))
        if k == ord("e"):
            PID.Ki = PID.Ki + 0.001
            clear_output(wait=True)
            print("PID.Ki" + str(PID.Ki))
        if k == ord("r"):
            PID.Kd = PID.Kd + 0.01
            clear_output(wait=True)
            print("PID.Kd" + str(PID.Kp))

        if k == ord("y"):
            PID.Kp = PID.Kp - 0.001
            clear_output(wait=True)
            print("PID.Kp" + str(PID.Kp))
        if k == ord("x"):
            PID.Ki = PID.Ki - 0.01
            clear_output(wait=True)
            print("PID.Ki" + str(PID.Ki))       
        if k == ord("c"):
            PID.Kd = PID.Kd - 0.001
            clear_output(wait=True)
            print("PID.Kd" + str(PID.Kd)) 
        if k == ord("d"):
            clear_output(wait=True)
            print(robot.state)

        return robot, PID

