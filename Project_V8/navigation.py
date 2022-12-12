
############# imports #############
import numpy as np
import math
import tdmclient.notebook
from tdmclient import ClientAsync, aw
import cv2


############# Global States #############
GLOBAL = 0
STOP = 1


############# Constants #############
nomSpeed = 350
goalThreshold = 30
weightController = 0.6


# Point tracker PID
Kp = 5.0
Ki = 0.0
Kd = 1.0
maxSumError = 50


############# Classes #############
class Point(object):
    def __init__(self, x, y):
        self.x = x 
        self.y = y

              
class Robot(object):
    def __init__(self, x, y, angle, state):
        self.x = x
        self.y = y
        self.angle = angle
        self.state = state
        self.pos = Point(self.x, self.y)
        self.pathCount = 0
    
    def actualisePos(self):
        self.pos = Point(self.x, self.y)


class PID(object):
    def __init__(self, Kp, Ki, Kd, sumError, subError, errorPrev, maxSumError):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.sumError = sumError
        self.subError = subError
        self.errorPrev = errorPrev
        self.maxSumError = maxSumError

        
############# Funnctions #############
def get_projection(start, goal, robot, weight):
    # part one: get the projection of the current location on the line
    # https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter05/section05.html
    xp2 = start.x
    yp2 = start.y
    xp1 = goal.x
    yp1 = goal.y
    xp_ = robot.x
    yp_ = robot.y
    
    #print("robot.x " + str(robot.x) + " robot.y " + str(robot.y))
      
    # P_-P1: v0
    # P2-P1: v1
    
    xv0 = xp_ - xp1
    yv0 = yp_ - yp1
    xv1 = xp2 - xp1
    yv1 = yp2 - yp1
    
    #print("xv0 " + str(xv0) + "yv0 " + str(yv0))

    # Proj = P1 + (P_-P1)(P2-P1)/(P2-P1)(P2-P1) (P2-P1)
    # Proj = P1 + (v0*v1)/(v1*v1) (v1)
    # Projx = xp1 + (xv0*xv1+yv0*yv1)/(xv1*2+yv1*2) * xv1
    # Projy = yp1 + (xv0*xv1+yv0*yv1)/(xv1*2+yv1*2) * yv1

    # Projx = xp1 + (xv0*xv1+yv0*yv1)/(xv1*2+yv1*2) * xv1
    # Projy = yp1 + (xv0*xv1+yv0*yv1)/(xv1*2+yv1*2) * yv1
    if not (xv1 == 0 and yv1 == 0):
        Projx = xp1 + (xv0 * xv1 + yv0 * yv1) / (xv1 ** 2 + yv1 ** 2) * xv1
        Projy = yp1 + (xv0 * xv1 + yv0 * yv1) / (xv1 ** 2 + yv1 ** 2) * yv1
        Proj = Point(Projx, Projy)
    else:
        print(" Caution! singularity in get_projection, line is undefined ")
        Projx = xp1 
        Projy = yp1
        Proj = Point(Projx, Projy)
    
    # part two: weighted avg of the current projection and goal
    Avgx = (1 - weight) * xp1 + weight * Projx
    Avgy = (1 - weight) * yp1 + weight * Projy
    Avg = Point(Avgx, Avgy)
    return Avg, Proj


# angle between the the orientation od the robot and the line formed by the robot to the targer point 
def compute_error(robot, target_point):
    
    angle_line = math.degrees(math.atan2(robot.y - target_point.y, robot.x - target_point.x)) + 180.0
    error = robot.angle - angle_line
    
    return error


# compute PID controller to orientate the robot in the direction of the target point
def pid_target_point(robot, PID, target_point):

    error = compute_error(robot, target_point)

    #Make the robot turn the correct side
    if error > 180:
        error = error - 360
    if error < -180:
        error = error + 360

    PID.sumError += error
    PID.subError = error - PID.errorPrev

    # Anti wind-up
    if PID.sumError > PID.maxSumError:
        PID.sumError = PID.maxSumError
    if PID.sumError < -PID.maxSumError:
        PID.sumError = -PID.maxSumError

    output = PID.Kp * error + PID.Ki * PID.sumError + PID.Kd * PID.subError

    PID.errorPrev = error
    
    return output, PID


def navigation(robot, path, PID):
        
    robot.actualisePos()
    
    # Definition of the line to follow
    prevPoint = path[robot.pathCount]
    nextPoint = path[robot.pathCount+1]

    # Robot in the neighberhood of goalPoint
    if  (nextPoint.x - goalThreshold) < robot.x < (nextPoint.x + goalThreshold) and (nextPoint.y - goalThreshold) < robot.y < (nextPoint.y + goalThreshold):
        robot.pathCount += 1
        if robot.pathCount == len(path) - 1:
            robot.state = STOP
            robot.pathCount = len(path) - 2
            
        # Definition of the line to follow
        prevPoint = path[robot.pathCount]
        nextPoint = path[robot.pathCount+1]
            
    # PID        
    target_point, proj_point = get_projection(prevPoint, nextPoint, robot, weightController)
    [outPID, PID]  = pid_target_point(robot, PID, target_point)

    speedL = nomSpeed - outPID
    speedR = nomSpeed + outPID

    # Converting motor speed type
    speedL = int(speedL)
    speedR = int(speedR)

    return  PID, robot, speedL, speedR, target_point
