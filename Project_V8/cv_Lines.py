import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
class Line3:
    " a fter calculating the hough transform we find r & theta of lines, we turn these into m&b as in y = mx+b"
    " if a line is horizontal, its m is infinite, so this is a special case"
    def __init__(self, r, theta, identity):
        self.identity = str(chr(65+identity))
        self.r = r
        if theta >= 0:
            self.theta = theta
        elif theta < 0:
            self.theta = theta + 2*math.pi
        self.type = 'regular'
        #print('--analysing new line with r & theta')
        #print('---'+str(r)+ ' '+ str(theta))
        if r == 0 and theta == math.pi/2:
            self.type = 'vertical'
            self.m = 0
            self.b = 0

        elif (r == 0 and theta == 0) or (r ==  0 and theta == math.pi): # to prevent singularities
            self.type = 'horizontal'
            self.m = None
            self.b = None
            self.x = r

        elif r == 0 and theta >= 2*math.pi:
            self.type = 'horizontal'
            self.theta = 0
            self.m = None
            self.b = None
            self.x = r

        elif r == 0:
            self.m = -1/math.tan(theta)
            self.b = 0

        elif 0 < theta < math.pi/2:
            ymax = r/math.cos(math.pi/2-theta)
            xmax = r/math.cos(theta)
            self.b = ymax
            self.m = -ymax/xmax

        elif math.pi > theta > math.pi/2:
            y_max = r/math.cos(math.pi/2 - theta)
            x_max = r/math.cos(math.pi-theta)
            self.b = y_max
            self.m = y_max/x_max

        elif math.pi < theta < math.pi*3/2:
            self.b = None
            self.m = None
            print('LINE OUT OF BOUNDS')
            self.type = 'out of bounds'

        elif 2*math.pi > theta > math.pi*3/2:
            x0 = r/math.cos(2*math.pi - theta)
            y0 = r/math.cos(theta - 3*math.pi/2)
            self.b = -y0
            self.m = y0/x0

        elif theta == 0:
            self.m = None
            self.b = None
            self.x = r
            self.type = 'horizontal'

        elif theta == math.pi/2:
            self.m = 0
            self.b = r
            self.type = 'vertical'

        elif theta == math.pi:
            self.m = None
            self.b = None
            self.x = -r
            self.type = 'horizontal'

        elif theta == math.pi*3/2:
            self.m = 0
            self.b = -r # ps this is out of the image
            self.type = 'vertical'

        elif theta >= math.pi*2:
            self.theta = 0
            self.m = None
            self.b = None
            self.x = r
            self.type = 'horizontal'

        else:
            print('lines, weird theta')
            print(r, theta)
            self.type = 'else'
