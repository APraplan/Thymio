import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

class Point:
    """ a point is a data structure to represent a position on the map,
    felt better than using tuples (x,y) the whole time """
    def __init__(self,x, y):
        self.x = x
        self.y = y

    def getCoordinates(self, trueCoords = False, shape = None):
        if trueCoords and shape != None:
            x = self.y
            y = shape[0]-self.x
        else:
            x = self.x
            y = self.y
        return (x,y)


