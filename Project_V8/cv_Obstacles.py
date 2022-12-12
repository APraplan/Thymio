import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import cv_Points as Points
class Obstacle:
    """
    an obstacle is really similar to a loop, but a loop only becomes an obstacle if it is unique & colored,
    neighbouring (sharing an edge) obstacles will be fused to form one big obstacle
    """

    def __init__(self, corners, edges):
        self.corners = corners
        self.edges = edges
        x = 0
        y = 0
        #for corner in corners:
        #    x += corner.x
        #    y += corner.y
        #self.centre = Points.Point(int(x/len(corners)), int(y/len(corners)))    #average of the corner coordinates
        for edge in edges:
            for point in edge.points:
                x += point.x/2
                y += point.y/2
        self.centre = Points.Point(int(x/len(edges)), int(y/len(edges))) #average of the corner coordinates
    ########################################################################################################################
    # to determine if this obstacle is coloured or not (and should be discarded)
    ########################################################################################################################

    def get_color(self, picture):
        return picture.get_color(self.centre)

    ########################################################################################################################
    # to determine if this obstacle is coloured or not (and should be discarded)
    ########################################################################################################################

    def checkFusable(self, obstacle):
        counter = 0 #two objects are fusable if they share an edge
        for edge in obstacle.edges:
            for edge_ in self.edges:
                if edge.points == edge_.points:
                    counter += 1
        return counter >= 1

    ########################################################################################################################
    # fuse parts of an obstacle into one
    ########################################################################################################################

    def fuseObstacles(self, obstacle):
        """
        fuse self and an other obstacle if they share one or more edges,
        to do this,
        common edges are discarded (as they will now be inside the obstacle & no longer are an edge)
        unique edges of both obstacles are preserved & appended to a new list
        """
        printfuse = False
        if printfuse:
            print('fusing to obstacles with edges:')
            print('obstacle1:')
            for edge in self.edges:
                print(edge.view())
            print('obstacle2:')
            for edge in obstacle.edges:
                print(edge.view())

        # two objects are fused if they share an (multiple) edge
        # when we fuse two obstacles, their common edges dissapear
        # other edges are added
        fusedEdges = set()
        for edge in self.edges:
            fusedEdges.add(edge)
        removeLater = []
        addLater = []
        for newEdge in obstacle.edges:
            #an edge is defined by two points, if they are the same, this is the same edge
            hasmatched = 0
            for oldEdge in fusedEdges:
                if oldEdge.points == newEdge.points: # this is a common edge and hence gets cancelled
                    hasmatched = 1
                    removeLater.append(oldEdge)
            if hasmatched == 0:
                addLater.append(newEdge)

        for edge in removeLater:
            fusedEdges.remove(edge)
        for edge in addLater:
            fusedEdges.add(edge)
        if fusedEdges == set():
            print('a duplicate loop slipped through the cracks, this shouldnt happen')
            print("self edgess:")
            for edge in self.edges:
                print(edge.points)
            print('other edges:')
            for edge in obstacle.edges:
                print(edge.points)
            assert fusedEdges != set()
        #fusedEdges now contains the fused edges of two objects, now we recreate points:
        newPoints = set()
        for edge in fusedEdges:
            for point in edge.points:
                newPoints.add(point)
        newObstacle = Obstacle(newPoints, fusedEdges)
        return newObstacle



