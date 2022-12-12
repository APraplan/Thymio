import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from cv_SortedIntersections import SortedIntersections
import cv_Obstacles as Obstacles
import cv_Pictures as Pictures
import cv_settings as settings
import cv_Node as Node
import cv_NodeSet as NodeSet
import cv_LoopSet as LoopSet
import cv_Obstacles as Obstacles


class Map:
    def __init__(self, filePath, starttime, picture):
        """
        filePath: the location where picture is stored
        picture: np.array with grayscale data
        """
        file = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        self.filePath = filePath
        self.time_of_origin = starttime
        self.data = file
        self.max_x, self.max_y = file.shape
        self.picture = picture
        self.intersections = picture.intersections

    ########################################################################################################################
    # INTERSECTIONS -> SORTED INTERSECTIONS
    ########################################################################################################################
    def sortIntersections(self):
            """
            use self.intersections to make a 'sortedintersections' obstacle
            ~ dictionary but with some extra functions to make a graph
            """
            # the intersections are currently sorted per line,
            # we put them in one list to make the dictionary
            self.intersectionList = []
            for intersections_of_line_i in self.intersections:
                for intersection in intersections_of_line_i:
                    self.intersectionList.append(intersection)
            for intersection in self.intersectionList:
                assert intersection.identity == (intersection.line1.identity,intersection.line2.identity)
            print('there are '+ str(len(self.intersectionList)) + ' intersections')
            self.sortedIntersections = SortedIntersections(self.intersectionList)

    ########################################################################################################################
    # SORTED INTERSECTIONS -> NODES
    ########################################################################################################################
    def makeGraph(self):
            """
            make a node for each intersection to make a graph later
            the difference between a node and an intersection is that a node has neighbouring nodes,
            while an intersection only knows which lines define it
            """
            self.nodeSet = NodeSet.nodeSet()
            for intersection in self.intersectionList:
                new_node = Node.node(intersection, self.sortedIntersections)
                self.nodeSet.addNode(new_node)
########################################################################################################################
            # CHECKING NEIGHBOURS
########################################################################################################################
            for node in self.nodeSet.nodes:
                if node.neighbour1 != None:
                    neighbour = node.neighbour1
                    neighbours_neighbours = [neighbour.neighbour1, neighbour.neighbour2, neighbour.neighbour3, neighbour.neighbour4]
                    assert node in neighbours_neighbours
                if node.neighbour2 != None:
                    neighbour = node.neighbour2
                    neighbours_neighbours = [neighbour.neighbour1, neighbour.neighbour2, neighbour.neighbour3, neighbour.neighbour4]
                    assert node in neighbours_neighbours
                if node.neighbour3 != None:
                    neighbour = node.neighbour3
                    neighbours_neighbours = [neighbour.neighbour1, neighbour.neighbour2, neighbour.neighbour3, neighbour.neighbour4]
                    assert node in neighbours_neighbours
                if node.neighbour4 != None:
                    neighbour = node.neighbour4
                    neighbours_neighbours = [neighbour.neighbour1, neighbour.neighbour2, neighbour.neighbour3, neighbour.neighbour4]
                    assert node in neighbours_neighbours


########################################################################################################################
            # NODES -> LOOPS
########################################################################################################################
    def makeLoops(self):
        """
        make loops using the nodes, save them in self.loopSet
        """
        # now we should have a complete graph with neigbouring nodes
        # now start making the loops
        self.loopSet = LoopSet.LoopSet()
        for node in self.nodeSet.nodes:
            self.loopSet.addClockwiseLoop(node)
                
        # remove all duplicate loops
        self.loopSet.remove_duplicates()
        print('there are '+str(len(self.loopSet.loopSet))+' distinct loops')


########################################################################################################################
            # VISUALISE LOOPS
########################################################################################################################
    def visualiseLoops(self):
            """
            visualise theresult so far
            """
            def addloops(ax, plt, loops):
                for loop in loops:
                    for edge in loop.edges:
                        px = []
                        py = []
                        for point in edge.points:
                            px.append(point.x)
                            py.append(point.y)
                        px.append(px[0]) #close the loop
                        py.append(py[0])
                        ax.plot(py, px, linewidth=2, color="r")

            def saveloops(loopSet):
                plt.close()
                fig, ax = plt.subplots()
                ax.imshow(self.data, cmap='gray')
                ax.autoscale(False)
                addloops(ax, plt, loopSet.loopSet)
                plt.title("Obstacle with loops")
                filePath = self.filePath[:len(self.filePath) - 4] + 'withloops.png'
                plt.xlabel('y [pixels]')
                plt.ylabel('x [pixels]')
                plt.savefig(filePath)
                if settings.SHOWSEPERATECLOSEDLOOPS:
                    print('SHOWING SEPERATE CLOSED LOOPS')
                    plt.show()
                plt.close()
            saveloops(self.loopSet)

    def getObstaclesFromGraph(self):
            """
            check if the centre of each loop is white or black
            white loops are discarded
            black loops are considered obstacles
            an obstacle might be split into two, so they obstacles are fused if they share a neighbouring edge
            """
########################################################################################################################
            # LOOPS -> OBSTACLES
########################################################################################################################
            # now we should have have a set with all the loops
            # we turn these into obstacles to check their colors
            ObstacleSet = set()
            for loop in self.loopSet.loopSet: #loop should be a set of corner points
                new_obstacle = Obstacles.Obstacle(loop.corners, loop.edges)
                # if the section is colored, we keep it, if it is white, we discard
                if new_obstacle.get_color(self.picture) <= settings.MAPBLACKTRESHOLD:
                    ObstacleSet.add(new_obstacle)

########################################################################################################################
            # FUSE OBSTACLES
########################################################################################################################
            # we now have a set of colored obstacles, we will try to fuse these if they share one or more edges
            old_fusedObstacleSet = set()
            new_fusedObstacleSet = ObstacleSet
            while old_fusedObstacleSet != new_fusedObstacleSet: #keep repeating (fusing) until it converges
                old_fusedObstacleSet = new_fusedObstacleSet
                new_fusedObstacleSet = set()
                for obstacle in old_fusedObstacleSet:
                    if len(new_fusedObstacleSet) == 0:
                        new_fusedObstacleSet.add(obstacle)
                    else:
                        wasfusable = False
                        for fusedObstacle in new_fusedObstacleSet:
                            if fusedObstacle.checkFusable(obstacle):
                                new_obstacle = fusedObstacle.fuseObstacles(obstacle)
                                new_fusedObstacleSet.remove(fusedObstacle)
                                new_fusedObstacleSet.add(new_obstacle)
                                wasfusable = True
                                break
                        if wasfusable == False: # the two obstacles didnt match so can be added to the list
                            new_fusedObstacleSet.add(obstacle)
            self.obstacles = new_fusedObstacleSet

    def visualiseObstacles(self):
            """
            visualise results
            """
            def addobstacles(ax, obstacles):
                print('there are ' +str(len(obstacles))+ ' obstacles')
                for obstacle in obstacles:
                    for edge in obstacle.edges:
                        px = []
                        py = []
                        for point in edge.points:
                            px.append(point.x)
                            py.append(point.y)
                        px.append(px[0])  # close the loop
                        py.append(py[0])
                        ax.plot(py, px, linewidth=2, color="r")

            #plt.imshow(self.data, cmap='gray') # why are you showing me 3 pics??????
            plt.close()
            fig, ax = plt.subplots()
            ax.imshow(self.data, cmap='gray')
            ax.autoscale(False)
            addobstacles(ax, self.get_all_obstacles())
            filePath = self.filePath[:len(self.filePath) - 4] + 'result.png'
            plt.title("result of obstacle analysis")
            plt.xlabel('y [pixels]')
            plt.ylabel('x [pixels]')
            plt.savefig(filePath)
            if settings.SHOWRESULT:
                plt.show()
            plt.close()

            # we now have a set with fused obstacles
            # this is what we wanted

########################################################################################################################
        # FINISH
########################################################################################################################
    def get_all_obstacles(self):
        return self.obstacles

    def showObstacles3(self):
        def addobstacles(ax, obstacles):
            for obstacle in obstacles:
                for edge in obstacle.edges:
                    px = []
                    py = []
                    for point in edge.points:
                        px.append(point.x)
                        py.append(point.y)
                    px.append(px[0])  # close the loop
                    py.append(py[0])
                    ax.plot(py, px, linewidth=2, color="r")
        plt.close()
        plt.imshow(self.data, cmap='gray')
        fig, ax = plt.subplots()
        ax.imshow(self.data, cmap='gray')
        ax.autoscale(False)
        addobstacles(ax, self.get_all_obstacles())
        filePath = self.filePath[:len(self.filePath) - 4] + 'result.png'
        plt.xlabel('y [pixels]')
        plt.ylabel('x [pixels]')
        plt.savefig(filePath)
        plt.show()
        plt.close()
