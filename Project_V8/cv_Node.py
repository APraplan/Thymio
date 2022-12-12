# each intersection is a node
# each node has 4 neighbours or borders
import cv_Edges as Edges
class node:
    """ to turn the intersections into a graph with neighbours, we create a node for each node
    and use the sortedIntersections functions to determine which nodes (intersections) are our
    neighbours"""

    def __init__(self, intersection, sortedIntersections):
        self.intersection = intersection
        lineh = intersection.lineh
        linel = intersection.linel
        # keep in mind: get_right_intersection can return None
        # the neighbours are organised clockwise
        self.intersection_of_neighbour1 = sortedIntersections.get_right_intersection(lineh, self.intersection)
        self.intersection_of_neighbour2 = sortedIntersections.get_right_intersection(linel, self.intersection)
        self.intersection_of_neighbour3 = sortedIntersections.get_left_intersection(lineh, self.intersection)
        self.intersection_of_neighbour4 = sortedIntersections.get_left_intersection(linel, self.intersection)
        self.neighbour1 = None #None -> border
        self.neighbour2 = None #these point to the neighbouring nodes, not to the intersections
        self.neighbour3 = None
        self.neighbour4 = None
        self.edge1 = None #edge connecting two nodes
        self.edge2 = None
        self.edge3 = None
        self.edge4 = None
########################################################################################################################
    # to add a neighbour:
########################################################################################################################

    def addneighbour1(self, node):
        """
        add node as neighbour1 to self
        """
        self.neighbour1 = node
        myCoordinate = self.intersection.coordinate
        neighbourCoordinate = node.intersection.coordinate
        self.edge1 = Edges.Edge(myCoordinate, neighbourCoordinate)
        #print(str(node) + ' connected to ' + str(self))

    def addneighbour2(self, node):
        """
        add node as neighbour2 to self
        """
        self.neighbour2 = node
        myCoordinate = self.intersection.coordinate
        neighbourCoordinate = node.intersection.coordinate
        self.edge2 = Edges.Edge(myCoordinate, neighbourCoordinate)
        #print(str(node) + ' connected to ' + str(self))

    def addneighbour3(self, node):
        """
        add node as neighbour3 to self
        """
        self.neighbour3 = node
        myCoordinate = self.intersection.coordinate
        neighbourCoordinate = node.intersection.coordinate
        self.edge3 = Edges.Edge(myCoordinate, neighbourCoordinate)
        #print(str(node) + ' connected to ' + str(self))

    def addneighbour4(self, node):
        """
        add node as neighbour4 to self
        """
        self.neighbour4 = node
        myCoordinate = self.intersection.coordinate
        neighbourCoordinate = node.intersection.coordinate
        self.edge4 = Edges.Edge(myCoordinate, neighbourCoordinate)
        #print(str(node) + ' connected to ' + str(self))

    ########################################################################################################################
    # to navigate a loop, we return the next neighbour following a clockwise pattern
    ########################################################################################################################

    def clockwiseNext(self, prev):
        """
        return the next neighbour and the edge connecting next and self,
        coming from prev,
        following a clockwise pattern
        """
        cond = False
        if cond:
            print(' ')
            print('============================')
            print('making clockwise loop, prev:')
            print(prev)
            print('neighbours: ')
            print(self.neighbour1)
            print(self.neighbour2)
            print(self.neighbour3)
            print(self.neighbour4)
            print(' ')
            print('analysis of neighbour 1')
            print('self')
            print(self)
            #print(self.neighbour1)
            if self.neighbour1 != None:
                print(self.neighbour1.neighbour1)
                print(self.neighbour1.neighbour2)
                print(self.neighbour1.neighbour3)
                print(self.neighbour1.neighbour4)
            print(' ')
            print('analysis of neighbour 2')
            print('self')
            print(self)
            if self.neighbour2 != None:
                print(self.neighbour2.neighbour1)
                print(self.neighbour2.neighbour2)
                print(self.neighbour2.neighbour3)
                print(self.neighbour2.neighbour4)
            print(' ')
            print('analysis of neighbour 3')
            print('self')
            print(self)
            if self.neighbour3 != None:
                print(self.neighbour3.neighbour1)
                print(self.neighbour3.neighbour2)
                print(self.neighbour3.neighbour3)
                print(self.neighbour3.neighbour4)
            print(' ')
            print('analysis of neighbour 4')
            print('self')
            print(self)
            if self.neighbour4 != None:
                print(self.neighbour4.neighbour1)
                print(self.neighbour4.neighbour2)
                print(self.neighbour4.neighbour3)
                print(self.neighbour4.neighbour4)
        assert  prev != None
        if prev == self.neighbour1:
            next = self.neighbour4
            edge = self.edge4
        elif prev == self.neighbour2:
            next = self.neighbour1
            edge = self.edge1
        elif prev == self.neighbour3:
            next = self.neighbour2
            edge = self.edge2
        elif prev == self.neighbour4:
            next = self.neighbour3
            edge = self.edge3
        if next != None:
            pass
            #print('go from '+str((self.intersection.coordinate.x, self.intersection.coordinate.y)) + ' to ' + str((next.intersection.coordinate.x, next.intersection.coordinate.y)))
        else:
            pass
            #print('we dont have a neighbour here')
        return next, edge

