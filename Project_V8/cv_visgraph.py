import cv_Points as Points
import matplotlib.pyplot as plt
import numpy as np
import math
class visgraph():
    """
    we tried to make the project using the pyvisgraph package,
    however, this package was giving us problems in two ways:
        - as we enlarge objects after having detected them, the enlarged versions may overlap,
          which means the thymio cannot pass through here. However, the pyvisgraph package ignores this problem
        - sometimes the shortest route may be around the map, rather than slaloming through the objects. We didnt
          manage to add a border object without running into problem 1. Without this border however, the thymio
          tends to drive itself off the table...
    for these two reasons we made our own visgraph package
    important data structures are:
        - self.edges: stored here all the edges (both from objects and map border)
          An edge is defined by two points
          An edge is used to check if and where it intersects with two points trying to 'see' each other
          By adding border edges to this set we solve problem 2
        - self.points: stored here are all the points (from objects, start & end point, but not from the border)
        - self.triangles: stored here are all the objects, split into triangles (from objects)
          by splitting an object in triangles, we can use vector theory to assert whether or not a point lies within this
          object or not. This allows us to solve problem 1
        - self.graph: a dictionary that, given a point, returns all the other points this point can see
          upon calling self.graph[point_a][point_b], if point_a and point_b can see each other,
          the distance between these points is returned
        - self.can_see: a primitive version of the graph, it is later converted
        - self.pointCoords is kept to assert all points are unique
    furthermore, a slight alteration was made to the dijkstra algorithm to work around problem 1
    the code assumes all polygones are convex (so all connections between corners are internal)
    """
    def __init__(self):
        """
        initialise empty data structures
        """
        self.edges = set()
        self.points = set()
        self.can_see = dict()
        #self.distances = dict()
        self.triangles = []
        self.pointCoords = []

    def addEdges(self, poly):
        """
            add all edges of ONE polygon to the visgraph
            not just the outer edges are added, but also every possible internal edge connecting two corners,
            to ensure you cannot see through an object
        """
        for i in range(len(poly)):
            for j in range(i+1, len(poly)):
                pointa = poly[i]
                pointb = poly[j]
                self.edges.add((pointa, pointb))

    def distance(self, pointa, pointb):
        """
            return the distance between two unique (distinct) points,
            expressed in pixels, not m
        """
        assert math.sqrt((pointa.x-pointb.x)**2 + (pointa.y-pointb.y)**2) != 0
        return math.sqrt((pointa.x-pointb.x)**2 + (pointa.y-pointb.y)**2)

    def checkVis(self, pointa, pointb):
        """
            check whether or not two points can see each other,
            if they can, add them to the data structure self.can_see in both directions
            a special case is when a line is vertical: here m (as in y = mx+b) is infinite
            and calculation of m causes division by zero errors

            two points can not see each other if there is at least one edge preventing this
            an edge prevents vision if the line connecting the two points intersects with the edge
            in the coordinate x , y such that:
                x in the x-domain of the two points
                AND
                x in the x-domain of the edge
            in the special vertical case:
                x in the x-domain of the two points
                AND
                y in the y-domain of the edge
        """
        if (pointb.x-pointa.x) != 0: # calculate parametrisation
            m1 = (pointb.y-pointa.y)/(pointb.x-pointa.x) # y = m*x+b
            b1 =  pointb.y-pointb.x*m1 # b = y - m*x
            line1type = 'regular'
        else:
            line1type = 'vertical'
        cansee = True # assume they can see each other, until proven false
        for edge in self.edges: # check for every edge, including borders
            pointc = edge[0]
            pointd = edge[1]
            if (pointc.x - pointd.x) != 0: # calculate parametrisation
                m2 = (pointc.y - pointd.y) / (pointc.x - pointd.x)  # y = m*x+b
                b2 = pointc.y - pointc.x * m2  # b = y - m*x
                line2type = 'regular'
            else:
                line2type = 'vertical'
            # determine if and where the edge intersects with the two points
            intersecting = False
            if line1type != 'vertical' and line2type != 'vertical':
                if m1 != m2:
                    x = (b2 - b1) / (m1 - m2)
                    y = m1 * x + b1
                    intersecting = True
            elif line1type == 'vertical' and line2type != 'vertical':
                x = pointa.x
                y = m2 * x + b2
                intersecting = True
            elif line2type == 'vertical' and line1type != 'vertical':
                x = pointc.x
                y = m1 * x + b1
                intersecting = True

            # if they intersect, determine if this blocks visibility
            if intersecting == True:
                if line1type != 'vertical' and line2type != 'vertical':
                    if ((round(pointa.x,2) > round(x,2) and round(x,2) > round(pointb.x,2)) or (round(pointa.x,2) <round(x,2) and round(x,2) < round(pointb.x,2))) and\
                            ((round(pointc.x,2) > round(x,2) and round(x,2) > round(pointd.x,2)) or (round(pointc.x,2) <round(x,2) and round(x,2) < round(pointd.x,2))): # they cant see each other
                        cansee = False
                elif line1type == 'vertical':
                    assert pointc.x != pointd.x
                    if ((round(pointc.x,2) > round(x,2) and round(x,2) > round(pointd.x,2)) or (round(pointd.x,2) > round(x,2) and round(x,2) > round(pointc.x,2))) and\
                            ((round(pointa.y,2) > round(y,2) and round(y,2) > round(pointb.y,2)) or(round(pointb.y,2) > round(y,2) and round(y,2) > round(pointa.y,2))):
                        cansee = False
                elif line2type == 'vertical':
                    assert pointa.x != pointb.x
                    if ((round(pointa.x,2) > round(x,2) and round(x,2) > round(pointb.x,2)) or (round(pointb.x,2) > round(x,2) and round(x,2) > round(pointa.x,2))) and \
                            ((round(pointc.y,2) > round(y,2) and round(y,2) > round(pointd.y,2)) or(round(pointd.y,2) > round(y,2) and round(y,2) > round(pointc.y,2))):
                       cansee = False
                else:
                    assert(0==1)
        # if, after checking with all edges, they can still see each other,
        # we add the points to the visibility data structures
        if cansee == True:
            if pointa not in self.can_see.keys():
                self.can_see[pointa] = []
            if pointb not in self.can_see.keys():
                self.can_see[pointb] = []
            if pointb not in self.can_see[pointa]:
                self.can_see[pointa].append((pointb, self.distance(pointa, pointb)))
            if pointa not in self.can_see[pointb]:
                self.can_see[pointb].append((pointa, self.distance(pointa, pointb)))
        return cansee
    def pointLiesInTriangle(self, point):
        """
        To determine whether a point lies inside a triangle, we solve this vector equation system

        https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
        Solve the following equation system:
        p = p0 + (p1 - p0) * s + (p2 - p0) * t
        The point p is inside the triangle if 0 <= s <= 1 and 0 <= t <= 1 and s + t <= 1.
        s, t and 1 - s - t
        are called the barycentric coordinates of the point p.
        """
        for triangle in self.triangles:
            assert len(triangle) == 3
            a = point.x       - triangle[0].x
            b = triangle[1].x - triangle[0].x
            c = triangle[2].x - triangle[0].x
            d = point.y       - triangle[0].y
            e = triangle[1].y - triangle[0].y
            f = triangle[2].y - triangle[0].y
            if (b*f-e*c) != 0 and c!=0:
                S = (a*f-d*c)/(b*f-e*c)
                T = (a-b*S)/c
                if 0 < S and S < 1 and 0 < T and T < 1 and S+T < 1:
                    return True
            elif c==0 and b !=0 and f != 0:
                # a -bS = 0
                # => S = a/b
                # d -eS -fT = 0
                # => T = (d-eS)/f
                S = a/b
                T = (d-e*S)/f
                if 0 < S and S < 1 and 0 < T and T < 1 and S+T < 1:
                    return True
            elif c==0 and b ==0:
                # S and T are free?
                # shouldnt happen?
                print('Detected a singularity in triangle when checking if points are isolated')
                return False
            elif c==0 and b !=0 and f == 0:
                # a -bS = 0
                # => S = a/b
                # d -eS = 0
                # S is free?
                # shouldnt happen?
                print('Detected a singularity in triangle when checking if points are isolated')
                return False

        return False

    def pointIsWithinBorders(self, point):
        """
        determine whether a point is within the borders of the image
        if it is outside of the image, it makes no sense to add it to the graph
        """
        return self.minx <= point.x and point.x <= self.maxx and self.miny <= point.y and point.y <= self.maxy

    def addTriangles(self, poly):
        """
        given a polygon,
        split it into triangles by pointing from one corner to all others
        this works assuming our polygones are convex
        """
        if len(poly) == 3:
            self.triangles.append(poly)
        else:
            for i in range(len(poly)-2): #0, 1, ..., length-3:
                first_corner = poly[0]
                second_corner = poly[1+i]
                third_corner = poly[2+i]
                triangle = [first_corner, second_corner, third_corner]
                self.triangles.append(triangle)

    def addPolysAndMakeGraph(self, polys):
        """
        make the visibility graph using given polygones,
        assuming start & end points & borders are already added to the data structures
        """
        for poly in polys:
            self.addEdges(poly) # add all edges of each polygone
            self.addTriangles(poly) # add all triangles of each polygone
            for point in poly:
                assert (point.x, point.y) not in self.pointCoords
                self.pointCoords.append((point.x, point.y))
                self.points.add(point) # add all points of each corner of each polygone
        for pointa in self.points: # for each point
            if not self.pointLiesInTriangle(pointa): # if the point isnt located inside an object
                if self.pointIsWithinBorders(pointa): # if the point is located within map borders
                    for pointb in self.points: # for each OTHER point
                        if pointa != pointb:
                            if self.pointIsWithinBorders(pointb): # if the point is located within map borders
                                if not self.pointLiesInTriangle(pointb): # if the point isnt located inside an object
                                    bool = self.checkVis(pointa, pointb) # check if the two points can see each other
                                    # if they can, they are added to the can_see data structure
        self.initgraph() # convert the self.can_see dictionary as described in the introduction

    def addStart(self, point):
        """
        add the thymio start point to the visgraph
        """
        self.startPoint = point
        assert (point.x, point.y) not in self.pointCoords
        self.pointCoords.append((point.x, point.y))
        self.points.add(point)

    def addEnd(self, point):
        """
        add the goal to the visgraph
        """
        self.endPoint = point
        assert (point.x, point.y) not in self.pointCoords
        self.pointCoords.append((point.x, point.y))
        self.points.add(point)

    def addBorder(self, minx, maxx, miny, maxy):
        """
        - add borders around the graph so the thymio knows it cant go through there,
          so at all times:
          minx < x < maxx
          miny < y < maxy
        - the border's corners could be added to the visgraph's points,
          but we havent found any situations where this was advantageous so far.
          on the other hand, it makes the graph much more complex
        - the border is treated as any other obstacle edge
        """
        # define corners
        leftUnder = Points.Point(minx, miny)
        leftUpper = Points.Point(minx, maxy)
        rightUnder = Points.Point(maxx, miny)
        rightUpper = Points.Point(maxx, maxy)
        OPTION = False
        if OPTION:
            assert (leftUnder.x, leftUnder.y) not in self.pointCoords
            self.pointCoords.append((leftUnder.x, leftUnder.y))
            assert (leftUpper.x, leftUpper.y) not in self.pointCoords
            self.pointCoords.append((leftUpper.x, leftUpper.y))
            assert (rightUnder.x, rightUnder.y) not in self.pointCoords
            self.pointCoords.append((rightUnder.x, rightUnder.y))
            assert (rightUpper.x, rightUpper.y) not in self.pointCoords
            self.pointCoords.append((rightUpper.x, rightUpper.y))
            self.points.add(leftUnder)
            self.points.add(leftUpper)
            self.points.add(rightUnder)
            self.points.add(rightUpper)
        # define edges
        self.edges.add((leftUnder,rightUnder))
        self.edges.add((leftUnder, leftUpper))
        self.edges.add((rightUpper, leftUpper))
        self.edges.add((rightUpper, rightUnder))
        # store border values for later reference
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

    def canSee(self, a, b):
        """
        help function to determine wether a & b can see each other,
        return a boolean & the distance
        """
        assert a in self.points
        assert a in self.can_see.keys()
        for c, distance in self.can_see[a]:
            if c == b:
                return True, distance
        else:
            return False, 'inf'

    def initgraph(self):
        """
            - convert self.can_see into self.graph to a new dictionary that,
              given a point, returns all the other points this point can see.
            - upon calling self.graph[point_a][point_b], if point_a and point_b can see each other,
              the distance between these points is returned
        """
        self.graph = dict()
        for node in self.points:
            self.graph[node] = dict()
        for a in self.can_see.keys():
            for b,distance in self.can_see[a]:
                self.graph[a][b] = distance


    def Dijkstra(self, S = None):
        """
        python implementation of the dijkstra algorithm,
        - based on pseudocode from: http://www.gitta.info/Accessibiliti/en/html/Dijkstra_learningObject1.html
          with some slight modification to take into account that a point can be isolated
          ( if it lies inside an obstacle, which can happen if obstalces overlap,
            or if the end goal is too close to an object )
          which would crash the regular algorithm (as 'u', being the point in Q
            with the lowest distance might be None, while Q is not empty because it still contains the isolated point)

        S: the 'source', being the thymio's end goal
        Q: a list with all unvisited nodes in the graph
        u: the point in Q with the lowest distance to S
        v: a neighbour of u, in Q
        """

        if S == None:
            S = self.endPoint #S: 'source'
        distance = dict()
        previous = dict()
        Q = [] # UVISITED NODES

        for vertex in self.points:
            distance[vertex] = 'inf'
            previous[vertex] = None
            Q.append(vertex)
        distance[S] = 0

        def get_node_with_lowest_distance_to_S_from_Q(Q):
            mindist = 'inf'
            min = None
            for point in Q:
                dst = distance[point]
                if dst != 'inf':
                    if mindist == 'inf':
                        mindist = dst
                        min = point
                    elif mindist > dst:
                        mindist = dst
                        min = point
            # assert(min != None)
            # a point can be isolated if it lies inside an obstacle,
            # so this assert had to be removed, even though it makes sense to have it :/
            return min

        while Q != []: #while Q is not empty:
            u = get_node_with_lowest_distance_to_S_from_Q(Q) #u := node in Q with smallest dist[ ]
            if u != None: # some points are isolated, not part of dijkstra
                Q.remove(u) # remove u from Q
                for v in self.graph[u].keys(): # for each neighbour of U
                    if v in Q: # where v has not yet been removed from Q.
                        alt = distance[u] + self.graph[u][v] #alt := dist[u] + dist_between(u, v)
                        if distance[v] == 'inf' or alt < distance[v]: #if alt < dist[v]
                            distance[v] = alt
                            previous[v] = u
            else: # some points are isolated, not part of dijkstra
                break
        return distance, previous

    def getRoute(self):
        """
        return the route from self.startPoint to self.endPoint,
        using the dijkstra algorithm defined above
        route: [point1, point2,...]
        """
        route = []
        route.append(self.startPoint)
        distance, previous = self.Dijkstra()
        current = self.startPoint
        while current != self.endPoint:
            prev = previous[current]
            if prev == None:
                print('route did not lead to the end point (huhhhh??)  (end point might be isolated or out of bounds)')
                # end point might be isolated
                break
            route.append(prev)
            current = prev
        route.reverse()
        self.route = route
        return route

    def showresult(self):
        """
        display the graph
        green dot: point of origin
        red dot: goal
        black lines: edges that cannot be crossed
        purple lines: triangles, used to determine wether a point lies inside an object
        blue lines: visibility graph
        red line: road from point of origin to goal
        """
        plt.close()
        plt.figure(1)
        for triangle in self.triangles:
            visx = []
            visy = []
            for point in triangle:
                visx.append(point.x)
                visy.append(point.y)
            visx.append(visx[0])
            visy.append(visy[0])
            visx = np.asarray(visx)
            visy = np.asarray(visy)
            #plt.plot(visx, visy, color='yellow', linewidth='4')

        for edge in self.edges:
            edgesx = []
            edgesy = []
            for point in edge:
                edgesx.append(point.x)
                edgesy.append(point.y)
            edgesx = np.asarray(edgesx)
            edgesy = np.asarray(edgesy)
            plt.plot(edgesx, edgesy, color='black', linewidth='2')
        plt.plot([], [], color='black', linewidth='2', label='edges')

        for key in self.can_see.keys():
            for cansee in self.can_see[key]:
                cansee = cansee[0]
                visx = []
                visy = []
                visx.append(key.x)
                visy.append(key.y)
                visx.append(cansee.x)
                visy.append(cansee.y)
                visx = np.asarray(visx)
                visy = np.asarray(visy)
                plt.plot(visx, visy, color='blue', linewidth='1')
        plt.plot([], [], color='blue', linewidth='1', label='visibility')
        visx = []
        visy = []
        for point in self.route:
            visx.append(point.x)
            visy.append(point.y)
        visx = np.asarray(visx)
        visy = np.asarray(visy)
        plt.plot(visx, visy, color='red', linewidth='1', label='shortest path')

        plt.plot(self.startPoint.x, self.startPoint.y, 'go', markersize=10, label='origin')
        plt.plot(self.endPoint.x, self.endPoint.y, 'ro', markersize=10, label='goal')
        plt.gca().invert_yaxis()
        plt.savefig('visibilityGraph')
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.legend()
        plt.show()
        plt.close()

"""
############ EXAMPLE USAGE CODE ##############
poly = [Points.Point(5,3), Points.Point(9,3), Points.Point(8,7), Points.Point(4,7)]
poly2 = [Points.Point(3,1), Points.Point(6,-0.5), Points.Point(8,4)]
polys = [poly, poly2]
vg = visgraph()
vg.addStart(Points.Point(1,1))
vg.addEnd(Points.Point(10.5,2))
vg.addBorder(0, 11, 0, 11)

vg.addPolysAndMakeGraph(polys)
route = vg.getRoute()
vg.showresult()


"""