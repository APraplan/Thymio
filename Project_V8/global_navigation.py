import numpy as np
import pyvisgraph as vg
import cv_visgraph as vg2
import math
import matplotlib.pyplot as plt
import cv_Points as Points

def find_centroid(polygone):
    ##input : - polygones : a list of polygone describe by its vertices
    ##        - i : the i-th polygone of the list

    ##output : centroid :  the centroid (x,y) of the i-th polygone
    centroid = [0, 0]

    n = len(polygone)
    signedArea = 0
    assert n != 0
    # For all vertices
    for j in range(n):
        x0 = polygone[j].x
        y0 = polygone[j].y
        x1 = polygone[(j + 1) % n].x
        y1 = polygone[(j + 1) % n].y

        # Calculate value of A
        # using shoelace formulas
        A = (x0 * y1) - (x1 * y0)
        signedArea += A

        # Calculating coordinates of
        # centroid of polygon
        centroid[0] += (x0 + x1) * A
        centroid[1] += (y0 + y1) * A

    signedArea *= 0.5
    centroid[0] = (centroid[0]) / (6 * signedArea)
    centroid[1] = (centroid[1]) / (6 * signedArea)

    return centroid


def homothety(polygones, cent, k, i):
    ##input : - polygones : a list of polygone describe by its vertices
    #         - i : the i-th polygone of the list
    #         - cent : the centroid(x,y) of the i-th polygone
    #         - k : scale factor(>1) to enlarge the polygone

    ##output : - polygones[i] : the i-th enlarge polygone

    for j in range(len(polygones[i])):
        polygones[i][j].x = polygones[i][j].x + k * (polygones[i][j].x - cent[0])
        polygones[i][j].y = polygones[i][j].y + k * (polygones[i][j].y - cent[1])
    return polygones[i]


def homothetyl(polygone, cent, k):
    """ assume the points are in the right order"""
    listWithCornersOfLines = []
    for i in range(len(polygone)):
        if i != len(polygone) - 1:
            listWithCornersOfLines.append((polygone[i], polygone[i + 1]))
        else:
            listWithCornersOfLines.append((polygone[i], polygone[0]))
    enlargedLines = []
    for linePoints in listWithCornersOfLines:
        A = linePoints[0]
        B = linePoints[1]
        xa = A.x
        xb = B.x
        ya = A.y
        yb = B.y
        xc = cent[0]
        yc = cent[1]
        if A.x != B.x and A.y != B.y:
            a1 = (yb - ya) / (xb - xa)
            b1 = ya - a1 * xa
            a3 = -1 / a1
            b3 = yc - a3 * xc
            xp1 = (b3 - b1) / (a1 - a3)
            yp1 = a1 * xp1 + b1
            xv1 = xp1 - xc
            yv1 = yp1 - yc

            sz = math.sqrt(xv1 ** 2 + yv1 ** 2)
            sc = (sz + k) / sz
            xp2 = xv1 * sc + xc
            yp2 = yv1 * sc + yc
            a2 = a1
            b2 = yp2 - a2 * xp2
            a = a2
            b = b2
        elif A.x == B.x and A.y != B.y:  # vertical case
            # print('not coded yet')
            # assert(0 == 1)
            a = None
            b = A.x
            if A.x > cent[0]:
                b += k
            else:
                b -= k
        elif A.x != B.x and A.y == B.y:  # horizontal case
            a = 0
            b = A.y
            if A.y > cent[1]:
                b += k
            else:
                b -= k
        else:
            assert (0 == 1)
        enlargedLines.append((a, b))
    intersectingLines = []
    for i in range(len(enlargedLines)):
        if i != len(enlargedLines) - 1:
            intersectingLines.append((enlargedLines[i], enlargedLines[i + 1]))
        else:
            intersectingLines.append((enlargedLines[i], enlargedLines[0]))
    poly = []
    for couple in intersectingLines:
        a1, b1 = couple[0]
        a2, b2 = couple[1]
        #assert (a1 != a2)
        if a1 != a2: # should always be the case, but somehow it isnt
            if a1 != None and a2 != None:
                x = (b2 - b1) / (a1 - a2)
                y = a1 * x + b1
            elif a1 == None and a2 != None:
                x = b1
                y = a2 * x + b2
            elif a1 != None and a2 == None:
                x = b2
                y = a1 * x + b1
            else:
                assert (0 == 1)
            poly.append(Points.Point(x, y))
    return poly

def convertPolygonesFromVgToPoint(polygones):
    polys = []
    for polygone in polygones:
        poly = []
        for corner in polygone:
            poly.append(Points.Point(corner.x, corner.y))
        polys.append(poly)
    return polys

def path_planning(polygones, start_point, goal_point, k, gridMap = None, minx = None, miny = None, maxx= None, maxy=None, usePackage = False, showGraph = True):
    ##input : - polygones : a list of polygone describe by its vertices
    #         - start_point : start point of the Thymio robot
    #         - goal_point : goal point of the Thymio robot
    #         - k: nb of pixels to enlarge each polygone (outwards)

    ##output : shortest_path : for a set of polygones and a start and goal point, give the shortest path

    # first part : enlarge all obstacles so that the Thymio robot can avoid them
    polygg = polygones.copy()
    for j in range(len(polygones)):
        cent = find_centroid(polygones[j])
        polygones[j] = homothetyl(polygones[j], cent, k)

    # second part : build the visibility graph and compute the shortest path
    if usePackage:
        g = vg.VisGraph()
        g.build(polygones)
        shortest = g.shortest_path(start_point, goal_point)
    else:
        # the '==' function in vg.points is not well defined and causes our code to crash
        # convert all points to be used in visgraph to our own Point class
        # now '==' will refer to the objects themselves, without knowing what coordinate they refer to
        # this works as long as all points refer to unique coordinates
        # since we use floats, this condition should be met in our use case
        polygones2 = convertPolygonesFromVgToPoint(polygones)
        start_point2 = Points.Point(start_point.x, start_point.y)
        goal_point2 = Points.Point(goal_point.x, goal_point.y)

        # initialise visibility graph with start, goal, borders and polygones
        visg = vg2.visgraph()
        visg.addStart(start_point2)
        visg.addEnd(goal_point2)
        if gridMap != None:
            visg.addBorder(0, gridMap.originaldata.shape[1], 0, gridMap.originaldata.shape[0])
        else:
            visg.addBorder(minx, maxx, miny, maxy)
        print((visg.minx,visg.miny, visg.maxx, visg.maxy))
        print((goal_point2))
        assert visg.pointIsWithinBorders(start_point2)
        assert visg.pointIsWithinBorders(goal_point2)
        visg.addPolysAndMakeGraph(polygones2)

        # calculate the shortest route using Dijkstra
        shortest = visg.getRoute()

        if showGraph:
            # visualisation showing all the graph information
            visg.showresult()

    # visualisation showing only essential information
    plot_path(shortest, polygg, polygones, start_point, goal_point, gridMap, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
    return shortest, polygones


def plot_path(shortest_, polygg, polyg, start_point, goal_point, gridMap, minx=None, maxx=None, miny=None, maxy=None):
    polyg_x = []
    polyg_y = []
    polyg_x_e = []
    polyg_y_e = []
    if gridMap != None:
        plt.imshow(gridMap.originaldata, cmap='gray')

    for i in range(len(polygg)):
        #k = len(polygg[i])
        for j in range(len(polygg[i])):
            polyg_x.append(polygg[i][j].x)
            polyg_y.append(polygg[i][j].y)
            if j < len(polyg[i]): #
                polyg_x_e.append(polyg[i][j].x)
                polyg_y_e.append(polyg[i][j].y)

            if j == len(polygg[i]) - 1:
                polyg_x.append(polygg[i][0].x)
                polyg_y.append(polygg[i][0].y)

                polyg_x_e.append(polyg[i][0].x)
                polyg_y_e.append(polyg[i][0].y)

            x_coord_pol = np.asarray(polyg_x)
            y_coord_pol = np.asarray(polyg_y)

            x_coord_pol_e = np.asarray(polyg_x_e)
            y_coord_pol_e = np.asarray(polyg_y_e)

            plt.plot(x_coord_pol, y_coord_pol, color='green', linewidth='3')
            plt.plot(x_coord_pol_e, y_coord_pol_e, color='red', linewidth='3')
            plt.grid()


        polyg_x.clear()
        polyg_y.clear()
        polyg_x_e.clear()
        polyg_y_e.clear()

    shortest_path_x = np.empty(len(shortest_))
    shortest_path_y = np.empty(len(shortest_))

    for i in range(len(shortest_)):
        shortest_path_x[i] = shortest_[i].x
        shortest_path_y[i] = shortest_[i].y

    plt.plot(shortest_path_x, shortest_path_y, color='orange')
    plt.scatter(start_point.x, start_point.y, color='red')
    plt.scatter(goal_point.x, goal_point.y, color='red')
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Example path planning")
    if gridMap == None:
        plt.gca().invert_yaxis()
        xlst = [minx, maxx, maxx, minx, minx]
        ylst = [miny, miny, maxy, maxy, miny]
        xlst = np.asarray(xlst)
        ylst = np.asarray(ylst)
        plt.plot(xlst,ylst, color='black', linewidth=5)
    plt.grid()
    plt.show()
    plt.close()
    return