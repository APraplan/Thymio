import math
import pyvisgraph as vg
def homothetyl(polygone, cent, k):
    """ assume the points are in the right order"""
    listWithCornersOfLines = []
    for i in range(len(polygone)):
        if i != len(polygone)-1:
            listWithCornersOfLines.append((polygone[i], polygone[i+1]))
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
            a1 = (yb - ya)/(xb - xa)
            b1 = ya - a1*xa
            a3 = -1/a1
            b3 = yc - a3*xc
            xp1 = (b3-b1)/(a1-a3)
            yp1 = a1*xp1+b1
            xv1 = xp1-xc
            yv1 = yp1-yc
            sz = math.sqrt(xv1**2 + yv1**2)
            sc = (sz+k)/sz
            xp2 = xv1*sc + xc
            yp2 = yv1*sc + yc
            a2 = a1
            b2 = yp2 - a2*xp2
            a = a2
            b = b2
        elif A.x == B.x and A.y != B.y: # vertical case
            #print('not coded yet')
            #assert(0 == 1)
            a = None
            b = A.x
            if A.x > cent.x:
                b+= k
            else:
                b-= k
        elif A.x != B.x and A.y == B.y: # horizontal case
            a = 0
            b = A.y
            if A.y > cent.y:
                b+=k
            else:
                b-=k
        else: assert(0==1)
        enlargedLines.append((a,b))
    intersectingLines = []
    for i in range(len(enlargedLines)):
        if i != len(enlargedLines)-1:
            intersectingLines.append((enlargedLines[i], enlargedLines[i+1]))
        else:
            intersectingLines.append((enlargedLines[i], enlargedLines[0]))
    poly = []
    for couple in intersectingLines:
        a1, b1 = couple[0]
        a2, b2 = couple[1]
        assert (a1 != a2)
        if a1 != None and a2 != None:
            x = (b2-b1)/(a1-a2)
            y = a1*x+b1
        elif a1 == None and a2 != None:
            x = b1
            y = a2*x+b2
        elif a1 != None and a2 == None:
            x = b2
            y = a1 * x + b1
        else: assert(0==1)
        poly.append(vg.Point(x, y))
    return poly


