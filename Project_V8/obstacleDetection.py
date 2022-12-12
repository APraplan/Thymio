"""
Welcome to obstacle detection,
required files:
Edges.py, GridMap.py, Intersections.py, Lines.py, Loops.py, LoopSet.py, Maps.py, Node.py, NodeSet.py, Obstacles.py,
Pictures.py, Points.py, settings.py, SortedIntersections.py, SubFile.py

required packages:
cv2 (for plotting), time (mostly for print statements and timing performance),
numpy (for convultions like erosion and saving images), matplotlib (for plotting)

The goal of obstacle detection is to read a picture (file) and detect all obstacles.
The obstacles are polygones of any shape, with any amount of edges.
To achieve this goal, the hough transpose is calculated.
An advantage of this method is that the program can easily be adjusted to detect obstacles of different colours - not implemented,
we are less prone to lighting issues
and an analytical parametrisation of each line is returned, rather than a grid with black/white chunks,
which makes it much easier to make a visibility graph and PID controller
the houghspace can also be modified to detect round obstacles - not implemented
"""
# import external packages
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# import RoboticsProject
import cv_GridMap as GridMap
import cv_settings as settings
import cv_Maps as Maps
import cv_Pictures as Pictures
import cv_Points as Points


settings.current_test = 'real'

def settingsProfile0(k=5,n=4):
    """
        to avoid restarting the whole program, settings can be changed externally
        profile 0: combination of plus and square kernels
    """
    settings.TRESHOLDINGBLACK = 1500
    settings.GRIDMAPBLACKTRESHOLD = 100
    # grid settings
    settings.GRIDSIZE = 150  # larger to speed things up a bit
    settings.BORDER = 20  # larger for rounded objects
    settings.OUTERBORDER = 20

    # erosion & dilation of the hough space
    settings.HOUGHDILATIONKERNELTYPE = 'plus'  # default square, alternative plus
    settings.HOUGHEROSIONKERNELTYPE = 'square'
    settings.HOUGHDILATIONKERNELSIZE = 3
    settings.HOUGHEROSIONKERNELSIZE = 3
    settings.HOUGHDILATIONIT = 2 * k
    settings.HOUGHEROSIONIT = k

    # scaling of hough space
    settings.MAX = 350

    # erosion & dilation of the picture
    settings.DILATIONKERNELTYPE = 'square'
    settings.EROSIONKERNELTYPE = 'plus'
    settings.KERNELSIZEEROSION = 3
    settings.KERNELSIZEDILATION = 3
    settings.EROSIONITERATIONS = n + 1  # n+2 was fine for regular objects, not for compelex
    settings.DILATIONITERATIONS = n

def settingsProfile1(k=5,n=0):
    """
        to avoid restarting the whole program, settings can be changed externally
        profile 1: only plus kernels
    """
    settings.PICTURETRESHOLDINGBLACK = 1500 # brightness of a line after sobel
    # 1500 works well for low res pic
    # too low: less clean edges
    # too high: migh fail to detect edges
    settings.GRIDMAPBLACKTRESHOLD = 50 # color black for gridmap
    # 0 for most black pixels,
    # however at the border there are some of +_ 20
    # 10 works fine, 100 fuses unnecessairy borders
    # 70 good -> didnt fuse borders

    settings.MAPBLACKTRESHOLD = 70 # color black for loops -> obstacles
    # 0 for most black pixels,
    # however at the border there are some of +_ 20
    # 30 to be safe
    # 10 is too low -> gaps out of edges
    # 70 is acceptable

    # grid settings
    settings.GRIDSIZE = 20  # larger to speed things up a bit
    settings.BORDER = 1 # instead of 40 # larger for rounded objects
    settings.OUTERBORDER = 20
    
    # scaling of hough space
    settings.MAX = 1000

    # erosion & dilation of the hough space
    settings.HOUGHDILATIONKERNELTYPE = 'plus'  # default square, alternative plus
    settings.HOUGHEROSIONKERNELTYPE = 'plus'
    settings.HOUGHDILATIONKERNELSIZE = 3
    settings.HOUGHEROSIONKERNELSIZE = 3
    settings.HOUGHDILATIONIT = k + 2
    settings.HOUGHEROSIONIT = k

    # erosion & dilation of the picture
    # too eroded results in moree accidental lines 'n + 3 '
    settings.DILATIONKERNELTYPE = 'plus'
    settings.EROSIONKERNELTYPE = 'plus'
    settings.KERNELSIZEEROSION = 3
    settings.KERNELSIZEDILATION = 3
    settings.EROSIONITERATIONS = n + 1
    settings.DILATIONITERATIONS = n

    # plus plus plus plus 5 0 seems to be working the best


def settingsProfile2(k=5, n=0):
    """
        to avoid restarting the whole program, settings can be changed externally
        profile 1: only plus kernels
    """
    settings.PICTURETRESHOLDINGBLACK = 1500  # brightness of a line after sobel
    # 1500 works well for low res pic
    # too low: less clean edges
    # too high: migh fail to detect edges
    #settings.GRIDMAPBLACKTRESHOLD = 80
    settings.GRIDMAPBLACKTRESHOLD = 50  # color black for gridmap
    # 0 for most black pixels,
    # however at the border there are some of +_ 20
    # 10 works fine, 100 fuses unnecessairy borders
    # 70 good -> didnt fuse borders

    settings.MAPBLACKTRESHOLD = 70  # color black for loops -> obstacles
    # 0 for most black pixels,
    # however at the border there are some of +_ 20
    # 30 to be safe
    # 10 is too low -> gaps out of edges
    # 70 is acceptable

    # grid settings
    settings.GRIDSIZE = 20  # larger to speed things up a bit
    settings.BORDER = 1  # instead of 40 # larger for rounded objects
    settings.OUTERBORDER = 20

    # scaling of hough space
    settings.MAX = 1000

    # erosion & dilation of the hough space
    settings.HOUGHDILATIONKERNELTYPE = 'plus'  # default square, alternative plus
    settings.HOUGHEROSIONKERNELTYPE = 'plus'
    settings.HOUGHDILATIONKERNELSIZE = 3
    settings.HOUGHEROSIONKERNELSIZE = 3
    settings.HOUGHDILATIONIT = k + 2
    settings.HOUGHEROSIONIT = k

    # erosion & dilation of the picture
    # too eroded results in moree accidental lines 'n + 3 '
    settings.DILATIONKERNELTYPE = 'plus'
    settings.EROSIONKERNELTYPE = 'plus'
    settings.KERNELSIZEEROSION = 3
    settings.KERNELSIZEDILATION = 3
    settings.EROSIONITERATIONS = n + 1
    settings.DILATIONITERATIONS = n

    # plus plus plus plus 5 0 seems to be working the best
    
def visualsProfile1():
    """
        minimal outputs,
        clearing chat,
        prompting objects to manually verify if an object is real (and not part of the thymio)
    """
    # some display settings
    settings.SHOWINITIAL = 1
    settings.SHOWSEPERATEOBJECTS = 1
    settings.PROMPTPICS = 1
    settings.CLEAR = 1
    settings.SHOWENDRESULT = 1

    #settings.SHOWRESULT = 0
    #settings.PROMPTPICS = 0
    #settings.SHOWSEPERATEOBJECTS = 0

def visualsProfile0():
    """
        only show end result
    """
    # some display settings
    settings.CLEAR = 1
    settings.SHOWENDRESULT = 1

def visualsProfile2():
    """
        show every step
    """
    settings.SHOWINITIAL = 1  # show the inital picture with border
    #settings.DISPLAYFUSING = 1
    settings.SHOWSEPERATEOBJECTS = 1  # show each isolated object that comes out of the gridded map
    settings.SHOWFILTERSTEPS = 1  # show the filter steps the pictures go through to enhance edges
    settings.SHOWHOUGH = 1  # show the hough space (original, scaled, tresholded, dilated, eroded)
    settings.SHOWANALYSEDPICTURE = 1  # show object with all detected lines
    settings.SHOWSEPERATECLOSEDLOOPS = 1  # show the loops formed out of the lines
    settings.SHOWRESULT = 1  # show the end result
    settings.SHOWENDRESULT = 1

def step1():
    " function for debugging purposes, part 1 of analysePicture() "
    #settingsProfile1()  # settings file is not optimized yet

    # read the file at settings.FILENAME_MAIN and chop it up into chunks
    chunkedMap = GridMap.GridMap(cv2.imread(settings.getFILENAME_MAIN(), cv2.IMREAD_GRAYSCALE), settings)

    # fuse the chunks into bigger png's with (hopefully) one obstacle each
    chunkedMap.fuseAllBorders()

    # display the results
    chunkedMap.displayAllISolatedObjects()  # optional
    return chunkedMap

def step2(chunkedMap):
    " function for debugging purposes, part 2 of analysePicture() "
    # now we have smaller png files with obstacles, analyse each one individually:
    i = -1
    for obstaclePng in chunkedMap.obstaclesPng:
        # i = 0, 1, 2, ... is the respective name of each obstacle
        i += 1

        # add a border around the picture
        obstaclePng.data = chunkedMap.addBorder(obstaclePng.data)

        # save figure
        filePath = chunkedMap.saveFig(i, obstaclePng)

        # create Picture object, at this level we use the data of the image to find lines & intersections
        pictureData = Pictures.Picture(cv2.imread(filePath, cv2.IMREAD_GRAYSCALE), filePath, i, settings)

        # the image goes through some filters to enhance edges
        blurredImage = pictureData.blurImage(pictureData.data)
        edgeEnhancedImage = pictureData.edgeEnhance(blurredImage)
        tresholdedImage = pictureData.treshold(edgeEnhancedImage)
        dilatedImage = pictureData.dilate(tresholdedImage)
        erodedImage = pictureData.erode(dilatedImage)
        pictureData.processedData = erodedImage

        # the hough transform is calculated & scaled to find edges
        houghTransform = pictureData.calculate_hough_transform(pictureData.processedData)
        houghTransform = pictureData.enhanceHough(houghTransform)

        # find maxima from the hough space, calculate lines & intersections , visualise
        maxima = pictureData.find_optima(houghTransform)
        pictureData.find_all_lines_from_maxima(maxima)
        pictureData.find_intersections_from_lines()
        pictureData.visualiseLinesAndIntersections()  # optional

        # create Map object, at this level we use lines & intersections to make and analyse a graph
        obstacleMap = Maps.Map(filePath, time.time(), pictureData)

        # sort the intersections, use them to make a graph out of the intersections (nodes with neighbours)
        obstacleMap.sortIntersections()
        obstacleMap.makeGraph()
        # use the neighbours to make closed loops of neighbours , visualise
        obstacleMap.makeLoops()
        obstacleMap.visualiseLoops()  # optional
        # keep black loops, discard white loops (these are a byproduct of lines intersecting outside of the obstacle)
        obstacleMap.getObstaclesFromGraph()
        obstacleMap.visualiseObstacles()  # optional

        # all the obstacles we found were returned in a local reference frame -> transform to global map & add to global set
        chunkedMap.convertObstaclesToGlobal(obstacleMap, obstaclePng)
    # do this twice as we have 2 borders: both for exterior border and individual borders
    chunkedMap.convertObstaclesToGlobal()
    return chunkedMap

def step3(chunkedMap):
    " function for debugging purposes, part 3 of analysePicture() "
    # display the results
    chunkedMap.displayGlobalObstacles()  # optional

    # export in a format suitable for the visibility graph:
    # [[corners_of_obstacle_1],[corners_of_obstacle_2]...]
    # with the corners a list of vg.Point(), ordered so that they can be followed into a loop
    output = chunkedMap.exportGlobalObstaclesFlippedXY()  # uses self.globalObstacles:

    # sometimes there are multiple points along the same line, delete these
    # (otherwise program crashes later when they use cos rule)
    simplifiedoutput = chunkedMap.simplify(output)

    # display result
    chunkedMap.displaySimplified(output, simplifiedoutput)
    return chunkedMap, simplifiedoutput, chunkedMap.determineThymioSize()


def analysePicture():
    """ analyse the picture saved at settings.getFILENAME_MAIN():
        - surround picture by white border
        - divide the picture into smaller parts containing individual obstacles
        - for each part:
            - go through some filter steps to enhance edges
            - use the hough transform to find lines
            # in a local coordinate system:
            - find intersections from these lines
            - make a graph out of the intersections
            - make loops out of the graph
            - use the loops to reconstruct the obstacles
            - convert the coordinates back to the global coordinate system (in the whole picture)
        - convert & simplify the obstacles to visgraph format
    """

    # read the file at settings.FILENAME_MAIN and chop it up into chunks
    chunkedMap = GridMap.GridMap(cv2.imread(settings.getFILENAME_MAIN(), cv2.IMREAD_GRAYSCALE), settings)

    # fuse the chunks into bigger png's with (hopefully) one obstacle each
    chunkedMap.fuseAllBorders()

    # display the results
    chunkedMap.displayAllISolatedObjects()  # optional

    # now we have smaller png files with obstacles, analyse each one individually:
    i = -1
    for obstaclePng in chunkedMap.obstaclesPng:
        # i = 0, 1, 2, ... is the respective name of each obstacle
        i += 1

        # save figure
        filePath = chunkedMap.saveFig(i, obstaclePng)

        # create Picture object, at this level we use the data of the image to find lines & intersections
        pictureData = Pictures.Picture(cv2.imread(filePath, cv2.IMREAD_GRAYSCALE), filePath, i, settings)

        # the image goes through some filters to enhance edges
        blurredImage = pictureData.blurImage(pictureData.data)
        edgeEnhancedImage = pictureData.edgeEnhance(blurredImage)
        tresholdedImage = pictureData.treshold(edgeEnhancedImage)
        dilatedImage = pictureData.dilate(tresholdedImage)
        erodedImage = pictureData.erode(dilatedImage)
        pictureData.processedData = erodedImage

        # the hough transform is calculated & scaled to find edges
        houghTransform = pictureData.calculate_hough_transform(pictureData.processedData)
        houghTransform = pictureData.enhanceHough(houghTransform)

        # find maxima from the hough space, calculate lines & intersections , visualise
        maxima = pictureData.find_optima(houghTransform)
        pictureData.find_all_lines_from_maxima(maxima)
        pictureData.find_intersections_from_lines()
        pictureData.visualiseLinesAndIntersections() # optional

        # create Map object, at this level we use lines & intersections to make and analyse a graph
        obstacleMap = Maps.Map(filePath, time.time(), pictureData)

        # sort the intersections, use them to make a graph out of the intersections (nodes with neighbours)
        obstacleMap.sortIntersections()
        obstacleMap.makeGraph()
        # use the neighbours to make closed loops of neighbours , visualise
        obstacleMap.makeLoops()
        obstacleMap.visualiseLoops() # optional
        # keep black loops, discard white loops (these are a byproduct of lines intersecting outside of the obstacle)
        obstacleMap.getObstaclesFromGraph()
        obstacleMap.visualiseObstacles() # optional

        # all the obstacles we found were returned in a local reference frame -> transform to global map
        chunkedMap.convertObstaclesToGlobal(obstacleMap, obstaclePng)

    # display the results
    chunkedMap.displayGlobalObstacles()  # optional

    # export in a format suitable for the visibility graph:
    # [[corners_of_obstacle_1],[corners_of_obstacle_2]...]
    # with the corners a list of vg.Point(), ordered so that they can be followed into a loop
    output = chunkedMap.exportGlobalObstaclesFlippedXY() # uses self.globalObstacles:

    # sometimes there are multiple points along the same line, delete these
    # (otherwise program crashes later when they use cos rule)
    simplifiedoutput = chunkedMap.simplify(output)

    # display result
    chunkedMap.displaySimplified(output, simplifiedoutput)
    return chunkedMap, simplifiedoutput, chunkedMap.determineThymioSize()

def analysePicture2():
    """
    Similar to analysePicture(), but each individual obstacle is also provided with a border
    - takes a bit longer
    - more allows for a smaller white border when searching for obstacles (gridMap), so obstacles can be
      placed closer to each other
    Steps:
        analyse the picture saved at settings.getFILENAME_MAIN():
        - surround picture by white border
        - divide the picture into smaller parts containing individual obstacles
        - for each part:
            - surround by border
            - go through some filter steps to enhance edges
            - use the hough transform to find lines
            # in a local coordinate system:
            - find intersections from these lines
            - make a graph out of the intersections
            - make loops out of the graph
            - use the loops to reconstruct the obstacles
            - convert the coordinates back to the global coordinate system (in the whole picture)
        - convert the coordinates once more, to take the second border into account
        - convert & simplify the obstacles to visgraph format
    """
    #settingsProfile1()  # settings file is not optimized yet

    # read the file at settings.FILENAME_MAIN and chop it up into chunks
    chunkedMap = GridMap.GridMap(cv2.imread(settings.getFILENAME_MAIN(), cv2.IMREAD_GRAYSCALE), settings)

    # fuse the chunks into bigger png's with (hopefully) one obstacle each
    chunkedMap.fuseAllBorders()

    # display the results
    chunkedMap.displayAllISolatedObjects()  # optional

    " function for debugging purposes, part 2 of analysePicture() "
    # now we have smaller png files with obstacles, analyse each one individually:
    i = -1
    for obstaclePng in chunkedMap.obstaclesPng:
        # i = 0, 1, 2, ... is the respective name of each obstacle
        i += 1

        # add a border around the picture
        obstaclePng.data = chunkedMap.addBorder(obstaclePng.data)

        # save figure
        filePath = chunkedMap.saveFig(i, obstaclePng)

        # create Picture object, at this level we use the data of the image to find lines & intersections
        pictureData = Pictures.Picture(cv2.imread(filePath, cv2.IMREAD_GRAYSCALE), filePath, i, settings)

        # the image goes through some filters to enhance edges
        blurredImage = pictureData.blurImage(pictureData.data)
        edgeEnhancedImage = pictureData.edgeEnhance(blurredImage)
        tresholdedImage = pictureData.treshold(edgeEnhancedImage)
        dilatedImage = pictureData.dilate(tresholdedImage)
        erodedImage = pictureData.erode(dilatedImage)
        pictureData.processedData = erodedImage

        # the hough transform is calculated & scaled to find edges
        houghTransform = pictureData.calculate_hough_transform(pictureData.processedData)
        houghTransform = pictureData.enhanceHough(houghTransform)

        # find maxima from the hough space, calculate lines & intersections , visualise
        maxima = pictureData.find_optima(houghTransform)
        pictureData.find_all_lines_from_maxima(maxima)
        pictureData.find_intersections_from_lines()
        pictureData.visualiseLinesAndIntersections()  # optional

        # create Map object, at this level we use lines & intersections to make and analyse a graph
        obstacleMap = Maps.Map(filePath, time.time(), pictureData)

        # sort the intersections, use them to make a graph out of the intersections (nodes with neighbours)
        obstacleMap.sortIntersections()
        obstacleMap.makeGraph()
        # use the neighbours to make closed loops of neighbours , visualise
        obstacleMap.makeLoops()
        obstacleMap.visualiseLoops()  # optional
        # keep black loops, discard white loops (these are a byproduct of lines intersecting outside of the obstacle)
        obstacleMap.getObstaclesFromGraph()
        obstacleMap.visualiseObstacles()  # optional

        # all the obstacles we found were returned in a local reference frame -> transform to global map & add to global set
        chunkedMap.convertObstaclesToGlobal(obstacleMap, obstaclePng)
    # do this twice as we have 2 borders: both for exterior border and individual borders
    chunkedMap.convertObstaclesToGlobal()

    " function for debugging purposes, part 3 of analysePicture() "
    # display the results
    chunkedMap.displayGlobalObstacles()  # optional

    # export in a format suitable for the visibility graph:
    # [[corners_of_obstacle_1],[corners_of_obstacle_2]...]
    # with the corners a list of vg.Point(), ordered so that they can be followed into a loop
    output = chunkedMap.exportGlobalObstaclesFlippedXY()  # uses self.globalObstacles:

    # sometimes there are multiple points along the same line, delete these
    # (otherwise program crashes later when they use cos rule)
    simplifiedoutput = chunkedMap.simplify(output)

    # display result
    chunkedMap.displaySimplified(output, simplifiedoutput)
    return chunkedMap, simplifiedoutput, chunkedMap.determineThymioSize()

def savePoints(gridMap, simplifiedoutput, thymiowidth):
    txt = ''
    txt += str(thymiowidth)
    x = str(round(gridMap.originaldata.shape[1] / 10000, 4))
    while len(x) < 6:
        x += '0'
    y = str(round(gridMap.originaldata.shape[0] / 10000, 4))
    while len(y) < 6:
        y += '0'

    txt += '{' + x + ',' + y + '}'
    for obstacle in simplifiedoutput:
        txt += '['
        txt += '/'
        for corner in obstacle:
            if corner.x >= 0:
                sign = '+'
            else:
                sign = '-'
            x = sign + str(abs(round(corner.x / 10000, 4)))
            while len(x) < 7:
                x += '0'

            if corner.y >= 0:
                sign = '+'
            else:
                sign = '-'
            y = sign + str(abs(round(corner.y / 10000, 4)))
            while len(y) < 7:
                y += '0'
            txt += '(' + x + ',' + y + ')' + '/'
        txt += ']'
    print('save polys:')
    print(txt)
    return txt

def getPolys(string):
    thymiosize = int(string[0:2])
    maxx = int(float(string[3:9])*10000)
    maxy = int(float(string[10:16])*10000)
    i = 0
    obstacles = []
    while i < len(string):
        if string[i] == '[':
            obstacle = []
        elif string[i] == ']':
            obstacles.append(obstacle)
        elif string[i] == '(':
            x = int(float(string[i+1:i+1+7])*10000)
            y = int(float(string[i+2+7:i+2+7+7])*10000)
            obstacle.append(Points.Point(x,y))
        i+=1
    return thymiosize, 0, 0, maxx, maxy, obstacles