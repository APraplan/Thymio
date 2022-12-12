import cv_SubFile as SubFile
import numpy as np
#import settings
import cv_Maps as Maps
import matplotlib.pyplot as plt
import pathlib
import time
import cv_Points as Points
import cv_Edges as Edges
import cv_Obstacles as Obstacles
from PIL import Image
import pyvisgraph as vg
import os
from IPython.display import clear_output

class GridMap:
    "when i try to analyse multiple objects on a map it causes absolute mayhem when calculating the hough transform,"
    "so im going to split up the map into grids"
    "using these grids we can analyse objects individually"
    "then these individual objects van be analysed with higher precision"
    def __init__(self, file, settings):
        self.settings = settings
        """
        file: a np.array
        surround the file by a white border
        cut it into smaller files
        check if the smaller files are completely white
        """
        print(' ')
        print('Splitting file into chunks')
        print('- file: '+str(settings.getFILENAME_MAIN()))
        print('- type: '+str(type(file)))
        print('- resolution: '+str(file.shape))
        self.offsetx = self.settings.OUTERBORDER
        self.offsety = self.settings.OUTERBORDER
        self.globalObstacles = set()
        self.data = file
        self.originaldata = file

        ###
        # I added a border all around to protect objects at the edge for two reasons:
        # 1.
            # when a slight inaccuracy in the hough transform -> error in line location
            # causes the intersection to fall out of the image,
            # the intersection would be wrongfully discarded
            # by adding a border, the intersection is in this border and isnt discarded
        # 2.
            # when an obstacle is partially situated outside of the picture, the edge (parallel with the border of the picture)
            # wouldnt be detected
        ###

        # border
        if self.settings.BORDERCOLOUR == 'white':
            white = 255
        else:
            white = max(self.originaldata[0]) # take the lowest value of the lower border
        newData = np.ndarray((self.data.shape[0]+2*self.settings.OUTERBORDER,self.data.shape[1]+2*self.settings.OUTERBORDER))
        horizontallayer = np.ndarray((1,self.data.shape[1]+2*self.settings.OUTERBORDER))
        for y in range(self.data.shape[1]+2*self.settings.OUTERBORDER):
            horizontallayer[0,y] = white
        for x in range(self.settings.OUTERBORDER):
            newData[x] = horizontallayer #higher border
            newData[newData.shape[0]-1-x] = horizontallayer #lower border
        for x in range(self.data.shape[0]):
            for i in range(self.settings.OUTERBORDER):
                newData[x+self.offsetx, i] = white #left border
                newData[x + self.offsetx, newData.shape[1]-1-i] = white #right border
            for y in range(self.data.shape[1]):
                newData[x+self.offsetx, y+self.offsety] = self.data[x,y]
        self.data = newData

        # display
        plt.close()
        plt.imshow(self.data, cmap='gray')
        plt.title("Picture surrounded by border")
        plt.xlabel('y [pixels]')
        plt.ylabel('x [pixels]')
        if self.settings.SHOWINITIAL:
            plt.show()
        filePath = str(pathlib.Path().resolve()) + '/TEMP/currentFig.png'
        plt.savefig(filePath)
        plt.close()

        # determine how many chunks to make
        self.gridsize = self.settings.GRIDSIZE
        self.obstaclesPng = set()
        self.globalObstacles = set()
        if self.data.shape[0] >= self.gridsize:
            self.gridh = self.data.shape[0] // self.gridsize
            self.gridheightrest = self.data.shape[0] % self.gridsize
        else:
            self.gridh = 0
            self.gridheightrest = self.data.shape[0]
        if self.data.shape[1] >= self.gridsize:
            self.gridw = self.data.shape[1] // self.gridsize
            self.gridwidthrest = self.data.shape[1] % self.gridsize
        else:
            self.gridw = 0
            self.gridwidthrest = self.data.shape[1]

        def get_sub_file(self, minx, maxx, miny, maxy):
            """
            get the data from a chunk defined by borders minx, maxx, miny & maxy
            """
            data = self.data
            x = 0
            saveddata = []
            for xdata in data:
                if x >= minx and x < maxx:
                    saveddata.append(xdata[miny:maxy])
                x += 1
            saveddata = np.array(saveddata, dtype='uint8')
            # check if the chunk is completely white
            allWhite = True
            for x in range(saveddata.shape[0]):
                for y in range(saveddata.shape[1]):
                    if saveddata[x, y] < self.settings.GRIDMAPBLACKTRESHOLD:
                        allWhite = False
            return saveddata, allWhite

        self.subfiles = dict() # save the chunks in a dictionary
        # cases in the middle of the picture
        for h in range(self.gridh):
            for w in range(self.gridw):
                subfile, allWhite = get_sub_file(self, h * self.gridsize, h * self.gridsize + self.gridsize,
                                                 w * self.gridsize, w * self.gridsize + self.gridsize)
                coordinate = (h, w)
                identitySet = set()
                identitySet.add(coordinate)
                self.subfiles[coordinate] = SubFile.SubFile(subfile, h * self.gridsize, w * self.gridsize, self.gridsize,
                                                          self.gridsize, identitySet, allWhite)

        # border cases:
        if self.gridheightrest != 0:
            for w in range(self.gridw):
                subfile, allWhite = get_sub_file(self, self.gridh * self.gridsize,
                                                 self.gridh * self.gridsize + self.gridheightrest, w * self.gridsize,
                                                 w * self.gridsize + self.gridsize)
                coordinate = (self.gridh, w)
                identitySet = set()
                identitySet.add(coordinate)
                self.subfiles[coordinate] = SubFile.SubFile(subfile, self.gridh * self.gridsize, w * self.gridsize,
                                                          self.gridheightrest, self.gridsize, identitySet, allWhite)
        if self.gridwidthrest != 0:
            for h in range(self.gridh):
                subfile, allWhite = get_sub_file(self, h * self.gridsize, h * self.gridsize + self.gridsize,
                                                 self.gridw * self.gridsize,
                                                 self.gridw * self.gridsize + self.gridwidthrest)
                coordinate = (h, self.gridw)
                identitySet = set()
                identitySet.add(coordinate)
                self.subfiles[coordinate] = SubFile.SubFile(subfile, h * self.gridsize, self.gridw * self.gridsize,
                                                          self.gridsize, self.gridwidthrest, identitySet, allWhite)

        if self.gridheightrest != 0 and self.gridwidthrest != 0:
            subfile, allWhite = get_sub_file(self, self.gridh * self.gridsize,
                                             self.gridh * self.gridsize + self.gridheightrest,
                                             self.gridw * self.gridsize,
                                             self.gridw * self.gridsize + self.gridwidthrest)
            coordinate = (self.gridh, self.gridw)
            identitySet = set()
            identitySet.add(coordinate)
            self.subfiles[coordinate] = SubFile.SubFile(subfile, self.gridh * self.gridsize, self.gridw * self.gridsize,
                                                      self.gridheightrest, self.gridwidthrest, identitySet, allWhite)


        # making neighbours
        if self.gridheightrest > 0:
            self.maxx = self.gridh
        else:
            self.maxx = self.gridh-1
        if self.gridwidthrest > 0:
            self.maxy = self.gridw
        else:
            self.maxy = self.gridw-1
        for x in range(self.maxx + 1):
            for y in range(self.maxy + 1):
                coordinate = (x, y)
                current = self.subfiles[coordinate]
                if x > 0:  # we're not in the highest layer
                    upperCoordinate = (x - 1, y)
                    upper = self.subfiles[upperCoordinate]
                    current.addUpper(upper)
                if self.gridheightrest > 0:
                    if x < self.gridh:  # we're not in the lowest layer
                        lowerCoordinate = (x + 1, y)
                        lower = self.subfiles[lowerCoordinate]
                        current.addLower(lower)
                else:
                    if x < self.gridh-1:  # we're not in the lowest layer
                        lowerCoordinate = (x + 1, y)
                        lower = self.subfiles[lowerCoordinate]
                        current.addLower(lower)

                if y > 0:  # we're not in the leftmost layer
                    leftCoordinate = (x, y - 1)
                    left = self.subfiles[leftCoordinate]
                    current.addLeft(left)
                if self.gridwidthrest > 0:
                    if y < self.gridw:  # we're not in the rightmost layer
                        rightCoordinate = (x, y + 1)
                        right = self.subfiles[rightCoordinate]
                        current.addRight(right)
                else:
                    if y < self.gridw - 1:  # we're not in the rightmost layer
                        rightCoordinate = (x, y + 1)
                        right = self.subfiles[rightCoordinate]
                        current.addRight(right)

        # we should now have a complete graph
        # checking:
        checking = True
        if checking:
            for x in range(self.maxx + 1):
                for y in range(self.maxy + 1):
                    identity = (x, y)
                    current = self.subfiles[identity]
                    noneSet = set()
                    if current.left != noneSet:
                        for left in current.left:
                            assert current in left.right
                    if current.right != noneSet:
                        for right in current.right:
                            assert current in right.left
                    if current.lower != noneSet:
                        for lower in current.lower:
                            assert current in lower.upper
                    if current.upper != noneSet:
                        for upper in current.upper:
                            assert current in upper.lower
                    if x == 0:
                        assert current.upper == noneSet
                    else:
                        assert current.upper != noneSet
                    if x == self.maxx:
                        assert current.lower == noneSet
                    else:
                        assert current.lower != noneSet
                    if y == 0:
                        assert current.left == noneSet
                    else:
                        assert current.left != noneSet
                    if y == self.maxy:
                        assert current.right == noneSet
                    else:
                        assert current.right != noneSet

    ### recycle add border code to add individual white borders
    def addBorder(self, data):
            # border
            if self.settings.BORDERCOLOUR == 'white':
                white = 255
            else:
                white = max(data[0])  # take the lowest value of the lower border
            newData = np.ndarray((data.shape[0] + 2 * self.settings.OUTERBORDER,
                                  data.shape[1] + 2 * self.settings.OUTERBORDER))
            horizontallayer = np.ndarray((1, data.shape[1] + 2 * self.settings.OUTERBORDER))
            for y in range(data.shape[1] + 2 * self.settings.OUTERBORDER):
                horizontallayer[0, y] = white
            for x in range(self.settings.OUTERBORDER):
                newData[x] = horizontallayer  # higher border
                newData[newData.shape[0] - 1 - x] = horizontallayer  # lower border
            for x in range(data.shape[0]):
                for i in range(self.settings.OUTERBORDER):
                    newData[x + self.offsetx, i] = white  # left border
                    newData[x + self.offsetx, newData.shape[1] - 1 - i] = white  # right border
                for y in range(data.shape[1]):
                    newData[x + self.offsetx, y + self.offsety] = data[x, y]

            return newData
    ################################################################################################################
    ################## NOW WE HAVE FULLY SPLIT UP OUR FILE, WE FUSE CHUNKS THAT FORM AN OBJECT TOGETHER ############
    ################################################################################################################

    def fuseColoredBorder(self, prev=None):
        """
        fuse the chunks that are black at their border (to reconstruct full objects)
        """
        def bubbleSort(neighbours, direction):
            item = neighbours
            if direction == 'horizontal':  # left to right
                for i in range(len(item)):
                    already_sorted = True
                    for j in range(len(item) - i - 1):
                        xj = item[j].ymin
                        xj_plus_1 = item[j + 1].ymin
                        if xj > xj_plus_1:
                            item[j], item[j + 1] = item[j + 1], item[j]
                        already_sorted = False
                    if already_sorted:
                        break
            elif direction == 'vertical':  # top to bottom, so smallest y first
                for i in range(len(item)):
                    already_sorted = True
                    for j in range(len(item) - i - 1):
                        xj = item[j].xmin
                        xj_plus_1 = item[j + 1].xmin
                        if xj > xj_plus_1:
                            item[j], item[j + 1] = item[j + 1], item[j]
                        already_sorted = False
                    if already_sorted:
                        break
            return neighbours
        ############################################
        # sorry about the if True's, I had to remove some if conditions
        ############################################
        fusedSomething = False # only fuse two chunks at a time
        fusedObject = None
        # first, we extend 'prev' which would be the last object extended before now
        # this way we dont stop extending prev until it is completely done
        if prev != None:
            current = prev
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                if current.checkUpperBorder() == False:  # so there is some color at the upper border
                                    neighbours_to_fuse = []
                                    for neighbour in current.upper:
                                        # to fuse something on top, we must first fuse all upper neighbours
                                        neighbours_to_fuse.append(neighbour)
                                    # as a result of fusing sometimes you are your own neighbour
                                    neighbours_to_fuse = current.differentChunksList(neighbours_to_fuse, self)
                                    if len(neighbours_to_fuse) == 1:
                                        fusedObject = current.fuseUpper(neighbours_to_fuse[0], self)
                                        fusedSomething = True
                                    elif len(neighbours_to_fuse) > 1:
                                        # multiple neighbours -> check their compatability:
                                        compatible = True
                                        for neighbour in neighbours_to_fuse:
                                            if neighbour.sizeh != neighbours_to_fuse[0].sizeh:
                                                compatible = False
                                        if compatible == True:  # now we sort these neighbours from left to right
                                            neighbours_to_fuse = bubbleSort(neighbours_to_fuse, 'horizontal')
                                            sizew = 0
                                            for neighbour in neighbours_to_fuse:
                                                sizew += neighbour.sizew
                                            if sizew == current.sizew:
                                                fusedNeighbours = neighbours_to_fuse[0]
                                                for neighbour in neighbours_to_fuse[1:]:
                                                    fusedNeighbours = fusedNeighbours.fuseRight(neighbour, self)
                                                    # now we have one big upper neighbour and we can fuse this one to our current:
                                                fusedObject = current.fuseUpper(fusedNeighbours, self)
                                                fusedSomething = True
                                            else:
                                                pass
                                                # print('WEIRD CASE OF NOT FUSING')
                                # to reduce complexity we only fuse one at a time
                            if fusedSomething == False:
                                if current.checkLowerBorder() == False:  # so there is some color at the upper border
                                    neighbours_to_fuse = []
                                    for neighbour in current.lower:
                                        # to fuse something on top, we must first fuse all upper neighbours
                                        neighbours_to_fuse.append(neighbour)
                                    # as a result of fusing sometimes you are your own neighbour
                                    neighbours_to_fuse = current.differentChunksList(neighbours_to_fuse, self)
                                    if len(neighbours_to_fuse) == 1:
                                        fusedObject = current.fuseLower(neighbours_to_fuse[0], self)
                                        fusedSomething = True
                                    elif len(neighbours_to_fuse) > 1:
                                        # multiple neighbours -> check their compatability:
                                        compatible = True
                                        for neighbour in neighbours_to_fuse:
                                            if neighbour.sizeh != neighbours_to_fuse[0].sizeh:
                                                compatible = False
                                        if compatible == True:  # now we sort these neighbours from left to right
                                            neighbours_to_fuse = bubbleSort(neighbours_to_fuse, 'horizontal')
                                            sizew = 0
                                            for neighbour in neighbours_to_fuse:
                                                sizew += neighbour.sizew
                                            if sizew == current.sizew:  # second compatability check
                                                fusedNeighbours = neighbours_to_fuse[0]
                                                for neighbour in neighbours_to_fuse[1:]:
                                                    fusedNeighbours = fusedNeighbours.fuseRight(neighbour, self)
                                                # now we have one big lower neighbour and we can fuse this one to our current:
                                                fusedObject = current.fuseLower(fusedNeighbours, self)
                                                fusedSomething = True
                                            else:
                                                pass
                                                # print('WEIRD CASE OF NOT FUSING')
                                                # not so weird really
                            if fusedSomething == False:
                                if current.checkRightBorder() == False:  # so there is some color at the right border
                                    neighbours_to_fuse = []
                                    for neighbour in current.right:
                                        # to fuse something on top, we must first fuse all upper neighbours
                                        neighbours_to_fuse.append(neighbour)
                                    # as a result of fusing sometimes you are your own neighbour
                                    neighbours_to_fuse = current.differentChunksList(neighbours_to_fuse, self)
                                    if len(neighbours_to_fuse) == 1:
                                        fusedObject = current.fuseRight(neighbours_to_fuse[0], self)
                                        fusedSomething = True
                                    elif len(neighbours_to_fuse) > 1:
                                        # multiple neighbours -> check their compatability:
                                        compatible = True
                                        for neighbour in neighbours_to_fuse:
                                            if neighbour.sizew != neighbours_to_fuse[0].sizew:
                                                compatible = False
                                        if compatible == True:  # now we sort these neighbours from top to bottom
                                            neighbours_to_fuse = bubbleSort(neighbours_to_fuse, 'vertical')
                                            sizeh = 0
                                            for neighbour in neighbours_to_fuse:
                                                sizeh += neighbour.sizeh
                                            if sizeh == current.sizeh:
                                                # for neighbour in neighbours_to_fuse: vertical bubblesort doestn work
                                                #    print(neighbour.xmin)
                                                fusedNeighbours = neighbours_to_fuse[0]
                                                for neighbour in neighbours_to_fuse[1:]:
                                                    fusedNeighbours = fusedNeighbours.fuseLower(neighbour, self)
                                                    # now we have one big lower neighbour and we can fuse this one to our current:
                                                fusedObject = current.fuseRight(fusedNeighbours, self)
                                                fusedSomething = True
                                            else:
                                                pass
                                                # print('WEIRD CASE OF NOT FUSING')
                            if fusedSomething == False:
                                if current.checkLeftBorder() == False:  # so there is some color at the right border
                                    neighbours_to_fuse = []
                                    for neighbour in current.left:
                                        # to fuse something on top, we must first fuse all upper neighbours
                                        neighbours_to_fuse.append(neighbour)
                                    # as a result of fusing sometimes you are your own neighbour
                                    neighbours_to_fuse = current.differentChunksList(neighbours_to_fuse, self)
                                    if len(neighbours_to_fuse) == 1:
                                        fusedObject = current.fuseLeft(neighbours_to_fuse[0], self)
                                        fusedSomething = True
                                    elif len(neighbours_to_fuse) > 1:
                                        # multiple neighbours -> check their compatability:
                                        compatible = True
                                        for neighbour in neighbours_to_fuse:
                                            if neighbour.sizew != neighbours_to_fuse[0].sizew:
                                                compatible = False
                                        if compatible == True:  # now we sort these neighbours from top to bottom
                                            neighbours_to_fuse = bubbleSort(neighbours_to_fuse, 'vertical')
                                            sizeh = 0
                                            for neighbour in neighbours_to_fuse:
                                                sizeh += neighbour.sizeh
                                            if sizeh == current.sizeh:
                                                fusedNeighbours = neighbours_to_fuse[0]
                                                for neighbour in neighbours_to_fuse[1:]:
                                                    fusedNeighbours = fusedNeighbours.fuseLower(neighbour, self)
                                                    # now we have one big lower neighbour and we can fuse this one to our current:
                                                fusedObject = current.fuseLeft(fusedNeighbours, self)
                                                fusedSomething = True
                                            else:
                                                pass
                                                # print('WEIRD CASE OF NOT FUSING')
        # we fused something to prev if possible, so we are sure we have the entire object
        # fusing files that belong to the same obstacle
        for x in range(self.maxx + 1):
            # to reduce complexity we only fuse one at a time
            if fusedSomething == False:
                for y in range(self.maxy + 1):
                    # to reduce complexity we only fuse one at a time
                    if fusedSomething == False:
                        identity = (x, y)
                        current = self.subfiles[identity]
                        if current.allWhite == False:
                            if True:
                                if current.checkUpperBorder() == False:  # so there is some color at the upper border
                                    neighbours_to_fuse = []
                                    for neighbour in current.upper:
                                        # to fuse something on top, we must first fuse all upper neighbours
                                        neighbours_to_fuse.append(neighbour)
                                    # as a result of fusing sometimes you are your own neighbour
                                    neighbours_to_fuse = current.differentChunksList(neighbours_to_fuse, self)
                                    if len(neighbours_to_fuse) == 1:
                                        fusedObject = current.fuseUpper(neighbours_to_fuse[0], self)
                                        fusedSomething = True
                                    elif len(neighbours_to_fuse) > 1:
                                        # multiple neighbours -> check their compatability:
                                        compatible = True
                                        for neighbour in neighbours_to_fuse:
                                            if neighbour.sizeh != neighbours_to_fuse[0].sizeh:
                                                compatible = False
                                        if compatible == True:  # now we sort these neighbours from left to right
                                            neighbours_to_fuse = bubbleSort(neighbours_to_fuse, 'horizontal')
                                            sizew = 0
                                            for neighbour in neighbours_to_fuse:
                                                sizew += neighbour.sizew
                                            if sizew == current.sizew:
                                                fusedNeighbours = neighbours_to_fuse[0]
                                                for neighbour in neighbours_to_fuse[1:]:
                                                    fusedNeighbours = fusedNeighbours.fuseRight(neighbour, self)
                                                    # now we have one big upper neighbour and we can fuse this one to our current:
                                                fusedObject = current.fuseUpper(fusedNeighbours, self)
                                                fusedSomething = True
                                            else:
                                                pass
                                                #print('WEIRD CASE OF NOT FUSING')
                            # to reduce complexity we only fuse one at a time
                            if fusedSomething == False:
                                if current.checkLowerBorder() == False:  # so there is some color at the upper border
                                    neighbours_to_fuse = []
                                    for neighbour in current.lower:
                                        # to fuse something on top, we must first fuse all upper neighbours
                                        neighbours_to_fuse.append(neighbour)
                                    # as a result of fusing sometimes you are your own neighbour
                                    neighbours_to_fuse = current.differentChunksList(neighbours_to_fuse, self)
                                    if len(neighbours_to_fuse) == 1:
                                        fusedObject = current.fuseLower(neighbours_to_fuse[0], self)
                                        fusedSomething = True
                                    elif len(neighbours_to_fuse) > 1:
                                        # multiple neighbours -> check their compatability:
                                        compatible = True
                                        for neighbour in neighbours_to_fuse:
                                            if neighbour.sizeh != neighbours_to_fuse[0].sizeh:
                                                compatible = False
                                        if compatible == True:  # now we sort these neighbours from left to right
                                            neighbours_to_fuse = bubbleSort(neighbours_to_fuse, 'horizontal')
                                            sizew = 0
                                            for neighbour in neighbours_to_fuse:
                                                sizew += neighbour.sizew
                                            if sizew == current.sizew: #second compatability check
                                                fusedNeighbours = neighbours_to_fuse[0]
                                                for neighbour in neighbours_to_fuse[1:]:
                                                    fusedNeighbours = fusedNeighbours.fuseRight(neighbour, self)
                                                # now we have one big lower neighbour and we can fuse this one to our current:
                                                fusedObject = current.fuseLower(fusedNeighbours, self)
                                                fusedSomething = True
                                            else:
                                                pass
                            if fusedSomething == False:
                                if current.checkRightBorder() == False:  # so there is some color at the right border
                                    neighbours_to_fuse = []
                                    for neighbour in current.right:
                                        # to fuse something on top, we must first fuse all upper neighbours
                                        neighbours_to_fuse.append(neighbour)
                                    # as a result of fusing sometimes you are your own neighbour
                                    neighbours_to_fuse = current.differentChunksList(neighbours_to_fuse, self)
                                    if len(neighbours_to_fuse) == 1:
                                        fusedObject = current.fuseRight(neighbours_to_fuse[0], self)
                                        fusedSomething = True
                                    elif len(neighbours_to_fuse) > 1:
                                        # multiple neighbours -> check their compatability:
                                        compatible = True
                                        for neighbour in neighbours_to_fuse:
                                            if neighbour.sizew != neighbours_to_fuse[0].sizew:
                                                compatible = False
                                        if compatible == True:  # now we sort these neighbours from top to bottom
                                            neighbours_to_fuse = bubbleSort(neighbours_to_fuse, 'vertical')
                                            sizeh = 0
                                            for neighbour in neighbours_to_fuse:
                                                sizeh += neighbour.sizeh
                                            if sizeh == current.sizeh:

                                                fusedNeighbours = neighbours_to_fuse[0]
                                                for neighbour in neighbours_to_fuse[1:]:
                                                    fusedNeighbours = fusedNeighbours.fuseLower(neighbour, self)
                                                        # now we have one big lower neighbour and we can fuse this one to our current:
                                                fusedObject = current.fuseRight(fusedNeighbours, self)
                                                fusedSomething = True
                                            else:
                                                pass

                            if fusedSomething == False:
                                if current.checkLeftBorder() == False:  # so there is some color at the right border
                                    neighbours_to_fuse = []
                                    for neighbour in current.left:
                                        # to fuse something on top, we must first fuse all upper neighbours
                                        neighbours_to_fuse.append(neighbour)
                                    # as a result of fusing sometimes you are your own neighbour
                                    neighbours_to_fuse = current.differentChunksList(neighbours_to_fuse, self)
                                    if len(neighbours_to_fuse) == 1:
                                        fusedObject = current.fuseLeft(neighbours_to_fuse[0], self)
                                        fusedSomething = True
                                    elif len(neighbours_to_fuse) > 1:
                                        # multiple neighbours -> check their compatability:
                                        compatible = True
                                        for neighbour in neighbours_to_fuse:
                                            if neighbour.sizew != neighbours_to_fuse[0].sizew:
                                                compatible = False
                                        if compatible == True:  # now we sort these neighbours from top to bottom
                                            neighbours_to_fuse = bubbleSort(neighbours_to_fuse, 'vertical')
                                            sizeh = 0
                                            for neighbour in neighbours_to_fuse:
                                                sizeh += neighbour.sizeh
                                            if sizeh == current.sizeh:
                                                fusedNeighbours = neighbours_to_fuse[0]
                                                for neighbour in neighbours_to_fuse[1:]:
                                                    fusedNeighbours = fusedNeighbours.fuseLower(neighbour, self)
                                                    # now we have one big lower neighbour and we can fuse this one to our current:
                                                fusedObject = current.fuseLeft(fusedNeighbours, self)
                                                fusedSomething = True
                                            else:
                                                pass

        return fusedSomething, fusedObject


    def fuseAllBorders(self):
        """
        fuse all the chunks into objects, until there are no more non white borders
        """
        fusedsomething = True
        fusedObject = None
        while fusedsomething:
            fusedsomething, fusedObject = self.fuseColoredBorder(fusedObject)
            if self.settings.DISPLAYFUSING and fusedObject != None:
                print('upper border is white: '+str(fusedObject.checkUpperBorder()))
                print('right border is white: '+str(fusedObject.checkRightBorder()))
                print('lower border is white: '+str(fusedObject.checkLowerBorder()))
                print('left border is white: '+str(fusedObject.checkLeftBorder()))
                fusedObject.display()
        objects = set()
        for key in self.subfiles.keys():
            if not self.subfiles[key].allWhite:
                objects.add(self.subfiles[key])
        self.obstaclesPng = objects
    
    def displayAllISolatedObjects(self):
        """
        display the result of fusing
        """
        i = 0
        removeLater = []
        for obstaclePng in self.obstaclesPng:
            plt.close()
            plt.imshow(obstaclePng.data, cmap='gray')
            filePath = str(pathlib.Path().resolve()) + '/AFTERISOLATION/' + 'isolatedObject'+ str(i)+'.png'
            plt.title("isolated obstacle "+str(i))
            plt.savefig(filePath)
            plt.xlabel('y [pixels]')
            plt.ylabel('x [pixels]')
            if self.settings.SHOWSEPERATEOBJECTS:
                if self.settings.CLEAR:
                    # clear prints
                    # https://stackoverflow.com/questions/19596750/is-there-a-way-to-clear-your-printed-text-in-python
                    os.system('cls' if os.name == 'nt' else "printf '\033c'")
                    # clear notebook prints
                    clear_output(wait=True)
                print('Displaying obstacle: ' + str(i))
                plt.show()
                if self.settings.PROMPTPICS:
                    keep = input('is this an obstacle? (y/n): ')
                    if keep == 'n':
                        removeLater.append(obstaclePng)
                        print('obstacle removed')
                if self.settings.CLEAR:
                    time.sleep(1)
                    # clear prints
                    # https://stackoverflow.com/questions/19596750/is-there-a-way-to-clear-your-printed-text-in-python
                    os.system('cls' if os.name == 'nt' else "printf '\033c'")
                    # clear notebook prints
                    clear_output(wait=True)
            plt.close()
            i+=1

        for obstaclePng in removeLater:
            self.obstaclesPng.remove(obstaclePng)
    ################################################################################################################
    ################### FOR EACH subPICTURE WE ANALYSE THE OBJECT AND PUT THE WHOLE THING BACK TOGETHER ############
    ################################################################################################################

    def displayObstacles(self):
        """
        display the result of fusing
        """
        for obstacle in self.obstaclesPng:
            obstacle.display()
            if self.settings.PROMPTPICS:
                keep = input('is this an obstacle? (1/0)')
                if not keep:
                    self.obstaclesPng.remove(obstacle)

    def saveFig(self, i, obstaclePng):
        """
        save a figure, i: name of the obstacle, obstaclePng: grayscale np.array
        """
        filePath = str(pathlib.Path().resolve()) + '/TEMP/' + str(i) + '.png'
        im = Image.fromarray(obstaclePng.data).convert('RGB')
        im.save(filePath)
        return filePath

    def convertObstaclesToGlobal(self, obstacleMap = None, obstaclePng = None):
        """
        all obstacles retrieved from an obstacle Png are defined in their local coordinate system
        -> convert them to a global system
        """
        if obstacleMap == None:
            obstacles = self.globalObstacles
        else:
            obstacles = obstacleMap.obstacles
        addLater = []
        removeLater = []
        for obstacle in obstacles:
            localCorners = obstacle.corners
            localEdges = obstacle.edges
            if obstaclePng == None:
                globaloffsetx = -1*self.offsetx
                globaloffsety = -1*self.offsety
            else:
                globaloffsetx = obstaclePng.xmin
                globaloffsety = obstaclePng.ymin
            globalCorners = set()
            for corner in localCorners:
                localx = corner.x
                localy = corner.y
                globalCorners.add(Points.Point(localx + globaloffsetx, localy + globaloffsety))

            globalEdges = set()
            for edge in localEdges:
                corners = edge.points
                globalCorners = []
                for corner in corners:
                    globalCorners.append(Points.Point(corner.x + globaloffsetx, corner.y + globaloffsety))
                globalEdges.add(Edges.Edge(globalCorners[0], globalCorners[1]))
            new_obstacle = Obstacles.Obstacle(globalCorners, globalEdges)

            if obstacle in self.globalObstacles:
                removeLater.append(obstacle)
            addLater.append(new_obstacle)
        for old_obstacle in removeLater:
            self.globalObstacles.remove(old_obstacle)
        for new_obstacle in addLater:
            self.globalObstacles.add(new_obstacle)

    def exportGlobalObstaclesFlippedXY(self):
        """
        in the other parts (navigation ect),
        obstacles are referred to in a flipped x & y system,
        origin in the top left corner
        the outputted obstacles are defined in the form:
        [[first corner of obstacle 1, second corner of obstacle 1, ..],
        [first corner of obstacle 2, second corner of obstacle 2, ..],
        ...]
        with corners as 'vg.point()'
        """
        polys = []
        for obstacle in self.globalObstacles:
            # The edges and points within these edges are sets, 
            # we have to convert them into lists to be able to order the edges & points
            def setToList(someSet):
                elements = []
                for el in someSet:
                    elements.append(el)
                return elements

            edgesList = setToList(obstacle.edges)
            firstEdge = edgesList[0]
            edgesList = edgesList[1:]

            firstEdgePoints = setToList(firstEdge.points)
            orderedPoints = [firstEdgePoints]
            i = 0
            while edgesList != []:
                #assert i != len(edgesList)
                if i == len(edgesList): # the assert makes more sense, but lets not crash it on purpose
                    break
                else:
                    currentPoints = setToList(edgesList[i].points)
                    if currentPoints[1].x == orderedPoints[0][0].x and currentPoints[1].y == orderedPoints[0][0].y:
                        orderedPoints.insert(0,currentPoints)
                        edgesList.remove(edgesList[i])
                        i = 0
                    elif currentPoints[0].x == orderedPoints[len(orderedPoints)-1][1].x and currentPoints[0].y == orderedPoints[len(orderedPoints)-1][1].y:
                        orderedPoints.append(currentPoints)
                        edgesList.remove(edgesList[i])
                        i = 0
                    elif currentPoints[0].x == orderedPoints[0][0].x and currentPoints[0].y == orderedPoints[0][0].y:
                        currentPoints = [currentPoints[1], currentPoints[0]]
                        orderedPoints.insert(0,currentPoints)
                        edgesList.remove(edgesList[i])
                        i = 0
                    elif currentPoints[1].x == orderedPoints[len(orderedPoints)-1][1].x and currentPoints[1].y == orderedPoints[len(orderedPoints)-1][1].y:
                        currentPoints = [currentPoints[1], currentPoints[0]]
                        orderedPoints.append(currentPoints)
                        edgesList.remove(edgesList[i])
                        i = 0
                    else:
                        i+=1

            offsetx = -self.offsetx
            offsety = -self.offsety
            globalCorners = []
            for Points in orderedPoints:
                corner = Points[0]
                # remove border and flip x & y for each corner
                x = corner.y + offsety
                y = corner.x + offsetx
                globalCorners.append(vg.Point(x,y))
            polys.append(globalCorners)
        return polys

    def displayGlobalObstacles(self):
        """
        display all the obstacles in one picture
        """
        def addobstacles(ax, obstacles):
            print('there are ' + str(len(obstacles)) + ' obstacles')
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
        plt.imshow(self.originaldata, cmap='gray')
        plt.xlabel('y [pixels]')
        plt.ylabel('x [pixels]')
        plt.title("Original picture")
        fig, ax = plt.subplots()
        ax.imshow(self.data, cmap='gray')
        ax.autoscale(False)
        addobstacles(ax, self.globalObstacles)
        filePath = str(pathlib.Path().resolve()) + '/TEMP/' + 'FINALRESULT.png'
        plt.title("Final result")
        plt.xlabel('y [pixels]')
        plt.ylabel('x [pixels]')
        plt.savefig(filePath)
        if self.settings.CLEAR:
            # clear prints
            # https://stackoverflow.com/questions/19596750/is-there-a-way-to-clear-your-printed-text-in-python
            os.system('cls' if os.name == 'nt' else "printf '\033c'")
            # clear notebook prints
            clear_output(wait=True)
        if self.settings.SHOWENDRESULT:
            plt.show()
        plt.close()

    def partialSimplify(self, output, criterion):
        """
        some lines cut an edge of an obstacle into two, causing unnecessairy intersections like the x in this top line
                x--x--x
               /     /
              /     /
             x_____x
        these cause the package for visibility graphs to crash
        ( they use the cosine rule and end up dividing by zero as theta = 180 )
        so, we filter them out
        in: array with obstacles in visgraph format
        out: array with simplified obstacles in visgraph format
        this algorithm fails to simplify the first three points, so it has to be ran three times with different
        points at index 0
        aslo remove points that are really close to each other
        """
        n = 1
        outputcopy = []
        deletedSomething = False
        for obstacle in output:
            i = 0
            simplified_corners_of_object = []
            # y = ax + b
            abHistory = []
            for corner in obstacle:
                if i == 0:
                    simplified_corners_of_object.append(corner)
                    abHistory.append((None, None))
                elif i == 1:

                    prev = simplified_corners_of_object[len(simplified_corners_of_object)-1]
                    if round(corner.x, n) != round(prev.x, n):
                        a = (corner.y-prev.y)/(corner.x-prev.x)
                        b = corner.y-a*corner.x
                    else:
                        a = None
                        b = corner.x
                    abHistory.append((a,b))
                    if criterion == 'location':
                        orderx = 0
                        while abs(corner.x) > 10 ** orderx:
                            orderx += 1
                        ordery = 0
                        while abs(corner.y) > 10 ** ordery:
                            ordery += 1
                        orderx -= 1
                        ordery -= 1
                        if abs(round((prev.x - corner.x) / 10 ** orderx, n)) <= 0.1 and abs(
                                round((prev.y - corner.y) / 10 ** ordery, n)) <= 0.1:
                            simplified_corners_of_object.remove(prev)
                            abHistory.remove((a,b))
                            deletedSomething = True
                    simplified_corners_of_object.append(corner)
                else:
                    if criterion == 'location':
                        orderx = 0
                        while abs(corner.x) > 10 ** orderx:
                            orderx += 1
                        ordery = 0
                        while abs(corner.y) > 10 ** ordery:
                            ordery += 1
                        orderx -= 1
                        ordery -= 1
                    prev = simplified_corners_of_object[len(simplified_corners_of_object)-1]
                    assert prev != corner
                    if corner.x != prev.x:
                        a = (corner.y-prev.y)/(corner.x-prev.x)
                        b = corner.y-a*corner.x
                    else:
                        a = None
                        b = corner.x
                    preva, prevb = abHistory[len(abHistory)-1]
                    abHistory.append((a,b))
                    if criterion == 'location':
                        if abs(round((prev.x - corner.x) / 10 ** orderx, n)) <= 0.1 and abs(
                                round((prev.y - corner.y) / 10 ** ordery, n)) <= 0.1:
                            simplified_corners_of_object.remove(prev)
                            abHistory.remove((a, b))
                            deletedSomething = True

                    elif a != None and preva != None:
                        ordera = 0
                        while abs(a) > 10 ** ordera:
                            ordera += 1
                        orderb = 0
                        while abs(b) > 10 ** orderb:
                            orderb += 1
                        ordera -= 1
                        orderb -= 1
                        if round(a/(10**ordera),n) == round(preva/(10**ordera),n) and round(b/(10**orderb),n) == round(prevb/(10**orderb),n):
                            simplified_corners_of_object.remove(prev)
                            deletedSomething = True
                            pass
                        else:
                            pass
                    elif a == None and preva == None:
                        orderb = 0
                        while abs(b) > 10 ** orderb:
                            orderb += 1
                        orderb -= 1
                        if round(b/(10**orderb),n) == round(prevb/(10**orderb),n):
                            simplified_corners_of_object.remove(prev)
                            deletedSomething = True
                            pass
                        else:
                            pass
                    else: # one a is 'none' -> not the same
                        pass
                    simplified_corners_of_object.append(corner)
                i+=1
            outputcopy.append(simplified_corners_of_object)
        # if only a line is left: remove it
        for obstacle in outputcopy:
            if len(obstacle) <= 2:
                outputcopy.remove(obstacle)
        return outputcopy, deletedSomething

    def simplify(self,output):
        """
        run the algorithm above three times with different corners at index 0
            in: array with obstacles in visgraph format
            out: array with simplified obstacles in visgraph format
        """
        deletedSomethingOuterLoop = True
        outputcopy, bool = self.partialSimplify(output, 'slope')
        while deletedSomethingOuterLoop: # I know this is inefficient but on a small scale it goes fast enough
            deletedSomethingOuterLoop = False
            deletedSomething = True
            while deletedSomething: # I know this is inefficient but on a small scale it goes fast enough
                deletedSomething = False
                outputcopy, bool = self.partialSimplify(outputcopy, 'slope')
                if bool:
                    deletedSomething = True
                    deletedSomethingOuterLoop = True
                copycopy = []
                for obstacle in outputcopy: #change which point comes first
                    obstacle2 = obstacle[len(obstacle)-1:]+obstacle[:len(obstacle)-1]
                    copycopy.append(obstacle2)
                outputcopy, bool = self.partialSimplify(copycopy, 'slope')
                if bool:
                    deletedSomething = True
                    deletedSomethingOuterLoop = True
                copycopy = []
                for obstacle in outputcopy: #change which point comes first, again
                    obstacle2 = obstacle[len(obstacle)-1:]+obstacle[:len(obstacle)-1]
                    copycopy.append(obstacle2)
                outputcopy, bool = self.partialSimplify(copycopy, 'slope')
                if bool:
                    deletedSomething = True
                    deletedSomethingOuterLoop = True
            deletedSomething = True
            while deletedSomething: # I know this is very inefficient but on a small scale it goes fast
                deletedSomething = False
                outputcopy, bool = self.partialSimplify(outputcopy, 'location')
                if bool:
                    deletedSomething = True
                    deletedSomethingOuterLoop = True
                copycopy = []
                for obstacle in outputcopy: #change which point comes first
                    obstacle2 = obstacle[len(obstacle)-1:]+obstacle[:len(obstacle)-1]
                    copycopy.append(obstacle2)
                outputcopy, bool = self.partialSimplify(copycopy, 'location')
                if bool:
                    deletedSomething = True
                    deletedSomethingOuterLoop = True
                copycopy = []
                for obstacle in outputcopy: #change which point comes first, again
                    obstacle2 = obstacle[len(obstacle)-1:]+obstacle[:len(obstacle)-1]
                    copycopy.append(obstacle2)
                outputcopy, bool = self.partialSimplify(copycopy, 'location')
                if bool:
                    deletedSomething = True
                    deletedSomethingOuterLoop = True
        return outputcopy

    def displaySimplified(self, polys, simplifiedpolys):
        plt.close()
        plt.figure(1)
        #plt.imshow(self.originaldata, cmap='gray')
        for poly in simplifiedpolys:
            for point in poly:
                plt.plot(point.x, point.y, 'go', markersize=10)
        for poly in polys:
            pol_x = []
            pol_y = []
            for point in poly:
                pol_x.append(point.x)
                pol_y.append(point.y)
                plt.plot(point.x,point.y, 'r*', markersize=8)
            pol_x.append(pol_x[0])
            pol_y.append(pol_y[0])
            pol_x = np.asarray(pol_x)
            pol_y = np.asarray(pol_y)
            plt.plot(pol_x, pol_y)
        plt.plot([], [], 'go', markersize=10, label='remaining points')
        plt.plot([],[], 'r*', markersize=8, label='original points')
        plt.title("Simplified result")
        plt.legend()
        plt.xlabel('y [pixels]')
        plt.ylabel('x [pixels]')
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()
    def determineThymioSize(self):
        if self.settings.PROMPTTHYMIOWIDTH:
            yn = input('Do you want to use the default thymio width? [y/n]: ')
            if yn == 'y':
                return self.settings.STANDARDTHYMIOWIDTH
            else:
                plt.close()
                plt.imshow(self.originaldata, cmap='gray')
                plt.title('What is the width of the thymio? (Pixels)')
                plt.show()
                plt.close()
                return int(input('What is the width of the thymio? [pixels]: '))
        else:
            return self.settings.STANDARDTHYMIOWIDTH

