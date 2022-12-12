import time
import matplotlib.pyplot as plt
import cv_settings as settings
import numpy
class SubFile:
    """
    a subfile is a part of the initial file with its own local x and y axis. We try to fuse all the subfiles that
    contain the same object, after having split up the whole picture so we can analyse each object seperately and put
    the objects back together afterwards
    """
    def __init__(self, data, xmin, ymin, dim_x, dim_y, identitys, allWhite, sizeh = 1, sizew = 1):
        self.shape = [dim_x, dim_y]
        self.data = data
        self.xmin = xmin
        self.ymin = ymin
        self.identity = set()
        for identity in identitys:
            self.identity.add(identity)
        self.allWhite = allWhite
        self.right = set()
        self.left = set()
        self.upper = set()
        self.lower = set()
        self.sizeh = sizeh
        self.sizew = sizew

    ################################################################################################################
    ################### ADDING NEIGHBOURS ##########################################################
    ################################################################################################################
    def addLeft(self, left):
        self.left.add(left)
    def addRight(self, right):
        self.right.add(right)
    def addUpper(self, upper):
        self.upper.add(upper)
    def addLower(self, lower):
        self.lower.add(lower)

    def display(self, data = None):
        """
        display a chunk (self)
        """
        if data == None:
            data = self.data
        print('DISPLAYING OBSTACLE:')
        print(self.identity)
        plt.imshow(data, cmap='gray')
        plt.show()
        time.sleep(0.1)
        plt.close()

    ################################################################################################################
    ################### FUSING subFiles ##########################################################
    ################################################################################################################

    def getBorder(self, identity): # a border is 10 pixels wide btw so we avoid only having one pixel wiggle room
        """
        return the data array with data of the border of a chunk
        """
        new_data = []
        for i in range(settings.BORDER):
            if identity == 'upper': #'up' -> analyse xdatas
                xdata = self.data[i]
                for data in xdata:
                    new_data.append(data)
            elif identity == 'lower':
                xdata = self.data[self.shape[0]-i-1]
                for data in xdata:
                    new_data.append(data)
            elif identity == 'left': #'left' -> analyse leftmost of each xdatas
                #new_data = []
                for xdata in self.data:
                    new_data.append(xdata[i])
            elif identity == 'right':
                #new_data = []
                for xdata in self.data:
                    new_data.append(xdata[self.shape[1]-i-1])
            else:
                print('invalid identity in subFile getBorder')
                assert(0 == 1)
        return new_data
    def checkLowerBorder(self):
        """
        check the lower border of chunk self for non white pixels
        """
        if self.allWhite == False:
            new_data = self.getBorder('lower')
            allWhiteBorder = True
            for data in new_data:
                if data <= settings.GRIDMAPBLACKTRESHOLD:
                    allWhiteBorder = False
            return allWhiteBorder
        else:
            return True

    def checkUpperBorder(self):
        """
        check the upper border of chunk self for non white pixels
        """
        if self.allWhite == False:
            new_data = self.getBorder('upper')
            allWhiteBorder = True
            for data in new_data:
                if data <= settings.GRIDMAPBLACKTRESHOLD:
                    allWhiteBorder = False
            return allWhiteBorder
        else:
            return True

    def checkLeftBorder(self):
        """
        check the left border of chunk self for non white pixels
        """
        if self.allWhite == False:
            new_data = self.getBorder('left')
            allWhiteBorder = True
            for data in new_data:
                if data <= settings.GRIDMAPBLACKTRESHOLD:
                    allWhiteBorder = False
            return allWhiteBorder
        else:
            return True

    def checkRightBorder(self):
        """
        check the right border of chunk self for non white pixels
        """
        if self.allWhite == False:
            new_data = self.getBorder('right')
            allWhiteBorder = True
            for data in new_data:
                if data <= settings.GRIDMAPBLACKTRESHOLD:
                    allWhiteBorder = False
            return allWhiteBorder
        else:
            return True

    def fuseUpper(self, upper, gridmap):
        """
        fuse a chunk with its upper neighbour of the same width
        """
        assert self.shape[1] == upper.shape[1]  # compatability check: same width
        new_data = numpy.ndarray((self.shape[0]+upper.shape[0], self.shape[1]))
        i = 0
        for xline in upper.data:
            new_data[i] = xline
            i+= 1
        for xline in self.data:
            new_data[i] = xline
            i+=1
        identity = set()
        for old_identity in self.identity:
            identity.add(old_identity)
        for old_identity in upper.identity:
            identity.add(old_identity)
        assert self.xmin > upper.xmin
        assert self.ymin == upper.ymin
        xmin = upper.xmin
        ymin = upper.ymin
        dim_y = self.shape[1]#+upper.shape[1]
        dim_x = self.shape[0] +upper.shape[0]
        sizew = self.sizew
        sizeh = self.sizeh + upper.sizeh
        assert dim_y <= sizew * settings.GRIDSIZE < dim_y + settings.GRIDSIZE
        assert dim_x <= sizeh * settings.GRIDSIZE < dim_x + settings.GRIDSIZE
        fusedFile = SubFile(new_data, xmin, ymin, dim_x, dim_y, identity, False, sizew=sizew, sizeh=sizeh)
        # add neighbours:
        for identity in fusedFile.identity:
            for neighbour in gridmap.subfiles[identity].right:
                neighbour.addLeft(fusedFile)
                fusedFile.addRight(neighbour)
            for neighbour in gridmap.subfiles[identity].left:
                neighbour.addRight(fusedFile)
                fusedFile.addLeft(neighbour)
            for neighbour in gridmap.subfiles[identity].upper:
                neighbour.addLower(fusedFile)
                fusedFile.addUpper(neighbour)
            for neighbour in gridmap.subfiles[identity].lower:
                neighbour.addUpper(fusedFile)
                fusedFile.addLower(neighbour)
            gridmap.subfiles[identity] = fusedFile
        # now there will be multiple neighbours on an edge and some neighbours are 'fusedFile' but thats okay
        return fusedFile

    def fuseLower(self, lower, gridmap):
        """
        fuse a chunk with its lower neighbour of the same width
        """
        assert self.shape[1] == lower.shape[1] #compatability check: same width
        new_data = numpy.ndarray((self.shape[0]+lower.shape[0], self.shape[1]))
        i = 0
        for xline in self.data:
            new_data[i] = xline
            i += 1
        for xline in lower.data:
            new_data[i] = xline
            i += 1
        identity = set()
        for old_identity in self.identity:
            identity.add(old_identity)
        for old_identity in lower.identity:
            identity.add(old_identity)
        assert self.xmin < lower.xmin
        assert self.ymin == lower.ymin
        xmin = self.xmin
        ymin = self.ymin
        dim_y = self.shape[1]#+lower.shape[1]
        dim_x = self.shape[0] +lower.shape[0]
        sizew = self.sizew
        sizeh = self.sizeh + lower.sizeh
        assert dim_y <= sizew * settings.GRIDSIZE < dim_y + settings.GRIDSIZE
        assert dim_x <= sizeh * settings.GRIDSIZE < dim_x + settings.GRIDSIZE
        fusedFile = SubFile(new_data, xmin, ymin, dim_x, dim_y, identity, False, sizew = sizew, sizeh = sizeh)
        # add neighbours:
        for identity in fusedFile.identity:
            for neighbour in gridmap.subfiles[identity].right:
                neighbour.addLeft(fusedFile)
                fusedFile.addRight(neighbour)
            for neighbour in gridmap.subfiles[identity].left:
                neighbour.addRight(fusedFile)
                fusedFile.addLeft(neighbour)
            for neighbour in gridmap.subfiles[identity].upper:
                neighbour.addLower(fusedFile)
                fusedFile.addUpper(neighbour)
            for neighbour in gridmap.subfiles[identity].lower:
                neighbour.addUpper(fusedFile)
                fusedFile.addLower(neighbour)
            gridmap.subfiles[identity] = fusedFile
        # now there will be multiple neighbours on an edge and some neighbours are 'fusedFile' but thats okay
        return fusedFile

    def fuseRight(self, right, gridmap): #to the right is in the y direction
        """
        fuse a chunk with its right neighbour of the same width
        """
        # shape: x, y = height, width = up, right
        assert self.shape[0] == right.shape[0]  # compatability check: same height
        new_data = numpy.ndarray((self.shape[0], self.shape[1]+right.shape[1]))
        j = 0
        for i in range(len(self.data)):
            xline = numpy.ndarray((1,self.shape[1]+right.shape[1]))
            m = 0
            for data in self.data[i]:
                xline[0, m] = data
                m+=1
            for data in right.data[i]:
                xline[0, m] = data
                m+=1
            new_data[j] = xline
            j += 1
        identity = set()
        for old_identity in self.identity:
            identity.add(old_identity)
        for old_identity in right.identity:
            identity.add(old_identity)
        assert self.xmin == right.xmin
        assert self.ymin < right.ymin
        xmin = self.xmin
        ymin = self.ymin
        dim_y = self.shape[1] + right.shape[1]
        dim_x = self.shape[0]  #+right.shape[0]
        sizew = self.sizew + right.sizew
        sizeh = self.sizeh
        assert dim_y <= sizew * settings.GRIDSIZE < dim_y + settings.GRIDSIZE
        assert dim_x <= sizeh * settings.GRIDSIZE < dim_x + settings.GRIDSIZE
        fusedFile = SubFile(new_data, xmin, ymin, dim_x, dim_y, identity, False, sizew=sizew, sizeh=sizeh)
        # add neighbours:
        for identity in fusedFile.identity:
            this_chunk = gridmap.subfiles[identity]
            rightneighbours = this_chunk.right
            for neighbour in rightneighbours:
                neighbour.addLeft(fusedFile)
                fusedFile.addRight(neighbour)
            for neighbour in gridmap.subfiles[identity].left:
                neighbour.addRight(fusedFile)
                fusedFile.addLeft(neighbour)
            for neighbour in gridmap.subfiles[identity].upper:
                neighbour.addLower(fusedFile)
                fusedFile.addUpper(neighbour)
            for neighbour in gridmap.subfiles[identity].lower:
                neighbour.addUpper(fusedFile)
                fusedFile.addLower(neighbour)
            gridmap.subfiles[identity] = fusedFile
        # now there will be multiple neighbours on an edge and some neighbours are 'fusedFile' but thats okay
        return fusedFile

    def fuseLeft(self, left, gridmap):
        """
        fuse a chunk with its left neighbour of the same width
        """
        assert self.shape[0] == left.shape[0]  # compatability check: same width
        new_data = numpy.ndarray((self.shape[0], self.shape[1] + left.shape[1]))
        j = 0
        for i in range(len(self.data)):
            xline = numpy.ndarray((1, self.shape[1] + left.shape[1]))
            m = 0
            for data in left.data[i]:
                xline[0, m] = data
                m += 1
            for data in self.data[i]:
                xline[0, m] = data
                m += 1
            new_data[j] = xline
            j += 1
        identity = set()
        for old_identity in self.identity:
            identity.add(old_identity)
        for old_identity in left.identity:
            identity.add(old_identity)
        assert self.xmin == left.xmin
        assert self.ymin > left.ymin
        xmin = left.xmin
        ymin = left.ymin
        dim_y = self.shape[1] + left.shape[1]
        dim_x = self.shape[0]
        sizew = self.sizew + left.sizew
        sizeh = self.sizeh
        assert dim_y <= sizew * settings.GRIDSIZE < dim_y + settings.GRIDSIZE
        assert dim_x <= sizeh * settings.GRIDSIZE < dim_x + settings.GRIDSIZE
        fusedFile = SubFile(new_data, xmin, ymin, dim_x, dim_y, identity, False, sizew=sizew, sizeh=sizeh)
        #add neighbours:
        for identity in fusedFile.identity:
            for neighbour in gridmap.subfiles[identity].right:
                neighbour.addLeft(fusedFile)
                fusedFile.addRight(neighbour)
            for neighbour in gridmap.subfiles[identity].left:
                neighbour.addRight(fusedFile)
                fusedFile.addLeft(neighbour)
            for neighbour in gridmap.subfiles[identity].upper:
                neighbour.addLower(fusedFile)
                fusedFile.addUpper(neighbour)
            for neighbour in gridmap.subfiles[identity].lower:
                neighbour.addUpper(fusedFile)
                fusedFile.addLower(neighbour)
            gridmap.subfiles[identity] = fusedFile
        # now there will be multiple neighbours on an edge and some neighbours are 'fusedFile' but thats okay
        return fusedFile

    ################################################################################################################
    ################### YOU ARE YOUR OWN NEIGHBOR AFTER FUSING => FILTER ###########################################
    ################################################################################################################

    def differentChunksSet(self, chunkSet, gridmap):
        """
        in a set of chunks, some chunks might be a subchunk of self
        -> return a set of chunks that dont refer to (a part of) self
        """
        #only keep chunks that dont point to self
        differentchunks = set()
        for chunk in chunkSet:
            # either all all or none of the id's should point to self btw
            counter = 0
            for chunkid in chunk.identity:
                if gridmap.subfiles[chunkid] == self:
                    counter += 1
            assert counter == len(chunk.identity) or counter == 0
            if counter == 0:
                differentchunks.add(chunk)
        return differentchunks

    def differentChunksList(self, chunkSet, gridmap):
        """
        in a set of chunks, some chunks might be a subchunk of self
        -> return a list of chunks that dont refer to (a part of) self
        """
        #only keep chunks that dont point to self
        differentchunks = list()
        for chunk in chunkSet:
            # either all all or none of the id's should point to self btw
            counter = 0
            for chunkid in chunk.identity:
                if gridmap.subfiles[chunkid] == self:
                    counter += 1
            assert counter == len(chunk.identity) or counter == 0
            if counter == 0:
                differentchunks.append(chunk)
        return differentchunks
