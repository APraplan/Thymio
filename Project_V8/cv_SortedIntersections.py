import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


class SortedIntersections:
    """
    Intersections so far have no way of knowing who their neighbours are, so we make a dictionary:
    line -> all intersections on this line sorted on x basis,
     - when a line is horizontal: sorted on y basis -
    we use this dictionary to get the intersections on the left and right of an intersection,
    following a given line.
    This is the backbone of our graph
    """
    #groups intersections into a dictionary
    def __init__(self, intersections):
        # 'intersections' should only contain those intersections that are corners, as these are the only relevant ones
        # we will make a dictionnary to get all the intersections of a certain line sorted from left to right
        # dictionarry: key = line.identity --> value = list(intersections)
        self.intersections_per_line = dict() # line -> list of intersections
        self.line_from_id = dict() # id -> line
        horizontalKeys = set() # set with keys of horizontal lines
        # add all lines from all intersections to line_from_id
        for intersection in intersections:
            if intersection.lineh.type == 'horizontal':
                horizontalKeys.add((intersection.lineh.identity, intersection.coordinate.x))

            if intersection.line1.identity in self.line_from_id.keys():
                #print('key: '+intersection.line1.identity+' --> '+str(self.line_from_id[intersection.line1.identity]) +' with id '+self.line_from_id[intersection.line1.identity].identity)
                assert self.line_from_id[intersection.line1.identity] == intersection.line1 #see if somehow line id got stolen -> somehow the program breaks here
            else:
                self.line_from_id[intersection.line1.identity] = intersection.line1
                assert self.line_from_id[intersection.line1.identity] == intersection.line1 # if this fails thats just weird
                                                            # there are multiple lines with the same identity??? -> figure this out
                                                            # fix: changed the names of lines to A, B, C, ... rather than indices
            if intersection.line2.identity in self.line_from_id.keys():
                assert self.line_from_id[intersection.line2.identity] == intersection.line2
            else:
                self.line_from_id[intersection.line2.identity] = intersection.line2
        # add all intersections to intersections_per_line
        for intersection in intersections:
            line1 = intersection.line1
            i = line1.identity
            line2 = intersection.line2
            j = line2.identity
            if i in self.intersections_per_line.keys():
                intersections_of_i = self.intersections_per_line[i]
            else:
                intersections_of_i = []
            intersections_of_i.append(intersection)
            self.intersections_per_line[i] = intersections_of_i
            if j in self.intersections_per_line.keys():
                intersections_of_j = self.intersections_per_line[j]
            else:
                intersections_of_j = []
            intersections_of_j.append(intersection)
            self.intersections_per_line[j] = intersections_of_j

        # assert linel is never horizontal and lineh.m > linel.m
        for lineID in self.intersections_per_line.keys():
            intersections_crossing_lineID = self.intersections_per_line[lineID]
            for intersection in intersections_crossing_lineID:
                assert intersection.linel.type != 'horizontal'
                if intersection.lineh.type != 'horizontal':
                    assert intersection.lineh.m > intersection.linel.m

        # sort the intersections of each line
        for lineID in self.intersections_per_line.keys():
            item = self.intersections_per_line[lineID]
            for intersection in item:
                assert intersection.line1.identity == lineID or intersection.line2.identity == lineID
            # if it is a horizontal line, a is undefined and they all have the same x value. For this case we sort from bottom to top
            # the keys are intersections
            # the horizontal line should be lineh
            currentLine = self.line_from_id[lineID]
            if currentLine.type == 'horizontal': #horizontal: x is a constant -> sort on basis of y!!!
                def sortkeyy(elem):
                    return elem.coordinate.y
                item.sort(key=sortkeyy)
                self.intersections_per_line[lineID] = item

            else: #ie they're both not horizontal and can be sorted from left to right = low x to high x
                def sortkeyx(elem):
                    return elem.coordinate.x
                item.sort(key=sortkeyx)
                self.intersections_per_line[lineID] = item
        # sanity check:
        # FOR EVERY LINE, WE FOLLOW ONE SAME LINE
        for lineID in self.intersections_per_line.keys():
            for intersection in self.intersections_per_line[lineID]:
                assert intersection.lineh.identity == lineID or intersection.linel.identity == lineID
        # sanity check:
        # FOR EVERY LINE, it is sorted
        for lineID in self.intersections_per_line.keys():
            one_intersection = self.intersections_per_line[lineID][0]
            if one_intersection.lineh.identity == lineID:
                lineType = one_intersection.lineh.type
            elif one_intersection.linel.identity == lineID:
                lineType = one_intersection.lineh.type
            else:
                print('sortedintersections err 0')
                assert(0==1)

            prevx = 0
            prevy = 0
            # FOR EVERY INTERSECTION ON A LINE
            for intersection in self.intersections_per_line[lineID]:
                #THE X COORDINATE IS HIGHER THAN THE PREVIOUS
                if intersection.coordinate.x < prevx:
                    #print(intersection.coordinate.x)
                    #print(intersection.coordinate.y)
                    #print(intersection.lineh.type)
                    #print(intersection.linel.type)
                    assert intersection.coordinate.x >= prevx
                if intersection.coordinate.x == prevx:
                    if not intersection.coordinate.y > prevy:
                        #print(intersection.coordinate.x)
                        #print(intersection.coordinate.y)
                        #print(intersection.lineh.type)
                        #print(intersection.linel.type)
                        assert intersection.coordinate.y >= prevy
                # print(str(intersection.coordinate.x)+'>'+str(prev))
                if intersection.coordinate.x >= prevx:
                    prevx = intersection.coordinate.x
                    prevy = intersection.coordinate.y

    ########################################################################################################################
    # get neighbouring intersections along a given line
    ########################################################################################################################

    def get_left_intersection(self, line, intersection): #can also be used to get lower in case of horizontal
        """
        return the intersection on the left, following a line, starting drom an intersection
        """
        if line == intersection.line1:
            key = intersection.identity[0]  # i dus
        elif line == intersection.line2:
            key = intersection.identity[1]
        else:
            print('error SortedIntersections 1')
            quit()
            # shouldnt be possible
        assert key == line.identity
        intersections_along_this_line = self.intersections_per_line[key]
        current_index = intersections_along_this_line.index(intersection)
        assert self.intersections_per_line[line.identity][
                   self.intersections_per_line[line.identity].index(intersection)] == intersection
        left_index = current_index - 1
        if left_index < 0:
            left_intersection = None  # this can happen
        else:
            left_intersection = intersections_along_this_line[left_index]
        return left_intersection

    def get_right_intersection(self, line, intersection): #can also be used to get upper in case of vertical
        """
                return the intersection on the right, following a line, starting drom an intersection
        """
        if line == intersection.line1:
            key = intersection.identity[0]  # i dus
        elif line == intersection.line2:
            key = intersection.identity[1]
        else:
            pass
            print('error SortedIntersections 2')
            # shouldnt be possible

        assert key == line.identity
        intersections_along_this_line = self.intersections_per_line[key]
        assert self.intersections_per_line[line.identity][
                   self.intersections_per_line[line.identity].index(intersection)] == intersection
        current_index = intersections_along_this_line.index(intersection)
        right_index = current_index + 1
        if right_index >= len(intersections_along_this_line):
            right_intersection = None  # this could happen
            # print('error SortedIntersections 3')
        else:
            right_intersection = intersections_along_this_line[right_index]
        return right_intersection

