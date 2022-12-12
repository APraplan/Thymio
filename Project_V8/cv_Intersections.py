import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from cv_Points import Point
import cv_Lines as Lines

import cv_settings as settings


class Intersection:
    " rules: "
    " an intersection is where two lines meet. "
    " lineh must ALWAYS be the steepest line out of the two "
    " the identity of an intersection is the id of the two intersecting lines combined "
    def __init__(self, line1, line2, identity, pictureData):
        assert line1.m != line2.m # assert the lines are not parallel
        self.line1 = line1
        self.line2 = line2
        self.identity = identity
        m1 = line1.m
        b1 = line1.b
        m2 = line2.m
        b2 = line2.b
        if line1.type != 'horizontal' and line2.type != 'horizontal':
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
        elif line1.type == 'horizontal':
            x = line1.x
            y = m2 * x + b2
        elif  line2.type == 'horizontal':
            x = line2.x
            y = m1 * x + b1
        else:
            print('intersections err 0')
            assert(0==1)
        self.coordinate = Point(x, y)

        # determine wether or not an intersection is within map bounds
        x_max, y_max = pictureData.shape
        self.is_in_picture = False
        if 0 <= x and x <= x_max: #allow some slight deviation
            if 0 <= y and y <= y_max:
                self.is_in_picture = True

        self.objects_at_this_intersection = [0, 0, 0, 0]
        self.type = 'undetermined'
        if True :#self.is_in_picture:
            if line1.type != 'horizontal' and line2.type != 'horizontal':
                # see paper for derivation formula
                if m1 > m2:
                    self.lineh = self.line1
                    self.linel = self.line2
                elif m1 < m2:
                    self.lineh = self.line2
                    self.linel = self.line1
                else:
                    print('Intersection error 1')
                    quit()

                    # what if line1 and line2 have the same m? shouldnt be possible: they wouldnt intersect

                # we define 4 directions, a is the one in between h and l, to the right. From here define b, c, d clockwise
                # to get these directions, we calculate m of the bissetrice between h and l
                bissetrice_angle = (self.lineh.theta + self.linel.theta) / 2
            elif line1.type == 'horizontal':
                # see paper for derivation formula
                self.lineh = self.line1
                self.linel = self.line2

                bissetrice_angle = (math.pi / 2 + self.linel.theta) / 2
            elif line2.type == 'horizontal':
                # see paper for derivation formula
                self.lineh = self.line2
                self.linel = self.line1
                bissetrice_angle = (math.pi / 2 + self.linel.theta) / 2
            else:
                assert(1 == 0)
