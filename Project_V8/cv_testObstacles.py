import obstacleDetection
import global_navigation
import cv_visgraph as vg2
import pyvisgraph as vg
import cv_Points as Points
#### temporary part bcs I have no picture ####
import cv_settings as settings
#settings.current_test = 'poly2'
#settings.TRESHOLDINGBLACK = 3000

##############################################

obstacleDetection.visualsProfile1()
obstacleDetection.settingsProfile1()
#chunkedMap = obstacleDetection.step1()
#chunkedMap = obstacleDetection.step2(chunkedMap)
#chunkedMap, polys, thymiowidth = obstacleDetection.step3(chunkedMap)

chunkedMap, polys, thymiowidth = obstacleDetection.analysePicture2()
string = obstacleDetection.savePoints(chunkedMap, polys, thymiowidth)
##############################################

#start = vg.Point(1,1)
#end = vg.Point(chunkedMap.originaldata.shape[1]-1, chunkedMap.originaldata.shape[0]-1)
#settings.GRIDSIZE = 40  # instead of 30
#settings.BORDER = 40 # instead of 20
#settings.OUTERBORDER = 40 # instead of 20

#settings.GRIDMAPBLACKTRESHOLD = 100 #instead of 50

#shortest, polygones = global_navigation.path_planning(polys, start, end, thymiowidth/2, gridMap = chunkedMap)

#string = "50{0.1080,0.0720}[/(+0.0839,+0.0002)/(+0.0630,+0.0001)/(+0.0613,+0.0120)/(+0.0392,+0.0063)/(+0.0384,+0.0102)/(+0.0362,+0.0256)/(+0.0639,+0.0329)/(+0.0669,+0.0197)/(+0.0700,+0.0165)/(+0.0810,+0.0202)/][/(+0.0033,+0.0486)/(+0.0290,+0.0683)/(+0.0259,+0.0716)/(-0.0001,+0.0718)/(-0.0000,+0.0520)/][/(+0.1060,+0.0486)/(+0.0932,+0.0702)/(+0.0762,+0.0611)/(+0.0893,+0.0388)/][/(+0.0071,+0.0377)/(-0.0000,+0.0346)/(-0.0000,+0.0201)/(+0.0033,+0.0219)/(+0.0060,+0.0210)/(+0.0135,+0.0194)/(+0.0168,+0.0292)/][/(+0.0360,+0.0000)/(+0.0156,-0.0000)/(+0.0133,+0.0107)/(+0.0361,+0.0123)/]"
#string = "50{0.1080,0.0720}[/(+0.0613,+0.0122)/(+0.0542,+0.0105)/(+0.0393,+0.0061)/(+0.0362,+0.0260)/(+0.0546,+0.0310)/(+0.0639,+0.0329)/(+0.0678,+0.0155)/(+0.0719,+0.0159)/(+0.0811,+0.0186)/(+0.0840,+0.0001)/(+0.0631,+0.0000)/][/(-0.0000,+0.0347)/(-0.0000,+0.0195)/(+0.0042,+0.0217)/(+0.0137,+0.0199)/(+0.0150,+0.0228)/(+0.0175,+0.0285)/(+0.0070,+0.0378)/][/(+0.0155,-0.0001)/(+0.0134,+0.0104)/(+0.0361,+0.0107)/(+0.0360,-0.0001)/][/(-0.0001,+0.0718)/(+0.0267,+0.0708)/(+0.0291,+0.0682)/(+0.0033,+0.0489)/(-0.0001,+0.0523)/][/(+0.0760,+0.0614)/(+0.0929,+0.0708)/(+0.1059,+0.0487)/(+0.0892,+0.0389)/]"
thymiosize, minx, miny, maxx, maxy, obstacles = obstacleDetection.getPolys(string)

start_point = Points.Point(1,1)
goal_point = Points.Point(1000,1)
shortest, polygones  = global_navigation.path_planning(obstacles, start_point, goal_point, thymiosize/2, gridMap=None, minx=minx, miny=miny, maxx=maxx, maxy=maxy,
              usePackage=False)



