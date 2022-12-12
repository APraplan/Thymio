### just a reminder: black <-> 0, white <-> 255 TREshold 125?
"""
A file where I tried to bring all settings together
depending on lighting, play with TRESHOLDINGBLACK, MAPBLACKTRESHOLD and GRIDMAPBLACKTRESHOLD
play with display settings if you want to see more of the steps
"""


### DISPLAY SETTINGS ###
SHOWINITIAL = 0 # show the inital picture with border
SHOWSEPERATEOBJECTS = 0 # show each isolated object that comes out of the gridded map
PROMPTPICS = 0 # for each isolated object (if SHOWSEPERATEOBJECTS), prompt the user to verify this is an actual object
SHOWFILTERSTEPS = 0 # show the filter steps the pictures go through to enhance edges
SHOWHOUGH = 0 # show the hough space (original, scaled, tresholded, dilated, eroded)
SHOWANALYSEDPICTURE = 0 # show object with all detected lines
SHOWSEPERATECLOSEDLOOPS = 0 # show the loops formed out of the lines
SHOWRESULT = 0 # show the end result
CLEAR = 0 #clear all prints
SHOWENDRESULT = 0
DISPLAYFUSING = 0

### FILTERING ###
# Edge enhancment
FILTERTYPE = 'sobel'
KERNELSIZEFILTER = 5

# Tresholding the lines after edge enhancment
TRESHOLDINGBLACK = 1500 # use showfiltersteps to determine this one, different for every picture and so on
                        # digital pictures: 10000
# erosion
KERNELSIZEEROSION = 3
EROSIONITERATIONS = 2
EROSIONKERNELTYPE = 'square' # square for all ones; plus for a 3x3 plus shaped kernel, this works better for diagonals

# dilation
KERNELSIZEDILATION = 3
DILATIONITERATIONS = 1
DILATIONKERNELTYPE = 'square' # square for all ones; plus for a 3x3 plus shaped kernel


### HOUGH SPACE ###
# hough space dimensions (R x Theta)
# 200 & 300 were chosen by the assistants in ex2
R_DIM = 800 #800 # R_MAX*100
THETA_DIM  = 1200 # the resolution: 100 x 2 x PI #628

GETTINGMAXIMA = 'likeExerciseSession'
NBOFMAXIMA = 10 # only retrieve NBOFMAXIMA from the hough space

# scale the hough space so the highest value becomes 'max', only if the maximum was lower than max
MAX = 400   # higher number seems to increase the amount of 'wrong' lines
            # 400 seems to work better for real pictures and nothing wrong with too manylines, just takes longer
            # 300 absolute minimum for maps

# hough dilation
HOUGHDILATIONKERNELSIZE = 4 # higher size seems to reduce the amount of lines by blending them together, while increasing their error
HOUGHDILATIONIT = 4
HOUGHDILATIONKERNELTYPE = 'square'

# hough erosion
HOUGHEROSIONKERNELSIZE = 4 # higher size seems to reduce the amount of lines by blending them together, while increasing their error
HOUGHEROSIONIT = 0
HOUGHEROSIONKERNELTYPE = 'square'


### IMAGE SETTINGS ###
# determine which file should be read
PICTUREALREADYPREPARED = 0
current_test = 'real'

def getFILENAME_MAIN():
    global current_test
    if current_test == 'wooden background':
        FILENAME_MAIN = 'report_images/woodBackGround.png'
    if current_test == 'picture07/12':
        FILENAME_MAIN = 'PICTURES/map_without_thymio7_12.png'

    if current_test == 'polyg':
        FILENAME_MAIN = 'PICTURES/polyg.png'

    if current_test == 'poly2':
        FILENAME_MAIN = 'PICTURES/poly2.png'

    if current_test == 'real':
        FILENAME_MAIN = 'map_without_thymio.png'

    if current_test == 'real2911b':
        FILENAME_MAIN = 'PICTURES/map_without_thymio2911b.png'
        
    if current_test == 'real2911':
        FILENAME_MAIN = 'PICTURES/map_without_thymio2911.png'

    if current_test == 'circle':
        FILENAME_MAIN = 'PICTURES/circle.png'

    if current_test == 'realtest':
        FILENAME_MAIN = 'PICTURES/realtest.png'

    if current_test == 'map2':
        FILENAME_MAIN = 'PICTURES/Map2.png'

    if current_test == 'room':
        FILENAME_MAIN = 'PICTURES/room.png'

    if current_test == 'line':
        FILENAME_MAIN = 'PICTURES/aline.png'

    if current_test == 'pentagonfilled':
        FILENAME_MAIN = 'PICTURES/pentagonfilled.png'

    if current_test == 'pentagon':
        FILENAME_MAIN = 'PICTURES/pentagon.png'

    if current_test == 'emptysquare':
        FILENAME_MAIN = 'PICTURES/asquare.png'

    if current_test == 'filledsquare':
        FILENAME_MAIN = 'PICTURES/afilledsquare.png'

    if current_test == 'map1reduced':
        FILENAME_MAIN = 'PICTURES/map1reducedpixels.png'

    if current_test == 'map1semireduced':
        FILENAME_MAIN = 'PICTURES/map1semireduced.png'

    if current_test == 'map1': #its closer to 1080p but same thing really
        FILENAME_MAIN = 'PICTURES/map1720p.png'

    assert FILENAME_MAIN != None
    return FILENAME_MAIN

PROMPTTHYMIOWIDTH = 0
STANDARDTHYMIOWIDTH = 80 #pixels, its only 40 really,
# but to compensate for slight control inaccuracys we make it a bit bigger

### MAP SETTINGS ###
MAPTYPE = 'graph'
MAPBLACKTRESHOLD = 100 # treshold to determine which loops get fused as obstacle

THETARANGE = '-PI/2 to PI' #'0 to 2*PI' # alternative: '-PI/2 to PI', as anything between Pi and 3/2 PI is out of the image


# GRIDMAP SETTINGS
GRIDMAPBLACKTRESHOLD = 70 # this threshold determines what parts of the grid are condsidered white and which ones are considered black
GRIDSIZE = 20 # one grid is GRIDSIZExGRIDSIZE pixels -> small grid might take a long time
# smaller: slower intial analysis (squared)
# bigger: might accidentally fuse two objects or steal parts of a different object

BORDER = 10 # this determines how many nonblack pixels around an object
# are required to stop fusing surrounding chunks, better safe than sorry here,
# but big values might accidentally fuse two objects
# 10 seems to work pretty well so far

OUTERBORDER = BORDER # go around the figure and add OUTERBORDER white pixels, using BORDER makes sense
BORDERCOLOUR = 'max white from first row' # takes the max brightness in the first row
# alternative: 'white'
# white: brightness = 255 -> this creates extra edges because of the big contrast between 255 and real life white (+_160)
# this edge tends to be way brighter than other spots in the hough transform
# increases line inacurracy and misses

assert BORDER <= GRIDSIZE # otherwise the program crashes when asserting the border is white
#assert OUTERBORDER <= BORDER #otherwise, extra pixels that are completely unnecessairy are introduced
