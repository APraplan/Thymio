import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import cv_Lines as Lines
from cv_Intersections import Intersection
import pathlib
# for pathlib.Path().resolve()
import time
#import settings
import cv_Points

class Picture:
    """ in this class a picture (file)
        - gets prepared (blurring, edge enhancing filter, tresholding, nonmaximasupression)
        - undergoes a hough transformation
        - the hough transformation is transformed to make maxima extra visible
        - lines are found in this hough transformation
        - intersections are created from these lines
    """

    # ps it has to be a .png
    def __init__(self, file, filePath, identity, settings):
        print(' ')
        print('Anaysing picture: '+str(identity))
        self.settings = settings # copy settings rather than importing so they can be externally changed
        self.data = file
        self.filePath = filePath
        ##### inspired by the solutions of the exercise session ######
        # We choose to have a final matrix of the following
        # dimensions.

        self.x_max, self.y_max = file.shape
        if settings.THETARANGE == '0 to 2*PI':
            self.theta_max, self.theta_min = 2.0 * math.pi, 0.0  # full circle -> different from assistants solution
        elif settings.THETARANGE == '-PI/2 to PI':
            self.theta_max, self.theta_min = 3 * math.pi / 2, -math.pi / 2  # this should improve precision around theta = 0
        self.r_min, self.r_max = 0.0, math.hypot(self.x_max, self.y_max)
        self.processedData = None
        if False:
            self.r_dim, self.theta_dim = settings.R_DIM, settings.THETA_DIM
        else:
            self.r_dim, self.theta_dim = int(self.r_max), int(200*(self.theta_max-self.theta_min))


    ########################################################################################################################
    # preparing data
    ########################################################################################################################


    def blurImage(self, data):
        """ blur the image
            input: image as np.array, output: image as np.array, """
        # first, we blur
        # bilateral filter sounds like the best option as it preserves edges
        bilateral_filtered = cv2.bilateralFilter(data, self.settings.KERNELSIZEFILTER, 15, 15)
        if self.settings.SHOWFILTERSTEPS:
            plt.close()
            plt.imshow(bilateral_filtered, cmap='gray')
            plt.xlabel('y [pixels]')
            plt.ylabel('x [pixels]')
            plt.title("Blurred using a bilateral filter")
            if self.settings.SHOWFILTERSTEPS:
                plt.show()
            plt.savefig(self.filePath[:len(self.filePath) - 4] + 'bilateralFiltered.png')
            plt.close()
        return bilateral_filtered

        ########################################################################################################################
        # filter to highlight edges
        ########################################################################################################################

    def edgeEnhance(self, data, filtertype = None):
        """ enhance edges in an image
                input: image as np.array, output: image as np.array, """
        if filtertype == None:
            filtertype = self.settings.FILTERTYPE
        if filtertype == 'sobel':
            ##### inspired by the solutions of the exercise session ######
            # second, we filter for edges using a 2 directional sobel filter
            sobelx = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=5)
            edgeEnhanced = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2.0), 1.0, cv2.pow(sobely, 2.0), 1.0, 0.0))

        elif filtertype == 'canny':  # gives double edges
            edgeEnhanced = cv2.Canny(data, 60, 120)

        elif filtertype == 'laplacian':  # gives bad result imo
            edgeEnhanced = cv2.Laplacian(data, cv2.CV_64F)
        else:
            print('pictures err 43')
            quit()
        plt.close()
        plt.imshow(edgeEnhanced, cmap='gray')
        plt.xlabel('y [pixels]')
        plt.ylabel('x [pixels]')
        plt.title("Enhanced edges by using a "+filtertype+" filter" )
        if self.settings.SHOWFILTERSTEPS:
            plt.show()
        plt.savefig(self.filePath[:len(self.filePath)-4]+'edgeenhanced.png')
        plt.close()
        return edgeEnhanced


        ########################################################################################################################
        # treshold to increase contrast
        ########################################################################################################################

    def treshold(self, data, black=None):
            """ treshold an image
                        input: image as np.array, output: image as np.array, """
            if black == None:
                black = self.settings.TRESHOLDINGBLACK
            datacopy = data.copy()
            for x in range(data.shape[0]):
                for y in range(data.shape[1]):
                    if data[x][y] < black:
                        datacopy[x][y] = 255  # white
                    elif data[x][y] >= black:
                        datacopy[x][y] = 0  # black
                    else:
                        print('pictures err 44 ')
            plt.close()
            plt.imshow(datacopy, cmap='gray')
            plt.title("Tresholded edge enhanced object with treshold: "+str(black))
            plt.xlabel('y [pixels]')
            plt.ylabel('x [pixels]')
            plt.savefig(self.filePath[:len(self.filePath)-4]+'tresholded.png')
            if self.settings.SHOWFILTERSTEPS:
                plt.show()
            plt.close()
            return datacopy

        ########################################################################################################################
        # erosion to make edges thinner
        ########################################################################################################################

    def erode(self, data, kernelsize=None, nb_iterations=None):
            """ erode edges in an image
                        input: image as np.array, output: image as np.array, """
            if kernelsize == None:
                kernelsize = self.settings.KERNELSIZEEROSION
            if nb_iterations == None:
                nb_iterations = self.settings.EROSIONITERATIONS
            # I used erosion from https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
            # this function enlarges the black part, to reduce the white part
            # so we first invert the data so the edges are the white part

            # invert data
            invertedData = data
            for x in range(data.shape[0]):
                for y in range(data.shape[1]):
                    if data[x][y] == 0:
                        invertedData[x][y] = 255  # white
                    elif data[x][y] == 255:
                        invertedData[x][y] = 0  # black

            # erode
            if self.settings.EROSIONKERNELTYPE == 'square':
                kernel = np.ones((kernelsize, kernelsize), np.uint8)
            elif self.settings.EROSIONKERNELTYPE == 'plus':
                kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0] ]).astype(np.uint8)
            else:
                print('picture error 45')
                quit()
            invertedErosion = cv2.erode(invertedData, kernel, iterations=nb_iterations)  # 2 still works

            # invert again
            erosion = invertedErosion
            for x in range(invertedErosion.shape[0]):
                for y in range(invertedErosion.shape[1]):
                    if invertedErosion[x][y] == 0:
                        erosion[x][y] = int(255)  # white
                    elif invertedErosion[x][y] == 255:
                        erosion[x][y] = int(0)  # black
            plt.close()
            plt.title("Edge enhanced picture after eroding "+str(nb_iterations)+" times")
            plt.xlabel('y [pixels]')
            plt.ylabel('x [pixels]')
            plt.imshow(erosion, cmap='gray')
            if self.settings.SHOWFILTERSTEPS:
                plt.show()
            plt.savefig(self.filePath[:len(self.filePath) - 4] + 'eroded.png')
            plt.close()
            return erosion

        ########################################################################################################################
        # dilation to make edges wider
        ########################################################################################################################

    def dilate(self, data, kernelsize= None, nb_iterations= None):
            """ dilate edges in an image
                        input: image as np.array, output: image as np.array, """
            if kernelsize== None:
                kernelsize = self.settings.KERNELSIZEDILATION
            if nb_iterations == None:
                nb_iterations = self.settings.DILATIONITERATIONS
            if nb_iterations > 0:
                if self.settings.DILATIONKERNELTYPE == 'square':
                    kernel = np.ones((kernelsize, kernelsize), np.uint8)  # 2 tends to be optimal
                elif self.settings.DILATIONKERNELTYPE == 'plus':
                    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0] ]).astype(np.uint8)
                dilated = cv2.erode(data, kernel, iterations=nb_iterations)  # 2 still works
                plt.close()
                plt.imshow(dilated, cmap='gray')
                plt.title("Edge enehanced picture after dilating "+str(nb_iterations)+' times')
                plt.xlabel('y [pixels]')
                plt.ylabel('x [pixels]')
                if self.settings.SHOWFILTERSTEPS:
                    plt.show()
                plt.savefig(self.filePath[:len(self.filePath) - 4] + 'dilated.png')
                plt.close()
                return dilated
            return data


    ########################################################################################################################
    # perform hough transform & now get all lines and intersections
    ########################################################################################################################


    def calculate_hough_transform(self, data):
        """ calculate the hough space of an image
                input: image as np.array, output: hough space as np.array, """
        img = data
        ##### based on solution of exercise session ######
        # Creating the empty hough transform matrix
        hough_space = np.zeros((self.r_dim, self.theta_dim))

        # hough
        for x in range(self.x_max):
            for y in range(self.y_max):
                if int(img[x, y]) == 255: continue
                for idx_theta in range(self.theta_dim):
                    if self.settings.THETARANGE == '-PI/2 to PI': # this doesnt work yet
                        # i = 0 -> theta = min
                        # i = theta_dim -> theta = max
                        # theta = min + i*(max-min)/dim
                        #theta = 1.0 * (idx_theta-self.theta_dim/4) * (self.theta_max-self.theta_min) / self.theta_dim
                        theta = self.theta_min + idx_theta * (self.theta_max-self.theta_min)/(self.theta_dim)
                        r = x * math.cos(theta) + y * math.sin(theta)
                        ir = self.r_dim * (1.0 * r) / self.r_max
                        if hough_space[int(ir), int(idx_theta)] != 255:
                            hough_space[int(ir), int(idx_theta)] += 1
                    elif self.settings.THETARANGE == '0 to 2*PI':
                        theta = 1.0 * idx_theta * self.theta_max / self.theta_dim
                        r = x * math.cos(theta) + y * math.sin(theta)
                        ir = self.r_dim * (1.0 * r) / self.r_max
                        if hough_space[int(ir), int(idx_theta)] != 255:
                            hough_space[int(ir), int(idx_theta)] += 1
        plt.close()
        plt.imshow(hough_space, origin='lower')
        plt.xlim(0, self.theta_dim)
        plt.ylim(0, self.r_dim)
        tick_locs = [i for i in range(0, self.theta_dim, 200)]
        tick_lbls = [round((self.theta_min+i * (self.theta_max-self.theta_min) / self.theta_dim), 1) for i in
                             range(0, self.theta_dim, 200)]
        plt.xticks(tick_locs, tick_lbls)
        tick_locs = [i for i in range(0, self.r_dim, 100)]
        tick_lbls = [round((1.0 * i * self.r_max) / self.r_dim, 1) for i in range(0, self.r_dim, 100)]

        # theta = (self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim)
        # => i = (theta - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
        #xlim = round(math.pi - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
        # (pi + pi/2)*dim/(2*pi) = dim * 3/2 /2 = dim*3/4
        xlim = round(self.theta_dim*3/4)
        y0 = 0
        ylim = self.r_dim
        xpts = [xlim, xlim]
        ypts = [y0, ylim]
        plt.plot(xpts, ypts, color='red')
        plt.yticks(tick_locs, tick_lbls)
        plt.xlabel(r'$\Theta$')
        plt.ylabel(r'$\rho$')
        plt.title('Hough Space')
        plt.savefig(self.filePath[:-4]+'houghspace.png')
        if self.settings.SHOWHOUGH:
            plt.show()
        plt.close()
        return hough_space

        ########################################################################################################################
        # edit hough to highlight maxima
        ########################################################################################################################

    def enhanceHough(self, data, max=None):
        """ to avoid having to configure the hough treshold by hand for every seperate picture,
            I scale them so they always have maximum value "max"
                        input: hough space as np.array, output: hough space as np.array, """
        if max == None:
            max = self.settings.MAX
        curr_max = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if curr_max < data[i, j]:
                    curr_max = data[i, j]
        if curr_max < max:
            data_copy = data
            if curr_max == 0:
                print('settings.TRESHOLDINGBLACK is probably too high, curr_max = 0 (hough space is all zero)')
            else:
                scale = max / curr_max
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        data_copy[i, j] = data[i, j]
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        temp = int(scale * data[i, j])
                        if temp > 255:
                            temp = 255
                        data_copy[i, j] = temp
                data = data_copy
        plt.close()
        plt.imshow(data, origin='lower')
        plt.xlim(0, self.theta_dim)
        plt.ylim(0, self.r_dim)
        tick_locs = [i for i in range(0, self.theta_dim, 200)]
        tick_lbls = [round((self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim), 1) for i in
                     range(0, self.theta_dim, 200)]
        plt.xticks(tick_locs, tick_lbls)
        tick_locs = [i for i in range(0, self.r_dim, 100)]
        tick_lbls = [round((1.0 * i * self.r_max) / self.r_dim, 1) for i in range(0, self.r_dim, 100)]

        # theta = (self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim)
        # => i = (theta - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
        # xlim = round(math.pi - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
        # (pi + pi/2)*dim/(2*pi) = dim * 3/2 /2 = dim*3/4
        xlim = round(self.theta_dim * 3 / 4)
        y0 = 0
        ylim = self.r_dim
        xpts = [xlim, xlim]
        ypts = [y0, ylim]
        plt.plot(xpts, ypts, color='red')
        plt.yticks(tick_locs, tick_lbls)
        plt.xlabel(r'$\Theta$')
        plt.ylabel(r'$\rho$')
        plt.title('scaled Hough Space')
        plt.savefig(self.filePath[:-4] + 'scaledHoughspace.png')
        if self.settings.SHOWHOUGH:
            plt.show()
        plt.close()
        return data

    def find_optima(self, hough_space):
        """ find the optima in the hough space.
            simplify the shapes by dilating & eroding (helps to avoid M00 = 0
            determine the centre of contours
                        input: tresholded hough space as np.array, output: list with maxima locations
        """
        maxima_locs = []
        # First we convert the pixels to uint8 in order to apply the thresholding
        converted_hough_space = cv2.convertScaleAbs(hough_space)

        ##### copied from exercise session ######
        _, thresholded_hough_space = cv2.threshold(converted_hough_space, 180, 255, cv2.THRESH_BINARY)

        # DISPLAY
        plt.close()
        plt.imshow(thresholded_hough_space, origin='lower')
        plt.xlim(0, self.theta_dim)
        plt.ylim(0, self.r_dim)
        tick_locs = [i for i in range(0, self.theta_dim, 200)]
        tick_lbls = [round((self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim), 1) for i in
                     range(0, self.theta_dim, 200)]
        plt.xticks(tick_locs, tick_lbls)
        tick_locs = [i for i in range(0, self.r_dim, 100)]
        tick_lbls = [round((1.0 * i * self.r_max) / self.r_dim, 1) for i in range(0, self.r_dim, 100)]

        # theta = (self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim)
        # => i = (theta - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
        # xlim = round(math.pi - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
        # (pi + pi/2)*dim/(2*pi) = dim * 3/2 /2 = dim*3/4
        xlim = round(self.theta_dim * 3 / 4)
        y0 = 0
        ylim = self.r_dim
        xpts = [xlim, xlim]
        ypts = [y0, ylim]
        plt.plot(xpts, ypts, color='red')
        plt.yticks(tick_locs, tick_lbls)
        plt.xlabel(r'$\Theta$')
        plt.ylabel(r'$\rho$')
        plt.title('thresholded Hough Space')
        plt.savefig(self.filePath[:-4] + 'thresholdedHoughspace.png')
        if self.settings.SHOWHOUGH:
            plt.show()
        plt.close()

        # DILATE
        if self.settings.HOUGHDILATIONIT > 0:
            if self.settings.HOUGHDILATIONKERNELTYPE == 'square':
                kernel = np.ones((self.settings.HOUGHDILATIONKERNELSIZE, self.settings.HOUGHDILATIONKERNELSIZE), np.uint8)
            elif self.settings.HOUGHDILATIONKERNELTYPE == 'plus':
                kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0] ]).astype(np.uint8)
                #kernel = np.array([np.array([0, 1, 0]),np.array([1, 1, 1]),np.array([0, 1, 0])])
            hough_maxima_dilated = cv2.dilate(thresholded_hough_space, kernel, iterations=self.settings.HOUGHDILATIONIT)

            # DISPLAY
            plt.close()
            plt.imshow(hough_maxima_dilated, origin='lower')
            plt.xlim(0, self.theta_dim)
            plt.ylim(0, self.r_dim)
            tick_locs = [i for i in range(0, self.theta_dim, 200)]
            tick_lbls = [round((self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim), 1) for i in
                         range(0, self.theta_dim, 200)]
            plt.xticks(tick_locs, tick_lbls)
            tick_locs = [i for i in range(0, self.r_dim, 100)]
            tick_lbls = [round((1.0 * i * self.r_max) / self.r_dim, 1) for i in range(0, self.r_dim, 100)]

            # theta = (self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim)
            # => i = (theta - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
            # xlim = round(math.pi - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
            # (pi + pi/2)*dim/(2*pi) = dim * 3/2 /2 = dim*3/4
            xlim = round(self.theta_dim * 3 / 4)
            y0 = 0
            ylim = self.r_dim
            xpts = [xlim, xlim]
            ypts = [y0, ylim]
            plt.plot(xpts, ypts, color='red')
            plt.yticks(tick_locs, tick_lbls)
            plt.xlabel(r'$\Theta$')
            plt.ylabel(r'$\rho$')
            plt.title('dilated Hough Space')
            plt.savefig(self.filePath[:-4] + 'dilatedHoughspace.png')
            if self.settings.SHOWHOUGH:
                plt.show()
            plt.close()
        else:
            hough_maxima_dilated = thresholded_hough_space
        if self.settings.HOUGHEROSIONIT>0:
            # ERODE
            if self.settings.HOUGHEROSIONKERNELTYPE == 'square':
                kernel = np.ones((self.settings.HOUGHEROSIONKERNELSIZE, self.settings.HOUGHEROSIONKERNELSIZE), np.uint8)
            elif self.settings.HOUGHEROSIONKERNELTYPE == 'plus':
                kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0] ]).astype(np.uint8)
                #kernel = np.array([np.array([0, 1, 0]),np.array([1, 1, 1]),np.array([0, 1, 0])])
            hough_maxima_eroded = cv2.erode(hough_maxima_dilated, kernel, iterations=self.settings.HOUGHEROSIONIT)

            # DISPLAY
            plt.close()
            plt.imshow(hough_maxima_eroded, origin='lower')
            plt.xlim(0, self.theta_dim)
            plt.ylim(0, self.r_dim)
            tick_locs = [i for i in range(0, self.theta_dim, 200)]
            tick_lbls = [round((self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim), 1) for i in
                         range(0, self.theta_dim, 200)]
            plt.xticks(tick_locs, tick_lbls)
            tick_locs = [i for i in range(0, self.r_dim, 100)]
            tick_lbls = [round((1.0 * i * self.r_max) / self.r_dim, 1) for i in range(0, self.r_dim, 100)]

            # theta = (self.theta_min + i * (self.theta_max - self.theta_min) / self.theta_dim)
            # => i = (theta - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
            # xlim = round(math.pi - self.theta_min)*self.theta_dim/(self.theta_max - self.theta_min)
            # (pi + pi/2)*dim/(2*pi) = dim * 3/2 /2 = dim*3/4
            xlim = round(self.theta_dim * 3 / 4)
            y0 = 0
            ylim = self.r_dim
            xpts = [xlim, xlim]
            ypts = [y0, ylim]
            plt.plot(xpts, ypts, color='red')
            plt.yticks(tick_locs, tick_lbls)
            plt.xlabel(r'$\Theta$')
            plt.ylabel(r'$\rho$')
            plt.title('eroded Hough Space')
            plt.savefig(self.filePath[:-4] + 'erodedHoughspace.png')
            if self.settings.SHOWHOUGH:
                plt.show()
                        #time.sleep(3)
            plt.close()
        else:
            hough_maxima_eroded = hough_maxima_dilated
        ##### copied from exercise session ######
        # find contours in the binary image
        output = hough_maxima_eroded.copy()
        contours, hierarchy = cv2.findContours(hough_maxima_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Converting to color for the visualisation
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)
            # calculate x, y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                maxima_locs.append((cX, cY))
                cv2.circle(output, (cX, cY), 2, (255, 0, 0), -1)
            else: # avoid "division by zero" crash
                print('maximum skipped as M00 == 0')
                #########################################
        return maxima_locs

        ########################################################################################################################
        # get lines from hough
        ########################################################################################################################

    def find_all_lines_from_maxima(self, maxima_locs):
        """ find lines from the maxima.
                input: list of maxima tuples, output: /, a list with Line objects is saved as self.lines
        """
        lines = []
        identity = 0
        lineInfo = []
        ##### inspired by exercise session ######
        for (j, i) in maxima_locs:
            if self.settings.THETARANGE == '0 to 2*PI':
                r = abs(round((1.0 * i * self.r_max) / self.r_dim, 2))
                theta = round((1.0 * j * self.theta_max) / self.theta_dim, 2)
            else:
                r = abs(round((1.0 * i * self.r_max) / self.r_dim, 2))
                #theta = round(((1.0 * int(j+(self.theta_dim/4)) * (self.theta_max)) / self.theta_dim), 2)
                theta = self.theta_min + j * (self.theta_max - self.theta_min) / (self.theta_dim)
                if theta < 0:
                    theta += 2*math.pi
                theta = round(theta, 2)
                #print('theta: '+str(theta))
            if theta >= 2*math.pi:
                theta = round(theta - 2*math.pi)
            if theta == round(math.pi, 2):
                theta = round(theta-0.01,2)
            if theta == round(3 * math.pi / 2, 2):
                theta = round(theta+0.01, 2)
            # since I extended the theta range from 0-> pi to 0-> 2pi, i have to get rid of the out of bounds zone
            # between pi and 3/2 pi
            if (theta <= math.pi or theta >= 3 * math.pi / 2) and round(r,2) > 0:
            #if True:
                info = (round(theta,2), round(r, 2))
                if info not in lineInfo: # dont introduce duplicate lines, this will introduce many duplicate intersections
                    new_line = Lines.Line3(r, theta, identity)
                    identity += 1
                    lineInfo.append((theta, r))
                    lines.append(new_line)
                else:
                    print('discarded a duplicate line (this should be pretty rare/impossible)')
        self.lines = lines


    def visualiseLinesAndIntersections(self):
        """
        visualise self.lines and self.intersections
        """
        def addlines(self, ax, lines):
            for line in lines:
                ##### inspired by exercise session ######
                px = []
                py = []
                for i in range(-self.y_max - 40, self.y_max + 40, 1):
                    px.append(math.cos(-line.theta) * i - math.sin(-line.theta) * line.r)
                    py.append(math.sin(-line.theta) * i + math.cos(-line.theta) * line.r)
                ax.plot(px, py, linewidth=2, color="r")

                # alternative way of plotting to assert a & b (y = ax+b) are determined the right way
                px = []
                py = []
                if line.type == 'horizontal':
                    for y in range(self.y_max):
                        x = line.x
                        px.append(x)
                        py.append(y)
                else:
                    for x in range(self.x_max):
                        y = int(line.m * x + line.b)
                        if 0 < y and y < self.y_max:
                            px.append(x)
                            py.append(y)
                ax.plot(py, px, linewidth=1, color="b")

        ########################################################################################################################
        # get intersections from lines
        ########################################################################################################################

        def addintersections(self, ax, intersections):
            """
                add intersections to ax plot
            """
            for intersections_of_line_i in intersections:
                for intersection in intersections_of_line_i:
                    x = intersection.coordinate.x
                    y = intersection.coordinate.y
                    ax.plot(y, x, "ro")
        plt.close()
        #plt.imshow(self.data, cmap='gray')
        #fix, ax2 = plt.subplots()
        #ax2.imshow(self.processedData, cmap='gray')
        #ax2.autoscale(False)

        fig, ax = plt.subplots()
        ax.imshow(self.processedData, cmap='gray')
        ax.autoscale(False)

        addlines(self, ax, self.lines)
        addintersections(self, ax, self.intersections)

        # path = str(pathlib.Path().resolve()) + settings.FILENAME_PICTURES
        filePath = self.filePath[:len(self.filePath) - 4] + 'withlines.png'
        plt.title("Obstacle with the detected lines")
        plt.xlabel('y [pixels]')
        plt.ylabel('x [pixels]')
        plt.savefig(filePath)
        if self.settings.SHOWANALYSEDPICTURE:
            plt.show()
        # time.sleep(0.5)
        plt.close()

    def find_intersections_from_lines(self):
        """
        use the lines to determine where they intersect,
        if this intersection is within the picture, the intersection is saved in self.intersections
        self.intersections is a list with Intersection objects
        """
        lines = self.lines
        number_of_lines = len(lines)
        intersections = []
        intersectionCoordinates = []
        for i in range(number_of_lines):  # 0 .... n-1
            intersections_of_line_i = []
            for j in range((i + 1), number_of_lines):  # i+1 ... n-1
                line1 = lines[i]
                assert line1.identity == str(chr(65+i))
                line2 = lines[j]
                assert line2.identity == str(chr(65+j))
                identity = (line1.identity, line2.identity)
                if line1.m != line2.m:  # otherwise they would be parallel
                    new_intersection = Intersection(line1, line2, identity, self.data)
                    if new_intersection.is_in_picture:
                        """
                                                while (new_intersection.coordinate.x, new_intersection.coordinate.y) in intersectionCoordinates:
                            print('we have two intersections with the same coordinate ')
                            print(intersectionCoordinates)
                            print((new_intersection.coordinate.x, new_intersection.coordinate.y))
                            if new_intersection.lineh.type == 'horizontal' and new_intersection.linel.type != 'vertical':
                                new_intersection.coordinate = Points.Point(new_intersection.coordinate.x, new_intersection.coordinate.y + 0.00001)
                            elif new_intersection.lineh.type == 'horizontal' and new_intersection.linel.type == 'vertical':
                                print('i dont know what to do here')
                                assert(0 == 1)
                            elif new_intersection.linel.type == 'vertical':
                                new_intersection.coordinate = Points.Point(new_intersection.coordinate.x+0.00001, new_intersection.coordinate.y)
                            else:
                                new_intersection.coordinate = Points.Point(new_intersection.coordinate.x + 0.00001,
                                                                            new_intersection.coordinate.y + 0.00001)
                        assert (new_intersection.coordinate.x, new_intersection.coordinate.y) not in intersectionCoordinates
                        """
                        if (new_intersection.coordinate.x, new_intersection.coordinate.y) not in intersectionCoordinates:
                            intersections_of_line_i.append(new_intersection)
                            # ^^two intersections should never have the same coordinate, this will mess up the sorting algorithm
                            intersectionCoordinates.append((new_intersection.coordinate.x, new_intersection.coordinate.y))
                        else:
                            print(' ')
                            print('intersection at '+str((new_intersection.coordinate.x, new_intersection.coordinate.y))+' was removed because it seems to be a duplicate')
                            print('this shouldnt be a problem')
                        assert new_intersection.line1.identity == line1.identity
                        assert new_intersection.line2.identity == line2.identity
                        if (new_intersection.lineh.type != 'horizontal'):
                            assert (new_intersection.lineh.m > new_intersection.linel.m)
                        assert new_intersection.linel.type != 'horizontal'
                else:
                    pass
            intersections.append(intersections_of_line_i)
        self.intersections = intersections
        return intersections

    def get_color(self, coordinate):
        """
        return the image grayscale data at coordinate (coordinate is a Point object)
        """
        return self.data[int(coordinate.x), int(coordinate.y)]  # should work??
        # get the color of a certain coordinate using self.data
