import cv2
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ipywidgets import interact
import functions as fct


def get_speed_recorded(nb, filename):
    speeds = np.loadtxt(filename)

    return np.array([[speeds[nb][0] / 6.25], [speeds[nb][1] / 4.2]])


def load_video(name_vid):
    cap = cv2.VideoCapture(name_vid)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    video_imgs = []

    while True:
        ret, frame = cap.read()
        if ret == True:
            video_imgs.append(frame)
        # Break the loop
        else:
            break

    return video_imgs


def browse_images(images, titles):
    n = len(images)

    def view_image(i):
        plt.figure(figsize=(12, 12))
        plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(titles[i], y=-0.5)
        plt.show()

    interact(view_image, i=(0, n - 1))


def browse_Two_images(images0, images1, titles):
    n = len(images0)

    def view_image(i):
        plt.figure(figsize=(12, 12))
        plt.subplot(211), plt.imshow(images0[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Result with cam (green = non filtered, red = filtered)'), plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(images1[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Results without cam (green = non filtered, red = filtered)'), plt.xticks([]), plt.yticks([])

        plt.show()

    interact(view_image, i=(0, n - 1))


def browse_tracker_results(images0, images1, images2, images3, images4, images5, images6):
    n = len(images0)

    def view_image(i):
        plt.figure(figsize=(12, 18))
        plt.subplot(421), plt.imshow(images0[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Results'), plt.xticks([]), plt.yticks([])
        plt.subplot(423), plt.imshow(images1[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Mask front'), plt.xticks([]), plt.yticks([])
        plt.subplot(425), plt.imshow(images2[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Mask dilated and eroded front'), plt.xticks([]), plt.yticks([])
        plt.subplot(427), plt.imshow(images3[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Result front'), plt.xticks([]), plt.yticks([])
        plt.subplot(424), plt.imshow(images4[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Mask back'), plt.xticks([]), plt.yticks([])
        plt.subplot(426), plt.imshow(images5[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Mask dilated and erroded back'), plt.xticks([]), plt.yticks([])
        plt.subplot(428), plt.imshow(images6[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('results back'), plt.xticks([]), plt.yticks([])

        plt.show()

    interact(view_image, i=(0, n - 1))


def browse_template_results(images0, images1, images2):
    n = len(images0)

    def view_image(i):
        plt.figure(figsize=(12, 18))

        plt.subplot(311), plt.imshow(images0[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Template'), plt.xticks([]), plt.yticks([])
        plt.subplot(312), plt.imshow(images1[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Matching result'), plt.xticks([]), plt.yticks([])
        plt.subplot(313), plt.imshow(images2[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Result'), plt.xticks([]), plt.yticks([])

        plt.show()

    interact(view_image, i=(0, n - 1))


# Class detection Thymio to detect the two distinct point of colors on the map
class DetectionThymioDemo(object):
    def __init__(self, data_file):
        self.colors = np.loadtxt(data_file)

    def detection_front_demo(self, frame):
        result_img = []
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([self.colors[0] - 10, 100, 50])
        hi = np.array([self.colors[0] + 10, 255, 255])
        mask = cv2.inRange(img, lo, hi)
        result_img.append(cv2.bitwise_not(mask))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        result_img.append(cv2.bitwise_not(mask))
        image_mask = cv2.bitwise_and(frame, frame, mask=mask)
        result_img.append(image_mask)
        elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(elements) > 0:
            zone = max(elements, key=cv2.contourArea)
            area = cv2.contourArea(zone)
            ((x, y), radius) = cv2.minEnclosingCircle(zone)
            if area > 350:
                cv2.circle(result_img[2], (int(x), int(y)), int(radius), (0, 255, 0), 4)
                return True, x, y, result_img
        return False, 0, 0, result_img


    def detection_back_demo(self, frame):
        result_img = []
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([self.colors[1] - 10, 100, 50])
        hi = np.array([self.colors[1] + 10, 255, 255])
        mask = cv2.inRange(img, lo, hi)
        result_img.append(cv2.bitwise_not(mask))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        result_img.append(cv2.bitwise_not(mask))
        image_mask = cv2.bitwise_and(frame, frame, mask=mask)
        result_img.append(image_mask)
        elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(elements) > 0:
            zone = max(elements, key=cv2.contourArea)
            area = cv2.contourArea(zone)
            ((x, y), radius) = cv2.minEnclosingCircle(zone)
            if area > 350:
                cv2.circle(result_img[2], (int(x), int(y)), int(radius), (0, 255, 0), 4)
                return True, x, y, result_img
        return False, 0, 0, result_img


    def state_demo(self, frame):

        frame_results = []

        frame_results.append(frame)
        Z = np.matrix([[0.0],
                       [0.0],
                       [0.0]])

        front_found, xf, yf, results_front = self.detection_front_demo(frame)
        frame_results.append(results_front[0])
        frame_results.append(results_front[1])
        frame_results.append(results_front[2])
        back_found, xb, yb, results_back = self.detection_back_demo(frame)
        frame_results.append(results_back[0])
        frame_results.append(results_back[1])
        frame_results.append(results_back[2])

        if front_found and back_found:
            Z[0] = 2 / 7 * xf + 5 / 7 * xb
            Z[1] = 2 / 7 * yf + 5 / 7 * yb
            Z[2] = math.degrees(math.atan2(yb - yf, xb - xf)) + 180.0
            cv2.circle(frame_results[0], (int(xf), int(yf)), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame_results[0], (int(xb), int(yb)), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame_results[0], (int(Z[0]), int(Z[1])), 5, (255, 0, 0), cv2.FILLED)
            end_point = (int(Z[0] + 30 * math.cos(math.radians(Z[2]))), int(Z[1] + 30 * math.sin(math.radians(Z[2]))))
            cv2.arrowedLine(frame_results[0], (int(Z[0]), int(Z[1])), end_point, (255, 0, 0), 2)
        elif front_found:
            cv2.circle(frame_results[0], (int(xf), int(xf)), 5, (0, 255, 0), cv2.FILLED)
        elif back_found:
            cv2.circle(frame_results[0], (int(xb), int(yb)), 5, (0, 0, 255), cv2.FILLED)

        return frame_results


def find_template_demo(template_name, img_name):

    mymap = cv2.imread(img_name)

    results = []

    goal_template = cv2.imread(template_name)
    results.append(goal_template)
    c, w, h = goal_template.shape[::-1]
    res = cv2.matchTemplate(mymap, goal_template, cv2.TM_CCOEFF_NORMED)
    results.append(res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left_goal = max_loc
    bottom_right_goal = (top_left_goal[0] + w, top_left_goal[1] + h)
    cv2.rectangle(mymap,top_left_goal, bottom_right_goal, (255, 0, 0), 4)
    results.append(mymap)


    plt.figure(figsize=(12, 12))

    plt.subplot(311), plt.imshow(results[0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Template'), plt.xticks([]), plt.yticks([])
    plt.subplot(312), plt.imshow(results[1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Matching result'), plt.xticks([]), plt.yticks([])
    plt.subplot(313), plt.imshow(results[2], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('result'), plt.xticks([]), plt.yticks([])

    plt.show()


def apply_tracker_on_vid(vid_name, data_name):

    results0 = []
    results1 = []
    results2 = []
    results3 = []
    results4 = []
    results5 = []
    results6 = []

    titles = []

    video_imgs = load_video(vid_name)

    tracker = DetectionThymioDemo(data_name)

    for idx_frame in tqdm(range(len(video_imgs))):
        img = video_imgs[idx_frame].copy()

        # Apply tracker
        res = tracker.state_demo(img)

        titles.append("Frame {}".format(idx_frame))

        results0.append(res[0])
        results1.append(res[1])
        results2.append(res[2])
        results3.append(res[3])
        results4.append(res[4])
        results5.append(res[5])
        results6.append(res[6])

    browse_tracker_results(results0, results1, results2, results3, results4, results5, results6)


def kalman_filter_demo(vid_name, data_name, speed_name):

    thymio_measures = []
    thymio_state_record = []
    thymio_state_record_no_cam = []

    results = []
    results_no_cam = []
    titles = []

    dt = 0.05
    thymio_state = np.matrix([[325.0], [505.0], [298.0], [0.0], [0.0]])  # X, Y, angle, v, v_rot

    thymio_state_no_cam = np.matrix([[325.0], [505.0], [298.0], [0.0], [0.0]])  # X, Y, angle, v, v_rot
    thymio_connected = True

    video_imgs = load_video(vid_name)

    tracker = fct.DetectionThymio(data_name)

    kalman = fct.KalmanFilter(dt, thymio_state)
    kalman_no_cam = fct.KalmanFilter(dt, thymio_state_no_cam)

    for idx_frame in tqdm(range(len(video_imgs))):
        # Get a frame
        img = video_imgs[idx_frame].copy()
        img_no_cam = video_imgs[idx_frame].copy()

        # Kalman predict step
        thymio_state = kalman.predict(dt)
        thymio_state_record.append(thymio_state)
        thymio_state_no_cam = kalman_no_cam.predict(dt)
        thymio_state_record_no_cam.append(thymio_state_no_cam)

        # Apply tracker
        thymio_visible, measure_cam = tracker.state(img)
        thymio_visible_no_cam = False

        # Get speeds
        measure_rob = get_speed_recorded(idx_frame, speed_name)

        measures = np.append(measure_cam, measure_rob, axis=0)
        thymio_measures.append(measures)

        # Kalman update step
        thymio_state = kalman.update(thymio_visible, thymio_connected, measures)
        thymio_state_no_cam = kalman_no_cam.update(thymio_visible_no_cam, thymio_connected, measures)

        # Drawing
        img = fct.draw_my_print(img, thymio_measures, (0, 255, 0))
        img = fct.draw_my_print(img, thymio_state_record, (255, 0, 0))
        img_no_cam = fct.draw_my_print(img_no_cam, thymio_measures, (0, 255, 0))
        img_no_cam = fct.draw_my_print(img_no_cam, thymio_state_record_no_cam, (255, 0, 0))

        results.append(img)
        results_no_cam.append(img_no_cam)

        titles.append("Frame {}".format(idx_frame))

    browse_Two_images(results, results_no_cam, titles)
    
    return thymio_measures, thymio_state_record, thymio_state_record_no_cam

def plot_position(start, end, thymio_measures, thymio_state_record):

    measure = []
    state = []

    for i in range(start, end):
        measure.append([thymio_measures[i][0, 0], thymio_measures[i][1, 0], thymio_measures[i][2, 0]])
        state.append([thymio_state_record[i][0, 0], thymio_state_record[i][1, 0], thymio_state_record[i][2, 0]])

    plt.plot([x[0] for x in measure], label="measured pos X")
    plt.plot([x[0] for x in state], label="filtered pos X")
    plt.plot([x[1] for x in measure], label="measured pos Y")
    plt.plot([x[1] for x in state], label="filtered pos Y")
    plt.plot([x[2] for x in measure], label="measured angle")
    plt.plot([x[2] for x in state], label="filtered angle")

    plt.xlabel("Time step")
    plt.ylabel("Thymio position X, Y and angle")
    plt.legend()


def plot_speed(start, end, thymio_measures, thymio_state_record):

    measure_speed = []
    state_speed = []

    for i in range(start, end):
        measure_speed.append([thymio_measures[i][3, 0], thymio_measures[i][4, 0]])
        state_speed.append([thymio_state_record[i][3, 0], thymio_state_record[i][4, 0]])

    plt.plot([x[0] for x in measure_speed], label="measured speed")
    plt.plot([x[0] for x in state_speed], label="filtered speed")
    plt.plot([x[1] for x in measure_speed], label="measured rot speed")
    plt.plot([x[1] for x in state_speed], label="filtered rot speed")

    plt.xlabel("Time step")
    plt.ylabel("Thymio position speed and rotational speed")
    plt.legend()