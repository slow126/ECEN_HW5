import cv2
import numpy as np
import os

cap = cv2.VideoCapture("IMG_0069.MOV")
count = 0
SAVE_FRAMES = False
PATH = "video_frames"
FILE_LIST = os.listdir(PATH)
FILE_LIST.sort()

SHOW_FRAME = False
STEP = 1
WINDOW = 15

lk_params = dict(winSize=(21, 21),
                  maxLevel=0,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def find_template(next_frame, prev_frame, corners):
    nextPts = []
    good_corners = []
    for i in range(len(corners)):
        corner = np.squeeze(corners[i])
        if(corner[0] > 20 and corner[1] > 20 and corner[1] < next_frame.shape[0] - 20 and corner[0] < next_frame.shape[1] - 20):
            template = prev_frame[int(corner[1] - WINDOW):int(corner[1] + WINDOW), int(corner[0] - WINDOW):int(corner[0] + WINDOW)]
            # cv2.imshow("template", template)
            # cv2.waitKey()
            match = cv2.matchTemplate(next_frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
            max_loc = tuple(map(sum, zip(max_loc, (20, 20))))
            nextPts += [max_loc]
            good_corners += [corner]
            # cv2.imshow("temp", temp)
            # cv2.waitKey()
    return good_corners, nextPts


for j in range(len(FILE_LIST) - STEP):
    file = FILE_LIST[j]
    m = STEP
    frame = cv2.imread(os.path.join(PATH, file))
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, .1, 1)
    if SHOW_FRAME:
        for i in range(len(corners)):
            cv2.circle(frame, tuple(np.squeeze(corners[i])), 4, (255, 255, 0), -1)
        cv2.imshow("gray", frame)
        cv2.waitKey()

    next_file = FILE_LIST[j + m]
    next_frame = cv2.imread(os.path.join(PATH, next_file))
    next_frame = cv2.resize(next_frame, (int(next_frame.shape[1] / 2), int(next_frame.shape[0] / 2)))
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

    prev_corners, next_corners = find_template(next_gray, gray, corners)

    for i in range(len(prev_corners)):
        cv2.line(frame, tuple(np.squeeze(prev_corners[i])), tuple(np.squeeze(next_corners[i])), (255, 0, 255), thickness=1)
    cv2.imwrite("template/frame-step-" + str(STEP).zfill(3) + "-" + str(j).zfill(5) + ".jpg", frame)
    # cv2.imshow("frame", frame)
    # cv2.waitKey(1)
    x = 1