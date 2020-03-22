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
STEP = 60
WINDOW = 15
PT_THRESHOLD = 0.9
SCALE = 4

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
            match = cv2.matchTemplate(next_frame, (template), cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
            if max_val > PT_THRESHOLD:
                max_loc = tuple(map(sum, zip(max_loc, (20, 20))))
                nextPts += [max_loc]
                good_corners += [corner]
    return good_corners, nextPts

for j in range(len(FILE_LIST) - STEP):
    file = FILE_LIST[j]
    m = STEP
    frame = cv2.imread(os.path.join(PATH, file))
    frame = cv2.resize(frame, (int(frame.shape[1] / SCALE), int(frame.shape[0] / SCALE)))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 50, .1, 1)
    next_corners = np.zeros_like(corners)
    if SHOW_FRAME:
        for i in range(len(corners)):
            cv2.circle(frame, tuple(np.squeeze(corners[i])), 4, (255, 255, 0), -1)
        cv2.imshow("gray", frame)
        cv2.waitKey()

    for i in range(m):
        next_file = FILE_LIST[j + i]
        next_frame = cv2.imread(os.path.join(PATH, next_file))
        next_frame = cv2.resize(next_frame, (int(next_frame.shape[1] / SCALE), int(next_frame.shape[0] / SCALE)))
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        prev_corners, next_corners = find_template(next_gray, gray, corners)
        retval, mask = cv2.findFundamentalMat(np.array(prev_corners), np.array(next_corners), cv2.FM_RANSAC, 3, 0.9)
        gray = next_gray
        prev_corners = prev_corners * mask


    print(len(next_corners))
    for i in range(len(prev_corners)):
        if(tuple(prev_corners[i]) != (0,0) and tuple(next_corners[i]) != (0,0)):
            cv2.line(frame, tuple(np.squeeze(prev_corners[i])), tuple(np.squeeze(next_corners[i])), (255, 0, 255), thickness=1)
    cv2.imwrite("inter_frame/frame-step-" + str(STEP).zfill(3) + "-" + str(j).zfill(5) + ".jpg", frame)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    x = 1