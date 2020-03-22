import cv2
import numpy as np
import os

cap = cv2.VideoCapture("IMG_0069.MOV")
count = 0
SAVE_FRAMES = False
PATH = "video_frames"
FILE_LIST = os.listdir(PATH)
FILE_LIST.sort()

if SAVE_FRAMES:
    if cap.isOpened() == False:
        print("Failed to open")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            count += 1
            if ret:
                # cv2.imshow("frame", frame)
                cv2.imwrite("video_frames/Frame-" + str(count).zfill(5)+".jpg", frame)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()


SHOW_FRAME = False
STEP = 30
PT_THRESHOLD = 30
SCALE = 4

lk_params = dict(winSize=(21, 21),
                  maxLevel=0,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                  # minEigThreshold=10
                 )


for j in range(len(FILE_LIST) - STEP):
    file = FILE_LIST[j]
    m = STEP
    frame = cv2.imread(os.path.join(PATH, file))
    frame = cv2.resize(frame, (int(frame.shape[1]/SCALE), int(frame.shape[0]/SCALE)))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 1000, .1, 1)
    if SHOW_FRAME:
        for i in range(len(corners)):
            cv2.circle(frame, tuple(np.squeeze(corners[i])), 4, (255, 255, 0), -1)
        cv2.imshow("gray", frame)
        cv2.waitKey()

    next_file = FILE_LIST[j + m]
    next_frame = cv2.imread(os.path.join(PATH, next_file))
    next_frame = cv2.resize(next_frame, (int(next_frame.shape[1] / SCALE), int(next_frame.shape[0] / SCALE)))
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

    next_corners, status, err = cv2.calcOpticalFlowPyrLK(gray, next_gray, corners, None, **lk_params)
    goodPts = cv2.threshold(err, PT_THRESHOLD, 1, cv2.THRESH_BINARY_INV)
    print(np.sum(goodPts[1]))
    for i in range(len(corners)):
        if(goodPts[1][i] != 0):
            cv2.line(frame, tuple(np.squeeze(corners[i])), tuple(np.squeeze(next_corners[i])), (255, 0, 255), thickness=1)
    cv2.imshow("frame", frame)
    cv2.imwrite('LK/frame-step-' + str(STEP).zfill(3) + '-' + str(i).zfill(5)+'.jpg', frame)
    # cv2.waitKey(1)
    x = 1