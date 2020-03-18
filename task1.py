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
                cv2.imshow("frame", frame)
                # cv2.imwrite("video_frames/Frame-" + str(count).zfill(5)+".jpg", frame)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()


for file in FILE_LIST:
    frame = cv2.imread(os.path.join(PATH, file))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.1, 1)
    for i in range(len(corners)):
        cv2.circle(frame, tuple(np.squeeze(corners[i])), 4, (255, 255, 0), -1)
    cv2.imshow("gray", frame)
    cv2.waitKey()
    x = 1