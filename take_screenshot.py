import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import imutils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    img = pyautogui.screenshot()
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = imutils.resize(img, width=500)
    img = cv2.flip(img, 1)

    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            is_fingers_folded = all(lm_list[i].y < lm_list[i-2].y for i in finger_tips)

            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2))
            finger_status = "Folded" if is_fingers_folded else "Open"
            cv2.putText(img, finger_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imwrite('hand_tracking.png', img)

    cv2.imshow("hand tracking", img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
