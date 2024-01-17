import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4
finger_fold_status = []

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Draw blue circles around fingertips
            for tip_id in finger_tips:
                tip_x, tip_y = int(lm_list[tip_id].x * w), int(lm_list[tip_id].y * h)
                cv2.circle(img, (tip_x, tip_y), 10, (255, 0, 0), cv2.FILLED)

            # Check if fingers are folded
            for tip_id in range(1, len(finger_tips)):
                if lm_list[finger_tips[tip_id]].x < lm_list[finger_tips[tip_id - 1]].x:
                    # Create green circle at fingertips
                    cv2.circle(img, (int(lm_list[finger_tips[tip_id]].x * w), int(lm_list[finger_tips[tip_id]].y * h)),
                               10, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Check if all fingers are folded
            if all(finger_fold_status):
                # Check if the thumb is raised up or down
                if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y:
                    print("LIKE")
                    cv2.putText(img, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    print("DISLIKE")
                    cv2.putText(img, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    mp_draw.draw_landmarks(img, hand_landmark,
                           mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                           mp_draw.DrawingSpec((0, 255, 0), 4, 2))
    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)