from torch import load, from_numpy, argmax
from src.core.constants import CLASSES, DEVICE
from src.core.deep import HandDrawCNN
from collections import deque
import mediapipe as mp
import numpy as np
import threading
import cv2

if __name__ == "__main__":
    # mediapipe style config
    config = {
        "l_thickness": 3,
        "l_color": (255, 255, 0),
        "t_font": 1,
        "t_color": (0, 0, 255),
        "t_thickness": 2,
        "width": 1280,
        "height": 720,
    }

    # mediapipe solutions module
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])

    # streaming state
    is_streaming = True
    lock = threading.Lock()

    # load model
    model = HandDrawCNN(num_classes=len(CLASSES)).to(DEVICE["kernel"])
    model.load_state_dict(load("models/hand_draw_model.pth"))
    model.eval()

    # define mediapipe hand detection
    with mp_hands.Hands(
            max_num_hands=1,  # Chỉ phát hiện 1 bàn tay
            model_complexity=0,  # Độ phức tạp của mô hình
            min_detection_confidence=0.5,  # Ngưỡng tin cậy phát hiện
            min_tracking_confidence=0.5) as hands:  # Ngưỡng tin cậy theo dõi

        # create draw point
        points = deque(maxlen=512)

        # canvas (draw board)
        canvas = np.zeros((config["height"], config["width"], 3), dtype=np.uint8)

        prd_class = None
        is_drawing = False
        is_shown = False

        while cap.isOpened():
            with lock:
                if not is_streaming or cap is None:
                    break

            # read camera frame
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Lật frame theo chiều ngang

            if not success: continue

            # improve performance (set fase)
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # hand detection
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = hands.process(frame)

            # if the hand is detected
            if results.multi_hand_landmarks:
                for h_lmk in results.multi_hand_landmarks:
                    # Check if the fingers are straight
                    if h_lmk.landmark[8].y < h_lmk.landmark[7].y and h_lmk.landmark[12].y < \
                            h_lmk.landmark[11].y and h_lmk.landmark[16].y < h_lmk.landmark[15].y:
                        if len(points):
                            is_drawing = False
                            is_shown = True

                            # canvas image processing for prediction
                            canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                            canvas_gs = cv2.medianBlur(canvas_gs, 9)
                            canvas_gs = cv2.GaussianBlur(canvas_gs, (5, 5), 0)

                            # find the drawing container
                            ys, xs = np.nonzero(canvas_gs)
                            if len(ys) and len(xs):
                                min_y, max_y, min_x, max_x = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
                                crp_image = canvas_gs[min_y:max_y, min_x: max_x]
                                crp_image = cv2.resize(crp_image, (28, 28))

                                # prepare data for model
                                crp_image = np.array(crp_image, dtype=np.float32)[None, None, :, :]
                                crp_image = from_numpy(crp_image)
                                crp_image = crp_image.to(DEVICE["kernel"])

                                # predict
                                logits = model(crp_image)
                                prd_cls_idx = argmax(logits, dim=1)
                                prd_class = CLASSES[prd_cls_idx]

                                # reset canvas, point
                                points = deque(maxlen=512)
                                canvas = np.zeros((config["height"], config["width"], 3), dtype=np.uint8)
                    else:
                        # if drawing, add points to the list
                        is_drawing = True
                        is_shown = False

                        # add points
                        points.append(
                            (int(h_lmk.landmark[8].x * frame.shape[1]),
                             int(h_lmk.landmark[8].y * frame.shape[0]
                        )))

                        # draw a line connecting the points
                        for i in range(1, len(points)):
                            # Vẽ đường trên frame hiển thị
                            cv2.line(frame, points[i - 1], points[i], config["l_color"], config["l_thickness"])
                            # Vẽ đường trên canvas (bảng vẽ)
                            cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), config["l_thickness"])

                    # draw a line connecting the points
                    mp_drawing.draw_landmarks(
                        frame,
                        h_lmk,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # show prediction results
                    if not is_drawing and is_shown:
                        cv2.putText(frame, f'Class: {prd_class}', (500, 50),
                                    cv2.FONT_HERSHEY_COMPLEX, config["t_font"],
                                    config["t_color"], config["t_thickness"], cv2.LINE_AA)
            # display
            cv2.imshow("Hand Draw", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
