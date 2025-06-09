from .constants import DEVICE, CLASSES
from torch import load as load_model, from_numpy, no_grad, FloatTensor, argmax
from .deep import HandDrawCNN
from PIL import Image, ImageTk
from collections import deque
import customtkinter as ctkt
import mediapipe as mp
import numpy as np
import threading
import cv2

class DrawingInfo(ctkt.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.label = {"prd": None, "cfd": None}
        self.prd_t = self.cfd_t = None
        font = ctkt.CTkFont(
            family="Jetbrains Mono Medium",
            size=12,
        )

        for idx, var in enumerate(self.label.keys()):
            is_prd = var == "prd"
            temp = self.prd_t if is_prd else self.label[var]
            temp = ctkt.CTkLabel(self, font=font, compound="left", text=f"{"Class dự đoán" if is_prd else "Nội dung dự đoán"}")
            temp.grid(row=0, column=idx if is_prd else idx, padx=10, pady=0, sticky="w")

            self.label[var] = ctkt.CTkTextbox(
                height=10,
                width=245,
                master=self,
                border_width=1,
                corner_radius=5,
                state="disabled",
                border_color="gray",
                fg_color="#2b2b2b",
            )
            self.label[var].grid(
                row=1,
                padx=10,
                pady=(0, 10),
                sticky="nsew",
                column=idx if is_prd else idx,
            )

    def update_text(self, text):
        for var in self.label.keys():
            self.label[var].configure(state="normal")
            self.label[var].delete("0.0", "end")
            self.label[var].insert("0.0", text[var])
            self.label[var].configure(state="disabled")

class DrawingFrame(ctkt.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # create label for video display
        self.video = ctkt.CTkLabel(self, text="")
        self.video.grid(row=0, column=0, padx=10, pady=10)

    def update_frame(self, frame):
        # Convert frame to PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)

        # Update label with new frame
        self.video.configure(image=photo)
        self.video.image = photo

class DrawingRecognizer(ctkt.CTk):
    def __init__(self, model_path):
        super().__init__()
        # load model
        self.model = HandDrawCNN(len(list(CLASSES.keys()))).to(DEVICE).to(DEVICE)
        self.model.load_state_dict(load_model(model_path)["state_dict"])
        self.model.eval()
        
        # canvas
        self.config = {
            "l_thickness": 3,
            "l_color": (255, 255, 0),
            "t_font": 1,
            "t_color": (0, 0, 255),
            "t_thickness": 2,
            "width": 1280,
            "height": 720,
        }
        # create draw point
        self.points = deque(maxlen=512)
        self.canvas = np.zeros((self.config["height"], self.config["width"], 3), dtype=np.uint8)
        
        # streaming state
        self.is_streaming = True
        self.lock = threading.Lock()
        self.prd_dict = None
        self.is_drawing = False
        self.is_shown = False

        # config mediapipe hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Configure window
        self.title("Hand Draw Recognition")
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure(0, weight=1)

        # create frame
        self.frame = DrawingFrame(master=self)
        self.frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # create frame
        self.text = DrawingInfo(master=self)
        self.text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # initialize camera
        self.capture = cv2.VideoCapture(0)

        # start video update
        self.start_video()

    def start_video(self):
        if self.capture.isOpened():
            with self.lock:
                if not self.is_streaming or self.capture is None:
                    return
                
            ret, frame = self.capture.read()
            frame = cv2.flip(frame, 1)

            frame = cv2.bilateralFilter(frame, 5, 50, 100)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = self.hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y < \
                            hand_landmarks.landmark[11].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                        if len(self.points):
                            self.is_drawing = False
                            self.is_shown = True

                            # canvas image processing for prediction
                            canvas_gs = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
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
                                crp_image = crp_image.to(DEVICE)

                                # predict
                                logits = self.model(crp_image)
                                prd_cls_idx = argmax(logits, dim=1)
                                self.prd_dict = {
                                    "content": list(CLASSES.values())[prd_cls_idx],
                                    "class": list(CLASSES.keys())[prd_cls_idx]
                                }

                                # reset canvas, point
                                self.points = deque(maxlen=512)
                                self.canvas = np.zeros((self.config["height"], self.config["width"], 3), dtype=np.uint8)
                    else:
                        # if drawing, add points to the list
                        self.is_drawing = True
                        self.is_shown = False

                        # add points
                        self.points.append(
                            (int(hand_landmarks.landmark[8].x * frame.shape[1]),
                             int(hand_landmarks.landmark[8].y * frame.shape[0]
                        )))

                        # draw a line connecting the points
                        for i in range(1, len(self.points)):
                            # Vẽ đường trên frame hiển thị
                            cv2.line(frame, self.points[i - 1], self.points[i], self.config["l_color"], self.config["l_thickness"])
                            # Vẽ đường trên canvas (bảng vẽ)
                            cv2.line(self.canvas, self.points[i - 1], self.points[i], (255, 255, 255), self.config["l_thickness"])
                    
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(232, 254, 255), thickness=1, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(255, 249, 161), thickness=1, circle_radius=2)
                    )
                    # show prediction results
                    if not self.is_drawing and self.is_shown:
                        self.text.update_text({"prd": self.prd_dict["class"], "cfd": self.prd_dict["content"]})
            self.frame.update_frame(frame)
        # update after 10ms
        self.after(10, self.start_video)

    def on_closing(self):
        if self.capture.isOpened():
            self.capture.release()
        self.quit()

    def preprocess(self, landmarks):
        # Convert to tensor
        tensor = FloatTensor(landmarks)
        return tensor