from src.core.drawing import DrawingRecognizer

if __name__ == "__main__":
    app = DrawingRecognizer("models/hand_draw_model.pth")
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()