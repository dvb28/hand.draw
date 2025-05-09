from torch import cuda

DATA_PACK  = 1000
# Model arguments
EPOCHS = 20
BATCH_SIZE = 32
OPTIM = {"alg": "sgd", "lr": 1e-4, "momentum": 0.5}
SAVE_PATH = "models/hand_draw_model.pth"
DEVICE = {"kernel": "cuda:0", "name": "GPU"} if cuda.is_available() else {"kernel": "cpu", "name": "CPU"}

# Classes
CLASSES = ["airplane", "ant", "apple", "axe", "banana", "barn", "baseball", "basket", "basketball", "bat", "bird"]
