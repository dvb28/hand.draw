from torch import cuda

DATA_PACK = 5000
# Model arguments
EPOCHS = 50
BATCH_SIZE = 32
OPTIM = {"alg": "adam", "lr": 1e-4, "momentum": 0.5}
SAVE_FOLDER = "models"
DEVICE = "cuda:0" if cuda.is_available() else "cpu"

# Classes
CLASSES = {
     "airplane": "máy bay", "ant": "con kiến", "apple": "quả táo", "axe": "cái búa", "banana": "quả chuỗi", "barn": "chuồng trại", "baseball": "bóng chày", "basket": "rổ", "basketball": "bóng rổ", "bat": "con dơi", "bird": "con chim"
}
