import logging
import numpy as np
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {}

        # Thêm các trường tùy chỉnh nếu có
        if hasattr(record, 'custom_field'):
            # Tạo một dictionary chứa thông tin log
            log_record = {
                **record.custom_field,
                'timestamp': self.formatTime(record, self.datefmt),
            }
        return json.dumps(log_record, ensure_ascii=False)


class Logger:
    def __init__(self):
        # Tạo logger
        self.logger = logging.getLogger('json_logger')
        self.logger.setLevel(logging.DEBUG)

        cr_time = datetime.now().strftime("%Y-%m-%d%H.%M.%S.%f")
        # Tạo handler
        file_handler = RotatingFileHandler(
            f"logs/{cr_time}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf8"
        )
        file_handler.setLevel(logging.DEBUG)

        # Thêm JSON formatter
        json_formatter = JSONFormatter()
        file_handler.setFormatter(json_formatter)

        self.logger.addHandler(file_handler)

    def log(self, epoch, epochs, per_batch, batch_size, optim, loss, accuracy):
        self.logger.info(
            f"Epoch:  - Batch: {per_batch} - Loss: {loss:.2f}",
            extra={
                "custom_field": {
                    "batch": per_batch,
                    "accuracy": np.round(float(loss), 2),
                    "loss": np.round(float(loss), 2),
                    "batch_size": batch_size,
                    "epoch": f"{epoch}/{epochs}",
                    "learning_rate": optim.param_groups[0]['lr'],
                    "momentum": optim.param_groups[0]['momentum']
                }
            }
        )
