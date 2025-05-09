# Hand Draw Project
## Giới thiệu
Hand Draw sử dụng mô hình học sâu CNN để nhận diện các hình vẽ được tạo bằng tay trong thời gian thực. Sử dụng camera để theo dõi chuyển động của bàn tay người dùng, ghi lại đường vẽ và nhận diện hình dạng được tạo ra.

## Tổng quan
Các công cụ, thư viện chính sử dụng trong dự án
- Python 3.12
- MediaPipe
- OpenCV
- Pytorch

## Cách thức hoạt động
- Phát hiện bàn tay: Sử dụng MediaPipe để phát hiện và theo dõi vị trí của bàn tay trong khung hình camera.
- Ghi lại đường vẽ: Khi di chuyển ngón trỏ, ghi lại các điểm và vẽ các đường nối.
- Nhận diện: Sau khi hoàn thành đường vẽ, sử dụng mô hình CNN để nhận diện hình dạng.
- Hiển thị kết quả: Kết quả nhận class diện được hiển thị trên màn hình.

## Dataset
Sử dụng 10 class trong bộ
[**Quickdraw Datasets**](https://quickdraw.withgoogle.com/data) của Google

```
# Classes
CLASSES = ["airplane", "ant", "apple", "axe", "banana", "barn", "baseball", "basket", 
           "basketball", "bat", "bird"]
```