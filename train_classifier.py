from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8n-cls.pt')
    model.train(data="directory", workers=4, epochs=500, cache=True,
                patience=0, task='classify')
