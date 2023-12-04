from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO()
    model.train(batch=32, imgsz=640, data="settings.yaml",
                workers=4, epochs=500, cache=True, patience=0, task='detect')
