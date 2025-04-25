from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="datasets\ppe\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
