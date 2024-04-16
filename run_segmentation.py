from segmentation.segment.predict import run

def start():
    resp = run(weights='/home/sumanthreddy/Desktop/Drdo_poc/segmentation/yolov7-seg.pt',
        # name='/home/sumanthreddy/Desktop/test.mp4',
        source="/home/sumanthreddy/Desktop/test.mp4",
        data = "/home/sumanthreddy/Desktop/Drdo_poc/segmentation/data/coco.yaml",)

start()