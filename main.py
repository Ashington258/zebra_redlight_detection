import cv2
import multiprocessing
from zebra_detection.zebra_crossing_detection import zebra_crossing_detection
from traffic_detection.red_light_detection import red_light_detection

def capture_frames(frame_queue):
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()

def run_zebra_detection(frame_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            zebra_crossing_detection(frame)

def run_red_light_detection(frame_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            red_light_detection(frame)

if __name__ == "__main__":
    frame_queue = multiprocessing.Queue(maxsize=10)

    p1 = multiprocessing.Process(target=capture_frames, args=(frame_queue,))
    p2 = multiprocessing.Process(target=run_zebra_detection, args=(frame_queue,))
    p3 = multiprocessing.Process(target=run_red_light_detection, args=(frame_queue,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
