import threading
from zebra_detection.zebra_crossing_detection import zebra_crossing_detection
from traffic_detection.red_light_detection import red_light_detection

def run_zebra_crossing_detection():
    zebra_crossing_detection()

def run_red_light_detection():
    red_light_detection()

if __name__ == "__main__":
    # 创建线程
    zebra_thread = threading.Thread(target=run_zebra_crossing_detection)
    red_light_thread = threading.Thread(target=run_red_light_detection)
    
    # 启动线程
    zebra_thread.start()
    red_light_thread.start()
    
    # 等待所有线程完成
    zebra_thread.join()
    red_light_thread.join()
