import time
import numpy as np
import cv2
import pyrealsense2 as rs
from matplotlib import pyplot


from haitai import HaiTai
import platform
import time
from gaze_tracker import GAZE_TRACKER
#from Memory.gpu_mem_track import MemTracker


if __name__ == '__main__':
    
    gaze_tracker = GAZE_TRACKER(0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    

    while True:
        rval,frame = cap.read()
        time_s = time.time()   
        _,flag,focusing,attention_vec,face_center = gaze_tracker.run(frame, [], [],False)
        
        
        gaze_tracker.writer.write(gaze_tracker.visualizer.image)
        time_e = time.time()
        key = cv2.waitKey(int(1000/30-(time_e-time_s)))
        if gaze_tracker.key == ord('p') or key == ord('p'):
            while(1):
                if cv2.waitKey(5) == ord('s'):
                    break
        if gaze_tracker.key == ord('q') or key == ord('q'):
            break
        
    gaze_tracker.writer.release()
    cap.release()
    cv2.destroyAllWindows()
    # haitai.close()

