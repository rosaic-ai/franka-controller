import cv2
import numpy as np
import pyrealsense2 as rs

class RSCapture():
    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False):
        self.name = name
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.align = rs.align(rs.stream.color)

    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return False, None
        image = np.asarray(color_frame.get_data())
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame:
                depth_image = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, (image, depth_image)
        return True, image

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()

# 카메라 생성 및 초기화
camera = RSCapture(name='wrist_1', serial_number='130322270132', depth=False)

try:
    while True:
        ret, frame = camera.read()
        if ret:
            cv2.imshow('RealSense', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No frame available")
except Exception as e:
    print(e)
finally:
    camera.close()
    cv2.destroyAllWindows()
