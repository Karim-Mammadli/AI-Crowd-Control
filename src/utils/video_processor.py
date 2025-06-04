import cv2  # type: ignore
import threading
import time
from datetime import datetime

class VideoProcessor:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_ready = False
        
        print(f"🎥 VideoProcessor initialized (camera ID: {camera_id})")
    
    def start_capture(self):
        """Start video capture with better error handling."""
        try:
            print(f"🔄 Attempting to start camera {self.camera_id}...")
            
            # Release any existing capture
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.5)
            
            # Try different backends if default fails
            backends_to_try = [
                cv2.CAP_DSHOW,    # DirectShow (Windows)
                cv2.CAP_MSMF,     # Microsoft Media Foundation (Windows)
                cv2.CAP_ANY,      # Any available backend
            ]
            
            for backend in backends_to_try:
                print(f"🔄 Trying backend: {backend}")
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                
                if self.cap.isOpened():
                    print(f"✅ Camera opened with backend {backend}")
                    break
                else:
                    print(f"❌ Failed with backend {backend}")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
            
            if not self.cap or not self.cap.isOpened():
                print("❌ All backends failed, trying basic VideoCapture...")
                self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_id}")
            
            # Configure camera settings
            print("🔧 Configuring camera settings...")
            
            # Set resolution (try multiple options)
            resolutions = [
                (640, 480),   # VGA
                (1280, 720),  # HD
                (320, 240),   # QVGA
            ]
            
            for width, height in resolutions:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Verify settings
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if actual_width > 0 and actual_height > 0:
                    print(f"✅ Resolution set to: {actual_width}x{actual_height}")
                    break
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"📹 FPS set to: {actual_fps}")
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Camera warm-up
            print("⏳ Warming up camera...")
            time.sleep(2.0)    # Give camera 2 seconds to initialize
            for _ in range(10):   # Flush initial frames
                self.cap.read()
            
            # Test frame capture
            print("🧪 Testing frame capture...")
            time.sleep(1.0)  # Add 1 second delay for camera to warm up

            for attempt in range(10):  # Increase attempts to 10
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Test frame captured: {test_frame.shape}")
                    # ADD: Flush camera buffer
                    for _ in range(5):
                        self.cap.grab()  # clear buffer
                    self.is_running = True
                    self.frame_ready = True
                    print(f"✅ Camera started successfully (ID: {self.camera_id})")
                    return True
                else:
                    print(f"⚠️ Test attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.5)  # retry delay is now 0.5 seconds
            
            raise Exception("Camera test failed - no frames captured")
            
        except Exception as e:
            print(f"❌ Camera startup error: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_running = False
            self.frame_ready = False
            return False
    
    def stop_capture(self):
        """Stop video capture."""
        print("🛑 Stopping camera capture...")
        self.is_running = False
        self.frame_ready = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            print("✅ Camera stopped")
    
    def get_frame(self):
        """Get current frame with improved error handling."""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return None
        
        try:
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                # Try to re-read a few times
                for retry in range(3):
                    time.sleep(0.01)
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break
                
                if not ret or frame is None:
                    print("⚠️ Failed to read frame from camera")
                    return None
            
            # Update current frame
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            return frame
            
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            return None
    
    def get_frame_info(self):
        """Get frame dimensions and info."""
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            return {'width': width, 'height': height, 'fps': fps}
        return None
    
    def is_camera_working(self):
        """Check if camera is working properly."""
        return self.is_running and self.frame_ready and self.cap is not None and self.cap.isOpened()