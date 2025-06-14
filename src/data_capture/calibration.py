import ArducamDepthCamera as ac
import cv2
import numpy as np
import mediapipe as mp
import time
from camera_calibration import CameraCalibrationManager

# --- Configuration ---
# ToF Camera
CAMERA_RANGE_MM = 4000.0
ROTATE_IMAGE = True
TOF_CONFIDENCE_THRESHOLD = 30
TOF_FRAME_TYPE = ac.FrameType.DEPTH # Using DEPTH as requested

# RGB Webcam
WEBCAM_INDEX = 8 # <<<--- IMPORTANT: Change this to your RGB webcam's actual index!
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# MediaPipe Pose Configuration
MP_MODEL_COMPLEXITY = 1
MP_MIN_DETECTION_CONFIDENCE = 0.75
MP_MIN_TRACKING_CONFIDENCE = 0.75

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class DualCameraPoseMapper:
    def __init__(self):
        self.tof_cam = ac.ArducamCamera()
        self.tof_range_mm = float(CAMERA_RANGE_MM)
        self.tof_confidence_threshold = TOF_CONFIDENCE_THRESHOLD
        self.tof_device_info = None
        self.show_confidence_map_window = True

        self.webcam = None
        self.webcam_width = WEBCAM_WIDTH
        self.webcam_height = WEBCAM_HEIGHT

        self.mp_pose_estimator = mp_pose.Pose(
            static_image_mode=False, model_complexity=MP_MODEL_COMPLEXITY,
            smooth_landmarks=True, enable_segmentation=False,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE)
        print(f"MediaPipe Pose initialized.")

        # Initialize camera calibration manager instead of real-time ArUco detection
        self.calibration_manager = CameraCalibrationManager("camera_calibration.json")
        if self.calibration_manager.is_loaded:
            print("✅ Pre-computed camera calibration loaded successfully")
            cal_info = self.calibration_manager.get_calibration_info()
            if cal_info and 'calibration_quality' in cal_info:
                quality = cal_info['calibration_quality']
                print(f"📊 Calibration quality: {quality.get('reprojection_error', 'N/A'):.3f}px error, {quality.get('coverage_percentage', 'N/A'):.1f}% coverage")
        else:
            print("⚠️  No pre-computed calibration found - landmarks will not be transformed to ToF coordinates")
            print("   Please run the ArUco calibration setup first to generate camera_calibration.json")

        # Landmark transformation tracking
        self.last_transformation_stats = {
            'total_landmarks': 0,
            'valid_landmarks': 0,
            'transformation_success_rate': 0.0
        }

    def on_confidence_threshold_changed(self, value_from_trackbar): # Not used currently
        self.tof_confidence_threshold = value_from_trackbar

    def setup_cameras(self):
        print("Opening ToF camera...")
        ret_open_tof = self.tof_cam.open(ac.Connection.CSI, 0)
        if ret_open_tof != 0:
            raise RuntimeError(f"Failed to open ToF camera (error code: {ret_open_tof}).")
        print("ToF camera opened.")
        try:
            self.tof_device_info = self.tof_cam.getCameraInfo()
            if self.tof_device_info:
                print(f"ToF Info: {self.tof_device_info.name}, {self.tof_device_info.width}x{self.tof_device_info.height}, Type: {self.tof_device_info.device_type}")
        except Exception as e:
            print(f"Warning: ToF camera info error: {e}")

        try:
            self.tof_cam.setControl(ac.Control.RANGE, int(self.tof_range_mm))
            current_range = self.tof_cam.getControl(ac.Control.RANGE)
            if current_range > 0: self.tof_range_mm = float(current_range)
            print(f"ToF range set to: {self.tof_range_mm}mm")
        except Exception as e:
            print(f"Warning: ToF range control error: {e}")

        ret_start_tof = self.tof_cam.start(TOF_FRAME_TYPE)
        if ret_start_tof != 0:
            self.tof_cam.close()
            raise RuntimeError(f"Failed to start ToF stream (type {TOF_FRAME_TYPE}, error {ret_start_tof})")
        print(f"ToF camera started with FrameType: {TOF_FRAME_TYPE}.")

        print(f"Opening RGB Webcam (Index: {WEBCAM_INDEX})...")
        self.webcam = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open RGB Webcam at index {WEBCAM_INDEX}.")
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
        actual_w = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self.webcam_width or actual_h != self.webcam_height:
            print(f"Webcam actual resolution {actual_w}x{actual_h}, updating.")
            self.webcam_width, self.webcam_height = actual_w, actual_h
        print(f"RGB Webcam opened: {self.webcam_width}x{self.webcam_height}")

    def _ensure_240x180_bgr(self, frame, source_name="Unknown"):
        if frame is None:
            # print(f"Warning: Frame from {source_name} is None, returning black 240x180 BGR.")
            return np.zeros((180, 240, 3), dtype=np.uint8)
        
        # Rotate if needed (original processing functions might do this, ensure consistency)
        # This function assumes frame is ALREADY rotated if ROTATE_IMAGE is true globally.

        # Ensure 8-bit
        if frame.dtype != np.uint8:
            # print(f"Normalizing {source_name} (dtype: {frame.dtype}) to uint8.")
            cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)

        # Ensure BGR
        if len(frame.shape) == 2 or frame.shape[2] == 1: # Grayscale
            # print(f"Converting {source_name} from grayscale to BGR.")
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] != 3:
            print(f"Warning: {source_name} has unexpected channel count {frame.shape[2]}. Attempting to use as is.")
            return np.zeros((180, 240, 3), dtype=np.uint8)


        # Ensure 240x180 (width x height for OpenCV resize)
        if frame.shape[1] != 240 or frame.shape[0] != 180:
            # print(f"Resizing {source_name} from {frame.shape[1]}x{frame.shape[0]} to 240x180.")
            frame = cv2.resize(frame, (240, 180)) # (width, height) for cv2.resize
        return frame

    def _ensure_240x180_gray(self, frame, source_name="Unknown"):
        if frame is None:
            # print(f"Warning: Frame from {source_name} is None, returning black 240x180 Gray.")
            return np.zeros((180, 240), dtype=np.uint8)

        # Rotate if needed (original processing functions might do this)
        # This function assumes frame is ALREADY rotated if ROTATE_IMAGE is true globally.

        # Ensure 8-bit
        if frame.dtype != np.uint8:
            # print(f"Normalizing {source_name} (dtype: {frame.dtype}) to uint8.")
            cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)

        # Ensure Grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # print(f"Converting {source_name} from BGR to Gray.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 3 and frame.shape[2] != 1: # Other multi-channel
             print(f"Warning: {source_name} has unexpected channel count {frame.shape[2]} for gray conversion.")
             return np.zeros((180, 240), dtype=np.uint8)


        # Ensure 240x180
        if frame.shape[1] != 240 or frame.shape[0] != 180:
            # print(f"Resizing {source_name} from {frame.shape[1]}x{frame.shape[0]} to 240x180.")
            frame = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_NEAREST)
        return frame

    def process_depth_to_bgr_display(self, depth_buf_mm):
        if depth_buf_mm is None or depth_buf_mm.size == 0:
            return self._ensure_240x180_bgr(None, "Depth (fallback)")
            
        depth_buf_clamped = np.clip(depth_buf_mm, 0.0, self.tof_range_mm)
        depth_normalized = (depth_buf_clamped / max(1.0, self.tof_range_mm)) * 255.0 # Avoid div by zero
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        if ROTATE_IMAGE:
            depth_uint8 = cv2.rotate(depth_uint8, cv2.ROTATE_180)
        return self._ensure_240x180_bgr(depth_uint8, "Depth")

    def process_amplitude_to_bgr_display(self, amplitude_buf):
        if amplitude_buf is None or amplitude_buf.size == 0:
            return None # Signal to fallback
        
        amp_proc = amplitude_buf.copy()
        if ROTATE_IMAGE:
            amp_proc = cv2.rotate(amp_proc, cv2.ROTATE_180)
        return self._ensure_240x180_bgr(amp_proc, "Amplitude")

    def process_confidence_for_aruco(self, confidence_buf):
        if confidence_buf is None or confidence_buf.size == 0:
            return None

        conf_proc = confidence_buf.copy()
        if ROTATE_IMAGE:
            conf_proc = cv2.rotate(conf_proc, cv2.ROTATE_180)
        return self._ensure_240x180_gray(conf_proc, "Confidence")

    def transform_landmarks_to_tof(self, rgb_landmarks):
        """Transform RGB landmarks to ToF coordinates using pre-computed calibration."""
        if not self.calibration_manager.is_loaded:
            return [], []
        
        transformed_landmarks, valid_indices = self.calibration_manager.transform_landmarks_rgb_to_tof(rgb_landmarks)
        
        # Update transformation statistics
        self.last_transformation_stats = {
            'total_landmarks': len(rgb_landmarks),
            'valid_landmarks': len(transformed_landmarks),
            'transformation_success_rate': len(transformed_landmarks) / len(rgb_landmarks) if len(rgb_landmarks) > 0 else 0.0
        }
        
        return transformed_landmarks, valid_indices

    def run(self):
        self.setup_cameras()

        main_tof_window_name = "ToF Output with Mapped Landmarks"
        rgb_window_name = "RGB Webcam with MediaPipe Pose"
        confidence_window_name = "ToF Confidence Map (for ArUco)"

        cv2.namedWindow(main_tof_window_name, cv2.WINDOW_NORMAL) # Use NORMAL for resizability
        cv2.resizeWindow(main_tof_window_name, 240*2, 180*2) # Make it a bit bigger
        if self.show_confidence_map_window:
            cv2.namedWindow(confidence_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(confidence_window_name, 240*2, 180*2)
        cv2.namedWindow(rgb_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(rgb_window_name, self.webcam_width//2, self.webcam_height//2)


        print("\nStarting dual camera processing. Press 'q' to quit. Press 'c' to reset calibration.")
        
        prev_time_main_loop = time.time()

        try:
            while True:
                loop_start_time = time.time()

                # --- ToF Camera Frame ---
                frame_data_tof = self.tof_cam.requestFrame(100) # Reduced timeout
                tof_depth_buf_mm = None
                tof_amplitude_buf = None
                tof_confidence_buf = None
                display_source_text = "No ToF Data"

                if frame_data_tof:
                    tof_depth_buf_mm = frame_data_tof.depth_data
                    if hasattr(frame_data_tof, 'amplitude_data') and frame_data_tof.amplitude_data is not None:
                        tof_amplitude_buf = frame_data_tof.amplitude_data
                    if hasattr(frame_data_tof, 'confidence_data') and frame_data_tof.confidence_data is not None:
                        tof_confidence_buf = frame_data_tof.confidence_data
                
                # --- RGB Webcam Frame ---
                ret_rgb, rgb_frame_original = self.webcam.read()
                if not ret_rgb:
                    print("Error: Failed to grab RGB webcam frame.")
                    time.sleep(0.01) # Avoid busy-looping if webcam fails
                    if frame_data_tof: self.tof_cam.releaseFrame(frame_data_tof)
                    continue
                
                # --- Prepare ToF display image (Prioritize Amplitude, fallback to Depth Vis) ---
                tof_display_bgr = None
                used_amplitude_for_display = False
                if tof_amplitude_buf is not None:
                    tof_display_bgr = self.process_amplitude_to_bgr_display(tof_amplitude_buf)
                    if tof_display_bgr is not None:
                        used_amplitude_for_display = True
                        display_source_text = "ToF Amplitude"
                
                if tof_display_bgr is None: 
                    tof_display_bgr = self.process_depth_to_bgr_display(tof_depth_buf_mm)
                    display_source_text = "ToF Depth Viz"
                
                # --- Prepare ToF Confidence Map for ArUco ---
                processed_tof_confidence_map = self.process_confidence_for_aruco(tof_confidence_buf)
                
                if self.show_confidence_map_window and processed_tof_confidence_map is not None:
                    cv2.imshow(confidence_window_name, processed_tof_confidence_map)

                # --- MediaPipe Pose on RGB Webcam ---
                rgb_frame_for_mp = cv2.cvtColor(rgb_frame_original, cv2.COLOR_BGR2RGB)
                rgb_frame_for_mp.flags.writeable = False
                mp_results = self.mp_pose_estimator.process(rgb_frame_for_mp)
                # rgb_frame_for_mp.flags.writeable = True # No longer needed as we draw on a copy
                
                annotated_rgb_frame = rgb_frame_original.copy()
                mp_landmarks_2d_rgb_pixels = []

                if mp_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_rgb_frame, landmark_list=mp_results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    for landmark_idx, landmark in enumerate(mp_results.pose_landmarks.landmark):
                        if landmark.visibility < 0.5 : continue # Skip low visibility landmarks for mapping
                        lx = landmark.x * self.webcam_width
                        ly = landmark.y * self.webcam_height
                        mp_landmarks_2d_rgb_pixels.append((lx, ly))

                    # --- Transform MediaPipe Landmarks to ToF Coordinates ---
                    if mp_landmarks_2d_rgb_pixels and self.calibration_manager.is_loaded:
                        transformed_landmarks, valid_indices = self.transform_landmarks_to_tof(mp_landmarks_2d_rgb_pixels)
                        
                        # Draw transformed landmarks on ToF display
                        for i, (x, y) in enumerate(transformed_landmarks):
                            x, y = int(x), int(y)
                            if 0 <= x < tof_display_bgr.shape[1] and 0 <= y < tof_display_bgr.shape[0]:
                                cv2.circle(tof_display_bgr, (x, y), 3, (0, 255, 0), -1)  # Green circle
                                # Optional: show landmark index
                                # cv2.putText(tof_display_bgr, str(valid_indices[i]), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,255,200), 1)
                        
                        # Display transformation statistics
                        stats = self.last_transformation_stats
                        if stats['total_landmarks'] > 0:
                            success_rate = stats['transformation_success_rate'] * 100
                            status_text = f"Landmarks: {stats['valid_landmarks']}/{stats['total_landmarks']} ({success_rate:.1f}%)"
                            cv2.putText(tof_display_bgr, status_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    elif mp_landmarks_2d_rgb_pixels and not self.calibration_manager.is_loaded:
                        # Show warning when landmarks detected but no calibration available
                        warning_text = "No calibration - landmarks not transformed"
                        cv2.putText(tof_display_bgr, warning_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

                # --- Add Display Source Text and FPS ---
                cv2.putText(tof_display_bgr, display_source_text, (5, tof_display_bgr.shape[0] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                current_loop_time = time.time()
                loop_fps = 1.0 / (current_loop_time - prev_time_main_loop + 1e-6) # Add epsilon to avoid div by zero
                prev_time_main_loop = current_loop_time
                cv2.putText(tof_display_bgr, f"MainLoop FPS: {int(loop_fps)}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(annotated_rgb_frame, f"MainLoop FPS: {int(loop_fps)}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                # --- Show Windows ---
                cv2.imshow(main_tof_window_name, tof_display_bgr)
                cv2.imshow(rgb_window_name, annotated_rgb_frame)
                
                if frame_data_tof:
                    self.tof_cam.releaseFrame(frame_data_tof)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('c'):
                    # Reload calibration
                    print("INFO: Reloading calibration...")
                    self.calibration_manager.load_calibration()
                    if self.calibration_manager.is_loaded:
                        print("✅ Calibration reloaded successfully")
                    else:
                        print("❌ Failed to reload calibration")

        finally:
            print("Stopping cameras and cleaning up...")
            if hasattr(self, 'mp_pose_estimator') and self.mp_pose_estimator:
                self.mp_pose_estimator.close()
            if hasattr(self, 'tof_cam') and self.tof_cam.isOpened():
                self.tof_cam.stop()
                self.tof_cam.close()
            if self.webcam and self.webcam.isOpened():
                self.webcam.release()
            cv2.destroyAllWindows()
            print("Cleanup complete.")

if __name__ == "__main__":
    print("Starting Dual Camera Pose Mapper with Pre-computed Calibration...")
    print("Ensure your Arducam ToF camera is connected via CSI.")
    print(f"Ensure your RGB Webcam is connected and accessible at index: {WEBCAM_INDEX}")
    print("Calibration file: camera_calibration.json (run ArUco calibration setup if missing)")
    print("Press 'q' to quit, 'c' to reload calibration")
    print("---")
    try:
        estimator = DualCameraPoseMapper()
        estimator.run()
    except RuntimeError as e:
        print(f"FATAL: Initialization or runtime error: {e}")
    except Exception as e:
        print(f"FATAL: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()