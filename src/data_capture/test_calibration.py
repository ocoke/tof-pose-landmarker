import ArducamDepthCamera as ac
import cv2
import numpy as np
import mediapipe as mp
import time

# --- Configuration ---
# ToF Camera
CAMERA_RANGE_MM = 4000.0
ROTATE_IMAGE = True
TOF_CONFIDENCE_THRESHOLD = 30 # For potential filtering, not directly used for display choice here
TOF_FRAME_TYPE = ac.FrameType.DEPTH # Using DEPTH as requested

# RGB Webcam
WEBCAM_INDEX = 8 # <<<--- IMPORTANT: Change this to your RGB webcam's actual index!
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# MediaPipe Pose Configuration
MP_MODEL_COMPLEXITY = 1
MP_MIN_DETECTION_CONFIDENCE = 0.75
MP_MIN_TRACKING_CONFIDENCE = 0.75

# ArUco Marker Configuration
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_6X6_250
ARUCO_MARKER_SIZE_METERS = 0.18
ARUCO_CALIBRATION_MARKER_ID = 777

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class DualCameraPoseMapper:
    def __init__(self):
        self.tof_cam = ac.ArducamCamera()
        self.tof_range_mm = float(CAMERA_RANGE_MM)
        # self.tof_confidence_threshold = TOF_CONFIDENCE_THRESHOLD # Not directly used for display choice
        self.tof_device_info = None
        self.show_confidence_map_window = True
        self.show_amplitude_map_window = True # New: Option to see amplitude separately

        self.webcam = None
        self.webcam_width = WEBCAM_WIDTH
        self.webcam_height = WEBCAM_HEIGHT

        self.mp_pose_estimator = mp_pose.Pose(
            static_image_mode=False, model_complexity=MP_MODEL_COMPLEXITY,
            smooth_landmarks=True, enable_segmentation=False,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE)
        print(f"MediaPipe Pose initialized.")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            print("Using cv2.aruco.DetectorParameters_create()")
        except AttributeError:
            print("cv2.aruco.DetectorParameters_create() not found, using cv2.aruco.DetectorParameters() (older OpenCV compatibility).")
            self.aruco_params = cv2.aruco.DetectorParameters()
        print(f"ArUco initialized with dictionary ID: {ARUCO_DICTIONARY_ID}")

        self.homography_rgb_to_tof = None
        self.last_calibration_time = 0
        self.calibration_interval = 2

        tof_fx, tof_fy = 200.0, 200.0
        tof_cx, tof_cy = 240 / 2.0, 180 / 2.0
        self.tof_camera_matrix = np.array([[tof_fx, 0, tof_cx], [0, tof_fy, tof_cy], [0, 0, 1.0]], dtype=np.float32)
        self.tof_dist_coeffs = np.zeros((5,1), dtype=np.float32)

        rgb_fx, rgb_fy = float(self.webcam_width * 0.8), float(self.webcam_width * 0.8)
        rgb_cx, rgb_cy = self.webcam_width / 2.0, self.webcam_height / 2.0
        self.rgb_camera_matrix = np.array([[rgb_fx, 0, rgb_cx], [0, rgb_fy, rgb_cy], [0, 0, 1.0]], dtype=np.float32)
        self.rgb_dist_coeffs = np.zeros((5,1), dtype=np.float32)

    # def on_confidence_threshold_changed(self, value_from_trackbar): # Not used for display choice
    #     self.tof_confidence_threshold = value_from_trackbar

    def setup_cameras(self):
        # (Setup code remains largely the same as your previous version, ensuring ToF starts with TOF_FRAME_TYPE)
        print("Opening ToF camera...")
        ret_open_tof = self.tof_cam.open(ac.Connection.CSI, 0)
        if ret_open_tof != 0: raise RuntimeError(f"Failed to open ToF camera (error code: {ret_open_tof}).")
        print("ToF camera opened.")
        try:
            self.tof_device_info = self.tof_cam.getCameraInfo()
            if self.tof_device_info: print(f"ToF Info: {self.tof_device_info.name}, {self.tof_device_info.width}x{self.tof_device_info.height}")
        except Exception as e: print(f"Warning: ToF camera info error: {e}")
        try:
            self.tof_cam.setControl(ac.Control.RANGE, int(self.tof_range_mm))
            current_range = self.tof_cam.getControl(ac.Control.RANGE)
            if current_range > 0: self.tof_range_mm = float(current_range)
            print(f"ToF range set to: {self.tof_range_mm}mm")
        except Exception as e: print(f"Warning: ToF range control error: {e}")
        ret_start_tof = self.tof_cam.start(TOF_FRAME_TYPE) # TOF_FRAME_TYPE is ac.FrameType.DEPTH
        if ret_start_tof != 0:
            self.tof_cam.close()
            raise RuntimeError(f"Failed to start ToF stream (type {TOF_FRAME_TYPE}, error {ret_start_tof})")
        print(f"ToF camera started with FrameType: {TOF_FRAME_TYPE}.")

        print(f"Opening RGB Webcam (Index: {WEBCAM_INDEX})...")
        self.webcam = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.webcam.isOpened(): raise RuntimeError(f"Failed to open RGB Webcam at index {WEBCAM_INDEX}.")
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
        actual_w = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self.webcam_width or actual_h != self.webcam_height:
            print(f"Webcam actual resolution {actual_w}x{actual_h}, updating.")
            self.webcam_width, self.webcam_height = actual_w, actual_h
        print(f"RGB Webcam opened: {self.webcam_width}x{self.webcam_height}")
        self.rgb_camera_matrix[0, 2] = self.webcam_width / 2.0
        self.rgb_camera_matrix[1, 2] = self.webcam_height / 2.0
        self.rgb_camera_matrix[0, 0] = float(self.webcam_width * 0.8)
        self.rgb_camera_matrix[1, 1] = float(self.webcam_width * 0.8)


    def _ensure_240x180_bgr(self, frame, source_name="Unknown"):
        if frame is None:
            return np.zeros((180, 240, 3), dtype=np.uint8)
        if frame.dtype != np.uint8:
            cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] != 3:
            return np.zeros((180, 240, 3), dtype=np.uint8)
        if frame.shape[1] != 240 or frame.shape[0] != 180: # width x height
            frame = cv2.resize(frame, (240, 180))
        return frame

    def _ensure_240x180_gray(self, frame, source_name="Unknown"):
        if frame is None:
            return np.zeros((180, 240), dtype=np.uint8)
        if frame.dtype != np.uint8:
            cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 3 and frame.shape[2] != 1:
             return np.zeros((180, 240), dtype=np.uint8)
        if frame.shape[1] != 240 or frame.shape[0] != 180: # width x height
            frame = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_NEAREST)
        return frame

    # This is your desired "Gray Scale Image" from preview.py
    def process_tof_depth_to_bgr_visualization(self, depth_buf_mm):
        if depth_buf_mm is None or depth_buf_mm.size == 0:
            return self._ensure_240x180_bgr(None, "Depth (fallback)") # Returns black 240x180 BGR

        depth_buf_clamped = np.clip(depth_buf_mm, 0.0, self.tof_range_mm)
        # Normalize to 0-255
        if self.tof_range_mm > 0:
            depth_normalized = (depth_buf_clamped / self.tof_range_mm) * 255.0
        else:
            # Fallback if range is zero
            # print("Warning: Camera range is zero. Using default for depth normalization.")
            depth_normalized = (depth_buf_clamped / CAMERA_RANGE_MM) * 255.0
        
        depth_uint8 = depth_normalized.astype(np.uint8)

        # Original preview.py rotated here, so we do it too before ensuring size etc.
        if ROTATE_IMAGE:
            depth_uint8 = cv2.rotate(depth_uint8, cv2.ROTATE_180)
        
        # This will convert to BGR and ensure 240x180
        return self._ensure_240x180_bgr(depth_uint8, "Depth Visualization")

    def process_amplitude_to_bgr_display(self, amplitude_buf): # For optional separate display
        if amplitude_buf is None or amplitude_buf.size == 0:
            return None
        amp_proc = amplitude_buf.copy()
        if ROTATE_IMAGE:
            amp_proc = cv2.rotate(amp_proc, cv2.ROTATE_180)
        return self._ensure_240x180_bgr(amp_proc, "Amplitude")

    def process_confidence_for_aruco_and_display(self, confidence_buf): # For ArUco and optional display
        if confidence_buf is None or confidence_buf.size == 0:
            return None
        conf_proc = confidence_buf.copy()
        if ROTATE_IMAGE:
            conf_proc = cv2.rotate(conf_proc, cv2.ROTATE_180)
        return self._ensure_240x180_gray(conf_proc, "Confidence")


    def detect_aruco(self, image_for_detection, camera_matrix, dist_coeffs):
        # (detect_aruco method remains the same as your previous version)
        if image_for_detection is None:
            return None, None, None, None, None
        gray_for_detection = image_for_detection
        if len(image_for_detection.shape) == 3 and image_for_detection.shape[2] == 3:
            gray_for_detection = cv2.cvtColor(image_for_detection, cv2.COLOR_BGR2GRAY)
        elif len(image_for_detection.shape) != 2: # Not grayscale, not BGR
             return None, None, None, None, None

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray_for_detection, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs = None, None
        if ids is not None:
            if camera_matrix is not None and dist_coeffs is not None:
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, ARUCO_MARKER_SIZE_METERS, camera_matrix, dist_coeffs)
                except: pass # Ignore errors if pose estimation fails
        return corners, ids, rejected, rvecs, tvecs

    def run(self):
        self.setup_cameras()

        main_tof_window_name = "ToF Depth Viz with Mapped Landmarks (240x180)" # Clarified name
        rgb_window_name = "RGB Webcam with MediaPipe Pose"
        confidence_window_name = "ToF Confidence Map (for ArUco)"
        amplitude_window_name = "ToF Amplitude Map (Optional)" # New window

        cv2.namedWindow(main_tof_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(main_tof_window_name, 240*2, 180*2)
        if self.show_confidence_map_window:
            cv2.namedWindow(confidence_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(confidence_window_name, 240*2, 180*2)
        if self.show_amplitude_map_window: # Create window for amplitude if shown
            cv2.namedWindow(amplitude_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(amplitude_window_name, 240*2, 180*2)
        cv2.namedWindow(rgb_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(rgb_window_name, self.webcam_width//2, self.webcam_height//2)

        print("\nStarting dual camera processing. Press 'q' to quit. Press 'c' to reset calibration.")
        
        prev_time_main_loop = time.time()

        try:
            while True:
                loop_start_time = time.time()

                frame_data_tof = self.tof_cam.requestFrame(100)
                tof_depth_buf_mm = None
                tof_amplitude_buf = None
                tof_confidence_buf = None

                if frame_data_tof:
                    tof_depth_buf_mm = frame_data_tof.depth_data
                    if hasattr(frame_data_tof, 'amplitude_data'): # Check if SDK provides it with FrameType.DEPTH
                        tof_amplitude_buf = frame_data_tof.amplitude_data
                    if hasattr(frame_data_tof, 'confidence_data'):
                        tof_confidence_buf = frame_data_tof.confidence_data
                
                ret_rgb, rgb_frame_original = self.webcam.read()
                if not ret_rgb:
                    print("Error: Failed to grab RGB webcam frame.")
                    time.sleep(0.01)
                    if frame_data_tof: self.tof_cam.releaseFrame(frame_data_tof)
                    continue
                
                # --- Main ToF Display: ALWAYS the depth visualization as per your request ---
                tof_display_bgr = self.process_tof_depth_to_bgr_visualization(tof_depth_buf_mm)
                display_source_text = "ToF Depth Viz" # This is now fixed
                
                # --- Process and optionally display Amplitude Map separately ---
                processed_tof_amplitude_bgr = None
                if tof_amplitude_buf is not None:
                    processed_tof_amplitude_bgr = self.process_amplitude_to_bgr_display(tof_amplitude_buf)
                    if self.show_amplitude_map_window and processed_tof_amplitude_bgr is not None:
                        cv2.imshow(amplitude_window_name, processed_tof_amplitude_bgr)
                
                # --- Process and display Confidence Map (used for ArUco) ---
                processed_tof_confidence_gray = self.process_confidence_for_aruco_and_display(tof_confidence_buf)
                if self.show_confidence_map_window and processed_tof_confidence_gray is not None:
                    cv2.imshow(confidence_window_name, processed_tof_confidence_gray)

                # --- MediaPipe Pose on RGB Webcam ---
                rgb_frame_for_mp = cv2.cvtColor(rgb_frame_original, cv2.COLOR_BGR2RGB)
                rgb_frame_for_mp.flags.writeable = False
                mp_results = self.mp_pose_estimator.process(rgb_frame_for_mp)
                
                annotated_rgb_frame = rgb_frame_original.copy()
                mp_landmarks_2d_rgb_pixels = []

                if mp_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_rgb_frame, landmark_list=mp_results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    for landmark_idx, landmark in enumerate(mp_results.pose_landmarks.landmark):
                        if landmark.visibility < 0.5: continue
                        lx = landmark.x * self.webcam_width
                        ly = landmark.y * self.webcam_height
                        mp_landmarks_2d_rgb_pixels.append((lx, ly))

                    # --- ArUco Based Calibration Attempt ---
                    current_time = time.time()
                    if current_time - self.last_calibration_time > self.calibration_interval or self.homography_rgb_to_tof is None:
                        if processed_tof_confidence_gray is not None: # Use processed confidence for ArUco
                            corners_rgb, ids_rgb, _, _, _ = self.detect_aruco(rgb_frame_original, self.rgb_camera_matrix, self.rgb_dist_coeffs)
                            corners_tof, ids_tof, _, _, _ = self.detect_aruco(processed_tof_confidence_gray, self.tof_camera_matrix, self.tof_dist_coeffs)

                            if ids_rgb is not None and ids_tof is not None:
                                common_ids_found = set(ids_rgb.flatten()).intersection(set(ids_tof.flatten()))
                                if ARUCO_CALIBRATION_MARKER_ID in common_ids_found:
                                    idx_rgb_list = np.where(ids_rgb.flatten() == ARUCO_CALIBRATION_MARKER_ID)[0]
                                    idx_tof_list = np.where(ids_tof.flatten() == ARUCO_CALIBRATION_MARKER_ID)[0]
                                    
                                    if len(idx_rgb_list) > 0 and len(idx_tof_list) > 0:
                                        idx_rgb, idx_tof = idx_rgb_list[0], idx_tof_list[0]
                                        rgb_marker_pts_for_H = corners_rgb[idx_rgb][0].astype(np.float32)
                                        tof_marker_pts_for_H = corners_tof[idx_tof][0].astype(np.float32)
                                        
                                        if len(rgb_marker_pts_for_H) == 4 and len(tof_marker_pts_for_H) == 4:
                                            H, status = cv2.findHomography(rgb_marker_pts_for_H, tof_marker_pts_for_H, cv2.RANSAC, 5.0)
                                            if H is not None:
                                                self.homography_rgb_to_tof = H
                                                self.last_calibration_time = current_time
                                                print(f"INFO: Homography calculated using ArUco marker ID {ARUCO_CALIBRATION_MARKER_ID}.")
                                            # else: print("DEBUG: Homography calculation failed.")
                                # else: print(f"DEBUG: Calib marker ID {ARUCO_CALIBRATION_MARKER_ID} not common or not found.")
                            # else: print("DEBUG: ArUco not detected in one/both views for calib.")
                        # else: print("DEBUG: ToF confidence map unavailable for ArUco calib.")
                
                # --- Map MediaPipe Landmarks to ToF Display Image (which is tof_display_bgr) ---
                if self.homography_rgb_to_tof is not None and mp_landmarks_2d_rgb_pixels:
                    landmarks_rgb_np = np.array([mp_landmarks_2d_rgb_pixels], dtype=np.float32)
                    try:
                        transformed_landmarks = cv2.perspectiveTransform(landmarks_rgb_np, self.homography_rgb_to_tof)
                        if transformed_landmarks is not None:
                            # print(f"DEBUG: Transformed landmarks (first 3 for ToF): {transformed_landmarks[0][:3]}") # DEBUG
                            # print(f"DEBUG: Drawing on ToF display of shape: {tof_display_bgr.shape}") # DEBUG
                            for pt_idx, pt_tof in enumerate(transformed_landmarks[0]):
                                x, y = int(pt_tof[0]), int(pt_tof[1])
                                if 0 <= x < tof_display_bgr.shape[1] and 0 <= y < tof_display_bgr.shape[0]:
                                    cv2.circle(tof_display_bgr, (x, y), 3, (0, 255, 0), -1)
                                    # cv2.putText(tof_display_bgr, str(pt_idx), (x+5,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0),1)
                        # else: print("DEBUG: perspectiveTransform returned None.")
                    except cv2.error as e:
                        print(f"ERROR: cv2.perspectiveTransform failed: {e}")
                        self.homography_rgb_to_tof = None 
                # elif self.homography_rgb_to_tof is None and mp_landmarks_2d_rgb_pixels:
                    # print("DEBUG: Homography not yet available for mapping.")


                # --- Add Display Source Text and FPS ---
                cv2.putText(tof_display_bgr, display_source_text, (5, tof_display_bgr.shape[0] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                current_loop_time = time.time()
                loop_fps = 1.0 / (current_loop_time - prev_time_main_loop + 1e-6)
                prev_time_main_loop = current_loop_time
                cv2.putText(tof_display_bgr, f"Loop FPS: {int(loop_fps)}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(annotated_rgb_frame, f"Loop FPS: {int(loop_fps)}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                cv2.imshow(main_tof_window_name, tof_display_bgr)
                cv2.imshow(rgb_window_name, annotated_rgb_frame)
                
                if frame_data_tof:
                    self.tof_cam.releaseFrame(frame_data_tof)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('c'):
                    self.homography_rgb_to_tof = None 
                    self.last_calibration_time = 0 
                    print("INFO: Calibration reset. Will attempt recalibration.")

        finally: # Identical to previous version
            print("Stopping cameras and cleaning up...")
            if hasattr(self, 'mp_pose_estimator') and self.mp_pose_estimator: self.mp_pose_estimator.close()
            if hasattr(self, 'tof_cam') and self.tof_cam.isOpened(): self.tof_cam.stop(); self.tof_cam.close()
            if self.webcam and self.webcam.isOpened(): self.webcam.release()
            cv2.destroyAllWindows()
            print("Cleanup complete.")

if __name__ == "__main__":
    print("Starting Dual Camera Pose Mapper...")
    # (Startup messages remain the same)
    try:
        estimator = DualCameraPoseMapper()
        estimator.run()
    except RuntimeError as e: print(f"FATAL: Initialization or runtime error: {e}")
    except Exception as e:
        print(f"FATAL: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()