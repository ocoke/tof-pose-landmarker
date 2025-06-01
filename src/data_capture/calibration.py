import ArducamDepthCamera as ac
import cv2
import numpy as np
import mediapipe as mp
import time

# --- Configuration ---
# ToF Camera
CAMERA_RANGE_MM = 4000.0  # ToF camera range in MILLIMETERS
ROTATE_IMAGE = True      # Rotate ToF camera image by 180 degrees
TOF_CONFIDENCE_THRESHOLD = 30 # Default confidence for filtering ToF data (0-255)
TOF_FRAME_TYPE = ac.FrameType.DEPTH # Using DEPTH as requested

# RGB Webcam
WEBCAM_INDEX = 8  # Adjust this to your RGB webcam's index (e.g., 0, 1, 2)
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# MediaPipe Pose Configuration
MP_MODEL_COMPLEXITY = 1
MP_MIN_DETECTION_CONFIDENCE = 0.75
MP_MIN_TRACKING_CONFIDENCE = 0.75

# ArUco Marker Configuration
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_6X6_250
ARUCO_MARKER_SIZE_METERS = 0.05
ARUCO_CALIBRATION_MARKER_ID = 0

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class DualCameraPoseMapper:
    def __init__(self):
        # ToF Camera
        self.tof_cam = ac.ArducamCamera()
        self.tof_range_mm = float(CAMERA_RANGE_MM)
        self.tof_confidence_threshold = TOF_CONFIDENCE_THRESHOLD
        self.tof_device_info = None
        self.show_confidence_map_window = True

        # RGB Webcam
        self.webcam = None
        self.webcam_width = WEBCAM_WIDTH
        self.webcam_height = WEBCAM_HEIGHT

        # MediaPipe
        self.mp_pose_estimator = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE)
        print(f"MediaPipe Pose initialized with complexity={MP_MODEL_COMPLEXITY}, "
              f"det_conf={MP_MIN_DETECTION_CONFIDENCE}, track_conf={MP_MIN_TRACKING_CONFIDENCE}")

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            print("Using cv2.aruco.DetectorParameters_create()")
        except AttributeError:
            print("cv2.aruco.DetectorParameters_create() not found, using cv2.aruco.DetectorParameters() (older OpenCV compatibility).")
            self.aruco_params = cv2.aruco.DetectorParameters()
        print(f"ArUco initialized with dictionary ID: {ARUCO_DICTIONARY_ID}")

        # Calibration
        self.homography_rgb_to_tof = None
        self.last_calibration_time = 0
        self.calibration_interval = 5 # seconds

        # Placeholder Intrinsics (CRITICAL: Replace with your actual calibrated values for accurate 3D work)
        # ToF camera (assuming 240x180 resolution for confidence/amplitude if used for ArUco)
        self.tof_camera_matrix = np.array([[200.0, 0, 240/2],  # fx, 0, cx
                                           [0, 200.0, 180/2],  # 0, fy, cy
                                           [0, 0, 1.0]], dtype=np.float32)
        self.tof_dist_coeffs = np.zeros((4,1), dtype=np.float32) # k1,k2,p1,p2 (can extend to k3 etc.)

        # RGB Webcam
        self.rgb_camera_matrix = np.array([[float(self.webcam_width*0.8), 0, self.webcam_width/2], # fx (estimate)
                                           [0, float(self.webcam_width*0.8), self.webcam_height/2], # fy (estimate)
                                           [0, 0, 1.0]], dtype=np.float32)
        self.rgb_dist_coeffs = np.zeros((4,1), dtype=np.float32)


    def on_confidence_threshold_changed(self, value_from_trackbar):
        self.tof_confidence_threshold = value_from_trackbar

    def setup_cameras(self):
        # Setup ToF Camera
        print("Opening ToF camera...")
        ret_open_tof = self.tof_cam.open(ac.Connection.CSI, 0)
        if ret_open_tof != 0:
            raise RuntimeError(f"Failed to open ToF camera with CSI (error code: {ret_open_tof}).")
        print("Successfully opened ToF camera.")

        try:
            self.tof_device_info = self.tof_cam.getCameraInfo()
            if self.tof_device_info:
                print(f"ToF Camera Info: Name: {self.tof_device_info.name}, "
                      f"Resolution: {self.tof_device_info.width}x{self.tof_device_info.height}, "
                      f"Type: {self.tof_device_info.device_type}")
                # Arducam ToF is typically 240x180 for depth/amplitude/confidence
                if self.tof_device_info.width != 240 or self.tof_device_info.height != 180:
                    print(f"Warning: ToF reported resolution {self.tof_device_info.width}x{self.tof_device_info.height} "
                          f"differs from expected 240x180. Adjust code if necessary.")
            else:
                print("Warning: Could not get ToF camera info.")
        except Exception as e:
            print(f"Warning: Error getting ToF camera info: {e}")

        try:
            desired_range_mm = int(self.tof_range_mm)
            self.tof_cam.setControl(ac.Control.RANGE, desired_range_mm)
            current_range_setting_mm = self.tof_cam.getControl(ac.Control.RANGE)
            print(f"Attempted to set ToF camera range to {desired_range_mm}mm. Current SDK setting: {current_range_setting_mm}mm")
            if current_range_setting_mm > 0:
                 self.tof_range_mm = float(current_range_setting_mm)
            print(f"Using ToF camera range for depth processing: {self.tof_range_mm}mm")
        except Exception as e:
            print(f"Warning: Could not set/get ToF camera range: {e}. Proceeding with assumed range: {self.tof_range_mm}mm.")

        ret_start_tof = self.tof_cam.start(TOF_FRAME_TYPE)
        if ret_start_tof != 0:
            self.tof_cam.close()
            raise RuntimeError(f"Failed to start ToF camera stream with type {TOF_FRAME_TYPE} (error code: {ret_start_tof})")
        print(f"ToF Camera started successfully with FrameType: {TOF_FRAME_TYPE}.")

        # Setup RGB Webcam
        print(f"Opening RGB Webcam (Index: {WEBCAM_INDEX})...")
        self.webcam = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open RGB Webcam at index {WEBCAM_INDEX}.")
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
        actual_webcam_width = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_webcam_height = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_webcam_width != self.webcam_width or actual_webcam_height != self.webcam_height:
            print(f"Warning: Webcam requested {self.webcam_width}x{self.webcam_height}, "
                  f"but got {actual_webcam_width}x{actual_webcam_height}. Updating dimensions.")
            self.webcam_width = actual_webcam_width
            self.webcam_height = actual_webcam_height
        
        print(f"RGB Webcam opened successfully. Resolution: {self.webcam_width}x{self.webcam_height}")
        # Update RGB camera matrix with actual dimensions if placeholders were rough
        self.rgb_camera_matrix[0, 2] = self.webcam_width / 2.0
        self.rgb_camera_matrix[1, 2] = self.webcam_height / 2.0
        # Rough estimate for focal length based on width (common heuristic if unknown)
        self.rgb_camera_matrix[0, 0] = float(self.webcam_width * 0.8) 
        self.rgb_camera_matrix[1, 1] = float(self.webcam_width * 0.8) 


    def process_depth_to_bgr(self, depth_buf_mm): # For display if amplitude is not available
        if depth_buf_mm is None or depth_buf_mm.size == 0:
            return np.zeros((180, 240, 3), dtype=np.uint8) # Return black BGR image
            
        depth_buf_clamped = np.clip(depth_buf_mm, 0.0, self.tof_range_mm)
        if self.tof_range_mm > 0:
            depth_normalized = (depth_buf_clamped / self.tof_range_mm) * 255.0
        else: # Fallback
            depth_normalized = (depth_buf_clamped / CAMERA_RANGE_MM) * 255.0
        depth_normalized_uint8 = depth_normalized.astype(np.uint8)
        bgr_frame = cv2.cvtColor(depth_normalized_uint8, cv2.COLOR_GRAY2BGR)
        
        if ROTATE_IMAGE:
            bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
        # Ensure output is 240x180
        if bgr_frame.shape[0] != 180 or bgr_frame.shape[1] != 240:
             bgr_frame = cv2.resize(bgr_frame, (240, 180))
        return bgr_frame

    def process_amplitude_to_bgr(self, amplitude_buf): # For display
        if amplitude_buf is None or amplitude_buf.size == 0:
            return None # Let caller decide fallback
            
        amplitude_processed = amplitude_buf.copy()
        if ROTATE_IMAGE:
            amplitude_processed = cv2.rotate(amplitude_processed, cv2.ROTATE_180)
        
        # Normalize if not already 8-bit (amplitude can be 10-bit, 12-bit, etc.)
        if amplitude_processed.dtype != np.uint8:
            cv2.normalize(amplitude_processed, amplitude_processed, 0, 255, cv2.NORM_MINMAX)
            amplitude_processed = amplitude_processed.astype(np.uint8)
        
        bgr_frame = cv2.cvtColor(amplitude_processed, cv2.COLOR_GRAY2BGR)
        # Ensure output is 240x180
        if bgr_frame.shape[0] != 180 or bgr_frame.shape[1] != 240:
             bgr_frame = cv2.resize(bgr_frame, (240, 180))
        return bgr_frame

    def process_confidence_for_aruco(self, confidence_buf):
        if confidence_buf is None or confidence_buf.size == 0:
            return None
        
        conf_for_aruco = confidence_buf.copy()
        if ROTATE_IMAGE:
            conf_for_aruco = cv2.rotate(conf_for_aruco, cv2.ROTATE_180)
        
        # Ensure it's 8-bit and single channel
        if conf_for_aruco.dtype != np.uint8:
            cv2.normalize(conf_for_aruco, conf_for_aruco, 0, 255, cv2.NORM_MINMAX)
            conf_for_aruco = conf_for_aruco.astype(np.uint8)
        if len(conf_for_aruco.shape) == 3 and conf_for_aruco.shape[2] == 3: # If it's BGR
            conf_for_aruco = cv2.cvtColor(conf_for_aruco, cv2.COLOR_BGR2GRAY)
        
        # Ensure output is 240x180
        if conf_for_aruco.shape[0] != 180 or conf_for_aruco.shape[1] != 240:
            conf_for_aruco = cv2.resize(conf_for_aruco, (240, 180), interpolation=cv2.INTER_NEAREST)
        return conf_for_aruco


    def detect_aruco(self, image, camera_matrix, dist_coeffs):
        if image is None:
            return None, None, None, None, None
        # ArUco detection expects a grayscale image
        gray_image = image
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray_image, self.aruco_dict, parameters=self.aruco_params)
        
        rvecs, tvecs = None, None
        if ids is not None: # If any markers are found
            # You MUST use accurate camera_matrix and dist_coeffs for the specific camera
            # from which 'image' was captured for estimatePoseSingleMarkers to be meaningful.
            if camera_matrix is not None and dist_coeffs is not None:
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, ARUCO_MARKER_SIZE_METERS, camera_matrix, dist_coeffs)
                except cv2.error as e: # Catch OpenCV errors during pose estimation
                    # print(f"cv2.error in estimatePoseSingleMarkers: {e}")
                    pass # rvecs, tvecs will remain None
                except Exception as e:
                    # print(f"Unexpected error in estimatePoseSingleMarkers: {e}")
                    pass
        return corners, ids, rejected, rvecs, tvecs

    def run(self):
        self.setup_cameras()

        main_tof_window_name = "ToF Output with Mapped MediaPipe Landmarks"
        rgb_window_name = "RGB Webcam with MediaPipe Pose"
        confidence_window_name = "ToF Confidence Map (for ArUco)"

        cv2.namedWindow(main_tof_window_name, cv2.WINDOW_AUTOSIZE)
        # cv2.createTrackbar( # Confidence filtering of display is removed for now for simplicity
        # "ToF Confidence Thr", main_tof_window_name, self.tof_confidence_threshold,
        # 255, self.on_confidence_threshold_changed
        # )
        if self.show_confidence_map_window:
            cv2.namedWindow(confidence_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(rgb_window_name, cv2.WINDOW_AUTOSIZE)

        print("\nStarting dual camera processing. Press 'q' to quit.")
        print(f"MediaPipe landmarks from RGB webcam will be mapped to ToF display.")
        print(f"ArUco calibration (Homography) will attempt using Marker ID {ARUCO_CALIBRATION_MARKER_ID}.")

        prev_time_tof = 0
        prev_time_rgb = 0

        try:
            while True:
                # --- ToF Camera Frame ---
                frame_data_tof = self.tof_cam.requestFrame(200)
                tof_depth_buf_mm = None
                tof_amplitude_buf = None # Raw amplitude
                tof_confidence_buf = None # Raw confidence

                if frame_data_tof:
                    tof_depth_buf_mm = frame_data_tof.depth_data # This is in mm
                    if hasattr(frame_data_tof, 'amplitude_data'): # Check if amplitude_data exists
                        tof_amplitude_buf = frame_data_tof.amplitude_data
                    if hasattr(frame_data_tof, 'confidence_data'):
                        tof_confidence_buf = frame_data_tof.confidence_data
                
                # --- RGB Webcam Frame ---
                ret_rgb, rgb_frame_original = self.webcam.read()
                if not ret_rgb:
                    print("Error: Failed to grab RGB webcam frame.")
                    time.sleep(0.01)
                    if frame_data_tof: self.tof_cam.releaseFrame(frame_data_tof)
                    continue
                
                # --- Prepare ToF display image (Prioritize Amplitude, fallback to Depth Vis) ---
                tof_display_bgr = None
                if tof_amplitude_buf is not None:
                    tof_display_bgr = self.process_amplitude_to_bgr(tof_amplitude_buf)
                
                if tof_display_bgr is None: # Fallback if amplitude processing failed or not available
                    tof_display_bgr = self.process_depth_to_bgr(tof_depth_buf_mm)
                
                # --- Prepare ToF Confidence Map for ArUco (rotated, normalized) ---
                processed_tof_confidence_map = self.process_confidence_for_aruco(tof_confidence_buf)
                
                if self.show_confidence_map_window and processed_tof_confidence_map is not None:
                    cv2.imshow(confidence_window_name, processed_tof_confidence_map)

                # --- MediaPipe Pose on RGB Webcam ---
                rgb_frame_for_mp = cv2.cvtColor(rgb_frame_original, cv2.COLOR_BGR2RGB)
                rgb_frame_for_mp.flags.writeable = False
                mp_results = self.mp_pose_estimator.process(rgb_frame_for_mp)
                rgb_frame_for_mp.flags.writeable = True # For drawing
                
                annotated_rgb_frame = rgb_frame_original.copy()
                mp_landmarks_2d_rgb_pixels = []

                if mp_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_rgb_frame, landmark_list=mp_results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    for landmark in mp_results.pose_landmarks.landmark:
                        lx = landmark.x * self.webcam_width
                        ly = landmark.y * self.webcam_height
                        mp_landmarks_2d_rgb_pixels.append((lx, ly))

                    # --- ArUco Based Calibration Attempt ---
                    if time.time() - self.last_calibration_time > self.calibration_interval or self.homography_rgb_to_tof is None:
                        if processed_tof_confidence_map is not None:
                            corners_rgb, ids_rgb, _, _, _ = self.detect_aruco(rgb_frame_original, self.rgb_camera_matrix, self.rgb_dist_coeffs)
                            corners_tof, ids_tof, _, _, _ = self.detect_aruco(processed_tof_confidence_map, self.tof_camera_matrix, self.tof_dist_coeffs)

                            if ids_rgb is not None and ids_tof is not None:
                                common_ids_found = set(ids_rgb.flatten()).intersection(set(ids_tof.flatten()))
                                if ARUCO_CALIBRATION_MARKER_ID in common_ids_found:
                                    idx_rgb = np.where(ids_rgb.flatten() == ARUCO_CALIBRATION_MARKER_ID)[0][0]
                                    idx_tof = np.where(ids_tof.flatten() == ARUCO_CALIBRATION_MARKER_ID)[0][0]
                                    
                                    rgb_marker_pts_for_H = corners_rgb[idx_rgb][0].astype(np.float32)
                                    tof_marker_pts_for_H = corners_tof[idx_tof][0].astype(np.float32)
                                    
                                    if len(rgb_marker_pts_for_H) >= 4 and len(tof_marker_pts_for_H) >= 4: # Need at least 4 points
                                        H, status = cv2.findHomography(rgb_marker_pts_for_H, tof_marker_pts_for_H, cv2.RANSAC, 5.0)
                                        if H is not None:
                                            self.homography_rgb_to_tof = H
                                            self.last_calibration_time = time.time()
                                            print(f"SUCCESS: Homography calculated and updated using ArUco marker ID {ARUCO_CALIBRATION_MARKER_ID}.")
                                        # else: print("Warning: Homography calculation failed (findHomography returned None).")
                                # else: print(f"Warning: Calibration marker ID {ARUCO_CALIBRATION_MARKER_ID} not found in both views or no common marker.")
                            # else: print("Warning: ArUco markers not detected in one or both views for calibration attempt.")
                        # else: print("Warning: ToF confidence map not available for ArUco detection during calibration attempt.")
                
                # --- Map MediaPipe Landmarks to ToF Display Image ---
                if self.homography_rgb_to_tof is not None and mp_landmarks_2d_rgb_pixels:
                    landmarks_rgb_np = np.array([mp_landmarks_2d_rgb_pixels], dtype=np.float32) # Shape (1, N, 2)
                    transformed_landmarks = cv2.perspectiveTransform(landmarks_rgb_np, self.homography_rgb_to_tof)
                    
                    if transformed_landmarks is not None:
                        for pt_tof in transformed_landmarks[0]:
                            x, y = int(pt_tof[0]), int(pt_tof[1])
                            if 0 <= x < tof_display_bgr.shape[1] and 0 <= y < tof_display_bgr.shape[0]: # Check bounds
                                cv2.circle(tof_display_bgr, (x, y), 3, (0, 255, 0), -1) # Green circle

                # --- FPS Calculation & Display ---
                curr_time = time.time()
                if prev_time_tof > 0 and frame_data_tof:
                    fps_tof = 1 / (curr_time - prev_time_tof)
                    cv2.putText(tof_display_bgr, f"ToF FPS: {int(fps_tof)}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                if frame_data_tof: prev_time_tof = curr_time # Update time only if ToF frame was processed

                if prev_time_rgb > 0 :
                    fps_rgb = 1 / (curr_time - prev_time_rgb)
                    cv2.putText(annotated_rgb_frame, f"RGB FPS: {int(fps_rgb)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                prev_time_rgb = curr_time


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
                    self.homography_rgb_to_tof = None # Reset homography
                    self.last_calibration_time = 0 # Force recalibration attempt
                    print("Calibration reset. Will attempt recalibration on next valid detection...")

        finally:
            print("Stopping cameras and cleaning up...")
            if hasattr(self, 'mp_pose_estimator') and self.mp_pose_estimator:
                self.mp_pose_estimator.close()
            if hasattr(self, 'tof_cam') and self.tof_cam.isOpened(): # Check if tof_cam was successfully opened
                self.tof_cam.stop()
                self.tof_cam.close()
            if self.webcam and self.webcam.isOpened():
                self.webcam.release()
            cv2.destroyAllWindows()
            print("Cleanup complete.")

if __name__ == "__main__":
    try:
        estimator = DualCameraPoseMapper()
        estimator.run()
    except RuntimeError as e:
        print(f"Initialization or runtime error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()