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
# Use DEPTH_AMPLITUDE_CONFIDENCE if available and provides all three
# If not, DEPTH_AMPLITUDE might be a good compromise, or stick to DEPTH
# and try to get amplitude through other means if needed.
# For now, let's assume DEPTH_AMPLITUDE will give us depth and amplitude,
# and we'll still try to access confidence if it comes along.
TOF_FRAME_TYPE = ac.FrameType.DEPTH_AMPLITUDE # Request Depth and Amplitude

# RGB Webcam
WEBCAM_INDEX = 8  # Adjust this to your RGB webcam's index (e.g., 0, 1, 2)
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# MediaPipe Pose Configuration
MP_MODEL_COMPLEXITY = 1
MP_MIN_DETECTION_CONFIDENCE = 0.75 # Increased for potentially more stable landmarks
MP_MIN_TRACKING_CONFIDENCE = 0.75

# ArUco Marker Configuration
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_6X6_250
ARUCO_MARKER_SIZE_METERS = 0.05 # Actual size of the marker (not strictly needed for homography but good practice)
# For robust calibration, you'd use cv2.solvePnP which needs actual 3D coordinates of marker corners
# For homography, we only need corresponding 2D points. We'll pick one marker.
ARUCO_CALIBRATION_MARKER_ID = 0 # ID of the marker to use for calibration

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
        self.show_confidence_map_window = True # To display the ToF confidence map

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
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        print(f"ArUco initialized with dictionary ID: {ARUCO_DICTIONARY_ID}")

        # Calibration
        self.homography_rgb_to_tof = None # To store the perspective transformation matrix
        self.last_calibration_time = 0
        self.calibration_interval = 5 # seconds, recalibrate if marker visible

        # --- ToF Camera Intrinsics (Placeholder - Replace with actual values) ---
        # These are essential for 3D reprojection. For homography, they are not directly used
        # but would be for solvePnP.
        # You would get these from a separate intrinsic calibration of the ToF camera's
        # IR sensor array (used for amplitude/confidence).
        # Example: fx, fy, cx, cy
        self.tof_camera_matrix = np.array([[300.0, 0, 240/2],
                                           [0, 300.0, 180/2],
                                           [0, 0, 1.0]])
        self.tof_dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion for ToF confidence map

        # --- RGB Webcam Intrinsics (Placeholder - Replace with actual values) ---
        # Calibrate your RGB webcam separately (e.g., using a chessboard) to get these.
        self.rgb_camera_matrix = np.array([[float(self.webcam_width), 0, self.webcam_width/2],
                                           [0, float(self.webcam_width), self.webcam_height/2], # Assuming fx ~ fy
                                           [0, 0, 1.0]])
        self.rgb_dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion for RGB webcam

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
                # Arducam ToF is 240x180
            else:
                print("Warning: Could not get ToF camera info.")
        except Exception as e:
            print(f"Warning: Error getting ToF camera info: {e}")

        try:
            desired_range_mm = int(self.tof_range_mm)
            self.tof_cam.setControl(ac.Control.RANGE, desired_range_mm)
            current_range_setting_mm = self.tof_cam.getControl(ac.Control.RANGE)
            print(f"Attempted to set ToF camera range to {desired_range_mm}mm. Current SDK setting: {current_range_setting_mm}mm")
            if current_range_setting_mm > 0: # Check if read was successful
                 self.tof_range_mm = float(current_range_setting_mm)
            print(f"Using ToF camera range for depth processing: {self.tof_range_mm}mm")
        except Exception as e:
            print(f"Warning: Could not set/get ToF camera range: {e}. Proceeding with assumed range: {self.tof_range_mm}mm.")

        ret_start_tof = self.tof_cam.start(TOF_FRAME_TYPE)
        if ret_start_tof != 0:
            self.tof_cam.close()
            raise RuntimeError(f"Failed to start ToF camera stream (error code: {ret_start_tof})")
        print(f"ToF Camera started successfully with FrameType: {TOF_FRAME_TYPE}.")

        # Setup RGB Webcam
        print(f"Opening RGB Webcam (Index: {WEBCAM_INDEX})...")
        self.webcam = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open RGB Webcam at index {WEBCAM_INDEX}.")
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
        # Verify actual resolution
        self.webcam_width = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.webcam_height = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"RGB Webcam opened successfully. Resolution: {self.webcam_width}x{self.webcam_height}")
        # Update RGB camera matrix with actual dimensions if placeholders were rough
        self.rgb_camera_matrix[0, 2] = self.webcam_width / 2
        self.rgb_camera_matrix[1, 2] = self.webcam_height / 2
        self.rgb_camera_matrix[0, 0] = float(self.webcam_width) # A rough estimate for fx
        self.rgb_camera_matrix[1, 1] = float(self.webcam_width) # A rough estimate for fy

    def process_tof_depth_for_display(self, depth_buf_mm):
        if depth_buf_mm is None or depth_buf_mm.size == 0:
            return None
        depth_buf_clamped = np.clip(depth_buf_mm, 0.0, self.tof_range_mm)
        if self.tof_range_mm > 0:
            depth_normalized = (depth_buf_clamped / self.tof_range_mm) * 255.0
        else:
            depth_normalized = (depth_buf_clamped / 4000.0) * 255.0 # Fallback
        depth_normalized_uint8 = depth_normalized.astype(np.uint8)
        bgr_frame = cv2.cvtColor(depth_normalized_uint8, cv2.COLOR_GRAY2BGR)
        if ROTATE_IMAGE:
            bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
        return bgr_frame

    def detect_aruco(self, image, camera_matrix=None, dist_coeffs=None):
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image, self.aruco_dict, parameters=self.aruco_params)
        
        rvecs, tvecs = None, None
        if ids is not None and camera_matrix is not None and dist_coeffs is not None:
            # Estimate pose of each marker
            # For this to be accurate, ARUCO_MARKER_SIZE_METERS must be correct
            # and camera_matrix & dist_coeffs must be accurately calibrated for the specific camera
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, ARUCO_MARKER_SIZE_METERS, camera_matrix, dist_coeffs)
            except Exception as e:
                # print(f"Warning: ArUco pose estimation failed: {e}")
                pass # Can happen if intrinsics are bad or marker too distorted

        return corners, ids, rejected, rvecs, tvecs

    def run(self):
        self.setup_cameras()

        main_tof_window_name = "ToF Grayscale with Mapped MediaPipe Landmarks"
        rgb_window_name = "RGB Webcam with MediaPipe Pose"
        confidence_window_name = "ToF Confidence Map (for ArUco)"

        cv2.namedWindow(main_tof_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(
            "ToF Confidence Thr", main_tof_window_name, self.tof_confidence_threshold,
            255, self.on_confidence_threshold_changed
        )
        if self.show_confidence_map_window:
            cv2.namedWindow(confidence_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(rgb_window_name, cv2.WINDOW_AUTOSIZE)

        print("\nStarting dual camera processing. Press 'q' to quit.")
        print(f"MediaPipe landmarks from RGB webcam will be mapped to ToF Grayscale (Amplitude) Image.")
        print(f"ArUco calibration (Homography) will attempt using Marker ID {ARUCO_CALIBRATION_MARKER_ID} "
              f"from RGB image and ToF Confidence Map.")

        prev_time_tof = 0
        prev_time_rgb = 0

        try:
            while True:
                # --- ToF Camera Frame ---
                frame_data_tof = self.tof_cam.requestFrame(200) # Timeout 200ms
                tof_depth_buf = None
                tof_amplitude_buf_original = None
                tof_confidence_buf_original = None

                if frame_data_tof:
                    if frame_data_tof.depth_data is not None:
                        tof_depth_buf = frame_data_tof.depth_data # This is in mm
                    if hasattr(frame_data_tof, 'amplitude_data') and frame_data_tof.amplitude_data is not None:
                        tof_amplitude_buf_original = frame_data_tof.amplitude_data
                    elif hasattr(frame_data_tof, 'confidence_data') and frame_data_tof.confidence_data is not None:
                         # Fallback: If no amplitude, use confidence also for grayscale display (less ideal)
                         # Or, use the depth visualization as placeholder
                         pass
                    if hasattr(frame_data_tof, 'confidence_data') and frame_data_tof.confidence_data is not None:
                        tof_confidence_buf_original = frame_data_tof.confidence_data
                
                # --- RGB Webcam Frame ---
                ret_rgb, rgb_frame_original = self.webcam.read()
                if not ret_rgb:
                    print("Error: Failed to grab RGB webcam frame.")
                    time.sleep(0.01)
                    if frame_data_tof: self.tof_cam.releaseFrame(frame_data_tof)
                    continue
                
                # --- Prepare ToF Amplitude Image for Display (Target: 240x180 Grayscale) ---
                tof_grayscale_display = None
                if tof_amplitude_buf_original is not None and tof_amplitude_buf_original.size > 0:
                    tof_amplitude_processed = tof_amplitude_buf_original.copy()
                    if ROTATE_IMAGE:
                        tof_amplitude_processed = cv2.rotate(tof_amplitude_processed, cv2.ROTATE_180)
                    # Ensure it's 8-bit, normalize if needed. Amplitude data is often 0-255 or 0-1023 etc.
                    if tof_amplitude_processed.dtype != np.uint8:
                         cv2.normalize(tof_amplitude_processed, tof_amplitude_processed, 0, 255, cv2.NORM_MINMAX)
                         tof_amplitude_processed = tof_amplitude_processed.astype(np.uint8)
                    if len(tof_amplitude_processed.shape) == 2: # If grayscale
                        tof_grayscale_display = cv2.cvtColor(tof_amplitude_processed, cv2.COLOR_GRAY2BGR)
                    else: # If already BGR (unlikely for raw amplitude)
                        tof_grayscale_display = tof_amplitude_processed
                else:
                    # Fallback: if no amplitude, use depth visualization
                    tof_grayscale_display = self.process_tof_depth_for_display(tof_depth_buf)
                    if tof_grayscale_display is None: # If depth is also bad
                         tof_grayscale_display = np.zeros((180, 240, 3), dtype=np.uint8) # Black screen
                
                # Ensure display is 240x180 (Arducam ToF native)
                if tof_grayscale_display.shape[0] != 180 or tof_grayscale_display.shape[1] != 240:
                    tof_grayscale_display = cv2.resize(tof_grayscale_display, (240, 180))


                # --- Prepare ToF Confidence Map for ArUco ---
                tof_confidence_for_aruco = None
                if tof_confidence_buf_original is not None and tof_confidence_buf_original.size > 0:
                    tof_confidence_for_aruco = tof_confidence_buf_original.copy()
                    if ROTATE_IMAGE:
                        tof_confidence_for_aruco = cv2.rotate(tof_confidence_for_aruco, cv2.ROTATE_180)
                    # Ensure it's 8-bit and single channel for ArUco
                    if tof_confidence_for_aruco.dtype != np.uint8:
                        cv2.normalize(tof_confidence_for_aruco, tof_confidence_for_aruco, 0, 255, cv2.NORM_MINMAX)
                        tof_confidence_for_aruco = tof_confidence_for_aruco.astype(np.uint8)
                    if len(tof_confidence_for_aruco.shape) == 3:
                        tof_confidence_for_aruco = cv2.cvtColor(tof_confidence_for_aruco, cv2.COLOR_BGR2GRAY)
                    
                    # Display confidence map
                    if self.show_confidence_map_window:
                        cv2.imshow(confidence_window_name, tof_confidence_for_aruco)

                # --- MediaPipe Pose on RGB Webcam ---
                rgb_frame_for_mp = cv2.cvtColor(rgb_frame_original, cv2.COLOR_BGR2RGB)
                rgb_frame_for_mp.flags.writeable = False # Optimization
                mp_results = self.mp_pose_estimator.process(rgb_frame_for_mp)
                rgb_frame_for_mp.flags.writeable = True
                
                annotated_rgb_frame = rgb_frame_original.copy()
                mp_landmarks_2d_rgb = [] # Store (x,y) pixel coordinates

                if mp_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_rgb_frame,
                        landmark_list=mp_results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    for landmark in mp_results.pose_landmarks.landmark:
                        # Landmarks from MediaPipe are normalized to [0,1]
                        lx = landmark.x * self.webcam_width
                        ly = landmark.y * self.webcam_height
                        # For this demo, we only need 2D landmarks from RGB.
                        # If MediaPipe provides world landmarks, you might use landmark.z too.
                        mp_landmarks_2d_rgb.append((lx, ly))

                    # --- ArUco Based Calibration (if MediaPipe found a pose) ---
                    # Attempt calibration periodically or if not yet calibrated
                    if time.time() - self.last_calibration_time > self.calibration_interval or self.homography_rgb_to_tof is None:
                        rgb_aruco_img = rgb_frame_original.copy() # Use original for detection
                        tof_aruco_img = tof_confidence_for_aruco # Use processed confidence map
                        
                        if tof_aruco_img is not None:
                            # Detect in RGB
                            # Using placeholder intrinsics for pose estimation, not strictly needed for homography from corners
                            corners_rgb, ids_rgb, _, rvecs_rgb, tvecs_rgb = self.detect_aruco(rgb_aruco_img, self.rgb_camera_matrix, self.rgb_dist_coeffs)
                            
                            # Detect in ToF Confidence Map
                            # Using placeholder intrinsics for ToF
                            corners_tof, ids_tof, _, rvecs_tof, tvecs_tof = self.detect_aruco(tof_aruco_img, self.tof_camera_matrix, self.tof_dist_coeffs)

                            if ids_rgb is not None and ids_tof is not None:
                                common_ids = set(ids_rgb.flatten()).intersection(set(ids_tof.flatten()))
                                # print(f"RGB IDs: {ids_rgb.flatten()}, ToF IDs: {ids_tof.flatten()}, Common: {common_ids}")
                                if ARUCO_CALIBRATION_MARKER_ID in common_ids:
                                    idx_rgb = np.where(ids_rgb.flatten() == ARUCO_CALIBRATION_MARKER_ID)[0]
                                    idx_tof = np.where(ids_tof.flatten() == ARUCO_CALIBRATION_MARKER_ID)[0]

                                    if len(idx_rgb) > 0 and len(idx_tof) > 0:
                                        rgb_marker_pts = corners_rgb[idx_rgb[0]][0].astype(np.float32) # Shape (4,2)
                                        tof_marker_pts = corners_tof[idx_tof[0]][0].astype(np.float32) # Shape (4,2)
                                        
                                        if len(rgb_marker_pts) == 4 and len(tof_marker_pts) == 4:
                                            H, status = cv2.findHomography(rgb_marker_pts, tof_marker_pts)
                                            if H is not None:
                                                self.homography_rgb_to_tof = H
                                                self.last_calibration_time = time.time()
                                                print(f"SUCCESS: Homography calculated using ArUco marker ID {ARUCO_CALIBRATION_MARKER_ID}.")
                                                # For drawing detected markers (optional)
                                                # cv2.aruco.drawDetectedMarkers(annotated_rgb_frame, [corners_rgb[idx_rgb[0]]], np.array([[ARUCO_CALIBRATION_MARKER_ID]]))
                                                # cv2.aruco.drawDetectedMarkers(tof_grayscale_display, [corners_tof[idx_tof[0]]], np.array([[ARUCO_CALIBRATION_MARKER_ID]]))
                                            else:
                                                print("Warning: Homography calculation failed.")
                                else:
                                    print(f"Warning: Calibration marker ID {ARUCO_CALIBRATION_MARKER_ID} not found in both views or no common marker.")
                            else:
                                print("Warning: ArUco markers not detected in one or both views for calibration.")
                        else:
                            print("Warning: ToF confidence map not available for ArUco detection.")
                
                # --- Map MediaPipe Landmarks to ToF Image if Homography Exists ---
                if self.homography_rgb_to_tof is not None and mp_landmarks_2d_rgb:
                    landmarks_rgb_np = np.array([mp_landmarks_2d_rgb], dtype=np.float32) # Shape (1, N, 2)
                    transformed_landmarks = cv2.perspectiveTransform(landmarks_rgb_np, self.homography_rgb_to_tof)
                    
                    if transformed_landmarks is not None:
                        for i, pt_tof in enumerate(transformed_landmarks[0]):
                            x, y = int(pt_tof[0]), int(pt_tof[1])
                            # Draw on the ToF grayscale display (which is 240x180)
                            if 0 <= x < tof_grayscale_display.shape[1] and 0 <= y < tof_grayscale_display.shape[0]:
                                cv2.circle(tof_grayscale_display, (x, y), 3, (0, 255, 0), -1) # Green circle
                                # cv2.putText(tof_grayscale_display, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)


                # --- FPS Calculation & Display ---
                # ToF FPS
                curr_time_tof = time.time()
                if prev_time_tof > 0 and frame_data_tof: # only if new frame
                    fps_tof = 1 / (curr_time_tof - prev_time_tof)
                    cv2.putText(tof_grayscale_display, f"ToF FPS: {int(fps_tof)}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                if frame_data_tof: prev_time_tof = curr_time_tof

                # RGB FPS
                curr_time_rgb = time.time()
                if prev_time_rgb > 0 :
                    fps_rgb = 1 / (curr_time_rgb - prev_time_rgb)
                    cv2.putText(annotated_rgb_frame, f"RGB FPS: {int(fps_rgb)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                prev_time_rgb = curr_time_rgb


                # --- Show Windows ---
                cv2.imshow(main_tof_window_name, tof_grayscale_display)
                cv2.imshow(rgb_window_name, annotated_rgb_frame)

                if frame_data_tof:
                    self.tof_cam.releaseFrame(frame_data_tof)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('c'): # Force recalibration attempt
                    self.homography_rgb_to_tof = None
                    self.last_calibration_time = 0
                    print("Attempting recalibration on next valid detection...")


        finally:
            print("Stopping cameras and cleaning up...")
            if hasattr(self, 'mp_pose_estimator') and self.mp_pose_estimator:
                self.mp_pose_estimator.close()
            if hasattr(self, 'tof_cam'):
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
        print(f"Initialization or runtime failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()