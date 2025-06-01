import ArducamDepthCamera as ac
import cv2
import numpy as np
import mediapipe as mp
import time

# --- Configuration ---
# CAMERA_RANGE is in MILLIMETERS. E.g., 4000 for 4 meters.
CAMERA_RANGE = 4000.0 
ROTATE_IMAGE = True 

# MediaPipe Pose Configuration
MODEL_COMPLEXITY = 1 
MIN_DETECTION_CONFIDENCE = 0.5 
MIN_TRACKING_CONFIDENCE = 0.5  

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class RealtimePoseEstimatorMediaPipe:
    def __init__(self):
        self.cam = ac.ArducamCamera()
        # self.range will store the range in MILLIMETERS
        self.range = float(CAMERA_RANGE) 
        self.pose = mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False, 
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
        print(f"MediaPipe Pose initialized with complexity={MODEL_COMPLEXITY}, "
              f"det_conf={MIN_DETECTION_CONFIDENCE}, track_conf={MIN_TRACKING_CONFIDENCE}")
        
        self.confidence_threshold = 30  # Default confidence threshold (0-255 range)
        self.show_confidence_map_window = True  # Control display of separate confidence map
        self.device_info = None # To store camera info

    def on_confidence_threshold_changed(self, value_from_trackbar):
        """Callback function for the confidence trackbar."""
        self.confidence_threshold = value_from_trackbar

    def setup_camera(self):
        print("Opening camera...")
        ret_open = self.cam.open(ac.Connection.CSI, 0) 
        
        if ret_open != 0: 
            raise RuntimeError(f"Failed to open camera with CSI connection (error code: {ret_open}).")
        print("Successfully opened camera with Connection.CSI.")

        # Get camera info (optional, but good for debugging)
        try:
            self.device_info = self.cam.getCameraInfo()
            if self.device_info:
                print(f"Camera Info: Name: {self.device_info.name}, Resolution: {self.device_info.width}x{self.device_info.height}, Type: {self.device_info.device_type}")
            else:
                print("Warning: Could not get camera info.")
        except Exception as e:
            print(f"Warning: Error getting camera info: {e}")

        # Set and verify camera range (in millimeters)
        try:
            desired_range_mm = int(self.range) 
            self.cam.setControl(ac.Control.RANGE, desired_range_mm)
            current_range_setting_mm = self.cam.getControl(ac.Control.RANGE)
            print(f"Attempted to set camera range to {desired_range_mm}mm. Current SDK setting: {current_range_setting_mm}mm")
            if current_range_setting_mm > 0:
                self.range = float(current_range_setting_mm)
            print(f"Using camera range for depth processing: {self.range}mm")
        except Exception as e:
            print(f"Warning: Could not set/get camera range: {e}. Proceeding with assumed range: {self.range}mm.")

        # Start camera depth stream
        ret_start = self.cam.start(ac.FrameType.DEPTH) # Request depth (and implicitly confidence if available)
        if ret_start != 0:
            self.cam.close()
            raise RuntimeError(f"Failed to start camera depth stream (error code: {ret_start})")
        print("Camera started successfully.")

    def process_camera_frame(self, depth_buf): # depth_buf is in mm, self.range is in mm
        # Clamp values to the effective range (0 to self.range millimeters)
        depth_buf_clamped = np.clip(depth_buf, 0.0, self.range)
        
        # Normalize to 0-255
        if self.range > 0:
            depth_normalized = (depth_buf_clamped / self.range) * 255.0
        else:
            # Fallback if range is zero (should not happen with proper setup)
            print("Warning: Camera range is zero. Using default 4000mm for depth normalization.")
            depth_normalized = (depth_buf_clamped / 4000.0) * 255.0 

        depth_normalized = depth_normalized.astype(np.uint8)

        # Convert to BGR (still grayscale visually, but 3-channel)
        bgr_frame = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

        if ROTATE_IMAGE:
            bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
        return bgr_frame
    
    def run(self):
        self.setup_camera()

        main_window_name = "Real-time ToF MediaPipe Pose Estimation"
        confidence_window_name = "Confidence Map"

        cv2.namedWindow(main_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(
            "Confidence Thr",
            main_window_name,
            self.confidence_threshold,
            255, 
            self.on_confidence_threshold_changed
        )
        
        print("\nStarting real-time MediaPipe Pose estimation with confidence filtering.")
        print(f"Default confidence threshold: {self.confidence_threshold} (adjustable via trackbar).")
        print("Press 'q' to quit.")
        prev_time = 0

        try:
            while True:
                frame_data = self.cam.requestFrame(200)
                if frame_data is None:
                    time.sleep(0.01)
                    continue

                if not isinstance(frame_data, ac.DepthData):
                    self.cam.releaseFrame(frame_data)
                    continue

                depth_buf = frame_data.depth_data
                confidence_buf_original = frame_data.confidence_data # Original orientation

                if depth_buf is None or depth_buf.size == 0:
                    print("Warning: Received empty depth buffer.")
                    self.cam.releaseFrame(frame_data)
                    continue
                
                # bgr_frame is already rotated inside process_camera_frame if ROTATE_IMAGE is True
                bgr_frame = self.process_camera_frame(depth_buf) 
                if bgr_frame is None or bgr_frame.size == 0:
                    print("Warning: Processed frame is empty.")
                    self.cam.releaseFrame(frame_data)
                    continue

                # --- Prepare confidence_buf to match bgr_frame's orientation ---
                current_confidence_buf_for_processing = None
                if confidence_buf_original is not None:
                    current_confidence_buf_for_processing = confidence_buf_original.copy()
                    if ROTATE_IMAGE: # If bgr_frame was rotated, rotate confidence_buf too
                        current_confidence_buf_for_processing = cv2.rotate(current_confidence_buf_for_processing, cv2.ROTATE_180)
                
                # --- Apply Confidence Masking using the correctly oriented confidence buffer ---
                if current_confidence_buf_for_processing is not None:
                    if current_confidence_buf_for_processing.shape[0] == bgr_frame.shape[0] and \
                       current_confidence_buf_for_processing.shape[1] == bgr_frame.shape[1]:
                        bgr_frame[current_confidence_buf_for_processing < self.confidence_threshold] = (0, 0, 0)
                    else:
                        print(f"Warning: Confidence buffer (shape: {current_confidence_buf_for_processing.shape}) "
                              f"mismatches frame shape ({bgr_frame.shape[:2]}). Skipping confidence filtering.")
                
                annotated_frame = bgr_frame.copy()

                # --- MediaPipe Processing ---
                rgb_frame_for_mp = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                rgb_frame_for_mp.flags.writeable = False
                results = self.pose.process(rgb_frame_for_mp)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # --- Calculate and Display FPS ---
                curr_time = time.time()
                if prev_time > 0: 
                    fps = 1 / (curr_time - prev_time)
                    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                prev_time = curr_time
                
                # --- Display Main Frame ---
                cv2.imshow(main_window_name, annotated_frame)

                # --- Display Confidence Map (Optional) ---
                # Uses current_confidence_buf_for_processing which is already correctly rotated
                if self.show_confidence_map_window and current_confidence_buf_for_processing is not None:
                    confidence_display = current_confidence_buf_for_processing.copy() # Already rotated if needed
                    cv2.normalize(confidence_display, confidence_display, 0, 255, cv2.NORM_MINMAX)
                    confidence_display = confidence_display.astype(np.uint8)
                    # No separate rotation needed here as current_confidence_buf_for_processing is already oriented
                    cv2.imshow(confidence_window_name, confidence_display)
                
                self.cam.releaseFrame(frame_data)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
        # ... (rest of the try-finally block remains the same) ...
        finally:
            print("Stopping camera and cleaning up...")
            if hasattr(self, 'pose') and self.pose:
                self.pose.close() 
            if hasattr(self, 'cam'):
                self.cam.stop()
                self.cam.close()
            cv2.destroyAllWindows()
            print("Cleanup complete.")
# --- Main Execution ---
if __name__ == "__main__":
    # Ensure CAMERA_RANGE at the top is set appropriately in MILLIMETERS.
    try:
        estimator = RealtimePoseEstimatorMediaPipe()
        estimator.run()
    except RuntimeError as e:
        print(f"Initialization or runtime failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()