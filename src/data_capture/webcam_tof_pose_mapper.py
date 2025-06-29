#!/usr/bin/env python3
"""
Webcam-ToF Pose Mapper
This script implements a complete pipeline for real-time pose detection and mapping between a webcam and Arducam ToF camera using ArUco marker calibration.
Uses a webcam for MediaPipe pose detection and maps the results onto the ToF camera display.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import sys
import os
import time
from pathlib import Path
from typing import Tuple, Optional, List, Any
try:
    import ArducamDepthCamera as ac
except ImportError:
    print("[WARNING] ArducamDepthCamera not available. ToF functionality will be limited.")
    ac = None

# Fix import by using parent folder
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

from camera_calibration import CameraCalibrationManager, ArUcoCalibrationSetup

class WebcamToFPoseMapper:
    """
    Maps pose detection from webcam to Arducam ToF camera display using calibration.
    """
    
    def __init__(self, webcam_id: int = 0, camera_range_mm: float = 4000.0, rotate_tof: bool = True):
        """
        Initialize with webcam ID and ToF camera settings.
        
        Args:
            webcam_id: Webcam camera ID for pose detection
            camera_range_mm: ToF camera range in millimeters
            rotate_tof: Whether to rotate ToF image 180 degrees
        """
        self.webcam_id = webcam_id
        self.camera_range_mm = camera_range_mm
        self.rotate_tof = rotate_tof
        
        # Camera objects
        self.webcam = None
        if ac is not None:
            self.tof_cam = ac.ArducamCamera()
        else:
            self.tof_cam = None
            print("[WARNING] ArducamDepthCamera not available. Running in webcam-only mode.")
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = None
        
        # Calibration objects
        self.calibration_setup = ArUcoCalibrationSetup()
        self.calibration_manager = None
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_file = "camera_calibration.json"
        
        # Display settings
        self.webcam_width = 640
        self.webcam_height = 480
        self.confidence_threshold = 30  # ToF confidence threshold
        
        print("[SUCCESS] Webcam-ToF Pose Mapper initialized")
        print(f"   Webcam ID: {webcam_id}")
        print(f"   ToF camera range: {camera_range_mm}mm")
        print(f"   Rotate ToF: {rotate_tof}")
    
    def initialize_cameras(self) -> bool:
        """
        Initialize both webcam and ToF camera.
        
        Returns True if both cameras initialized successfully
        """
        print("[LOG] Initializing cameras...")
        
        # Initialize webcam
        self.webcam = cv2.VideoCapture(self.webcam_id)
        if not self.webcam.isOpened():
            print(f"[ERROR] Failed to open webcam (ID: {self.webcam_id})")
            return False
        
        # Set webcam properties
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
        print("[SUCCESS] Webcam initialized successfully")
        
        # Initialize ToF camera
        if self.tof_cam is not None and ac is not None:
            try:
                ret_open = self.tof_cam.open(ac.Connection.CSI, 0)
                if ret_open != 0:
                    print(f"[ERROR] Failed to open ToF camera (error code: {ret_open})")
                    return False
                
                # Set ToF camera range
                desired_range_mm = int(self.camera_range_mm)
                self.tof_cam.setControl(ac.Control.RANGE, desired_range_mm)
                current_range = self.tof_cam.getControl(ac.Control.RANGE)
                print(f"[SUCCESS] ToF camera initialized with range: {current_range}mm")
                
                # Start ToF camera depth stream
                ret_start = self.tof_cam.start(ac.FrameType.DEPTH)
                if ret_start != 0:
                    self.tof_cam.close()
                    print(f"[ERROR] Failed to start ToF camera depth stream (error code: {ret_start})")
                    return False
                
                print("[SUCCESS] ToF camera depth stream started")
                
            except Exception as e:
                print(f"[ERROR] ToF camera initialization failed: {e}")
                return False
        else:
            print("[WARNING] ToF camera not available, using webcam-only mode")
        
        return True
    
    def initialize_mediapipe(self) -> bool:
        """
        Initialize MediaPipe pose detection.
        
        Returns True if MediaPipe initialized successfully
        """
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[SUCCESS] MediaPipe pose detection initialized")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize MediaPipe: {e}")
            return False
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture frames from webcam and ToF camera.
        
        Returns:
            tuple: (webcam_frame, tof_frame, depth_buf, confidence_frame, confidence_aruco_frame) or (None, None, None, None) if capture fails
        """
        if self.webcam is None:
            return None, None, None
        
        # Capture webcam frame
        ret, webcam_frame = self.webcam.read()
        if not ret:
            return None, None, None, None
        
        # Capture ToF frame
        if self.tof_cam is not None and ac is not None:
            try:
                frame_data = self.tof_cam.requestFrame(200)
                if frame_data is None:
                    return webcam_frame, None, None, None
                
                if not isinstance(frame_data, ac.DepthData):
                    self.tof_cam.releaseFrame(frame_data)
                    return webcam_frame, None, None, None
                
                depth_buf = frame_data.depth_data
                confidence_buf = frame_data.confidence_data
                
                if depth_buf is None or depth_buf.size == 0:
                    self.tof_cam.releaseFrame(frame_data)
                    return webcam_frame, None, None, None
                
                # Process ToF depth data for display
                tof_frame = self.process_tof_frame(depth_buf, confidence_buf)
                
                # Process confidence data for ArUco detection
                confidence_aruco_frame = self.process_confidence_frame_for_aruco(confidence_buf)
                
                self.tof_cam.releaseFrame(frame_data)
                return webcam_frame, tof_frame, depth_buf, confidence_buf, confidence_aruco_frame
                
            except Exception as e:
                print(f"[WARNING] ToF frame capture failed: {e}")
                return webcam_frame, None, None, None, None
        else:
            # ToF camera not available, return placeholder
            return webcam_frame, None, None, None, None
    
    def process_confidence_frame_for_aruco(self, confidence_buf: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Process ToF confidence data into a frame suitable for ArUco detection.
        
        Args:
            confidence_buf: Raw confidence data from ToF camera
            
        Returns:
            Processed BGR frame for ArUco detection, or None if no confidence data
        """
        if confidence_buf is None:
            return None
        
        # Process confidence buffer
        confidence_processed = confidence_buf.copy()
        
        # Rotate if specified (to match other frame orientations)
        if self.rotate_tof:
            confidence_processed = cv2.rotate(confidence_processed, cv2.ROTATE_180)
        
        # Normalize confidence values to 0-255 range
        cv2.normalize(confidence_processed, confidence_processed, 0, 255, cv2.NORM_MINMAX)
        confidence_processed = confidence_processed.astype(np.uint8)
        
        # Convert to BGR for consistency with other processing
        bgr_frame = cv2.cvtColor(confidence_processed, cv2.COLOR_GRAY2BGR)
        
        return bgr_frame

    def process_tof_frame(self, depth_buf: np.ndarray, confidence_buf: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process ToF depth data into displayable frame.
        
        Args:
            depth_buf: Raw depth data in millimeters
            confidence_buf: Optional confidence data
            
        Returns:
            Processed BGR frame
        """
        # Clamp values to the effective range
        depth_buf_clamped = np.clip(depth_buf, 0.0, self.camera_range_mm)
        
        # Normalize to 0-255
        if self.camera_range_mm > 0:
            depth_normalized = (depth_buf_clamped / self.camera_range_mm) * 255.0
        else:
            depth_normalized = (depth_buf_clamped / 4000.0) * 255.0
        
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Convert to BGR
        bgr_frame = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
        
        # Apply confidence filtering if available
        if confidence_buf is not None:
            confidence_buf_processed = confidence_buf.copy()
            if self.rotate_tof:
                confidence_buf_processed = cv2.rotate(confidence_buf_processed, cv2.ROTATE_180)
            
            if (confidence_buf_processed.shape[0] == bgr_frame.shape[0] and 
                confidence_buf_processed.shape[1] == bgr_frame.shape[1]):
                bgr_frame[confidence_buf_processed < self.confidence_threshold] = (0, 0, 0)
        
        # Rotate if specified
        if self.rotate_tof:
            bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
        
        return bgr_frame
    
    def draw_aruco_markers(self, frame: np.ndarray, corners, ids) -> np.ndarray:
        """
        Draw detected ArUco markers on frame.
        
        Args:
            frame: Input frame
            corners: Detected marker corners
            ids: Detected marker IDs
            
        Returns:
            Frame with markers drawn
        """
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Highlight calibration marker
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.calibration_setup.CALIBRATION_MARKER_ID:
                    corner = corners[i][0]
                    center = np.mean(corner, axis=0).astype(int)
                    cv2.circle(frame, tuple(center), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"CAL MARKER {marker_id}", 
                              (center[0] - 50, center[1] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def load_existing_calibration(self) -> bool:
        """
        Try to load existing calibration data.
        
        Returns True if calibration loaded successfully
        """
        self.calibration_manager = CameraCalibrationManager(self.calibration_file)
        
        if self.calibration_manager.is_loaded:
            self.is_calibrated = True
            print("[SUCCESS] Existing calibration loaded")
            return True
        else:
            print("[INFO] No existing calibration found")
            return False
    
    def phase1_live_calibration(self) -> bool:
        """
        Phase 1: Interactive ArUco calibration setup between webcam and ToF camera.
        
        Returns:
            bool: True if calibration was successful
        """
        print("\n[LOG] Phase 1: Live Calibration Setup")
        print("=" * 50)
        print("Instructions:")
        print("   1. Position ArUco marker ID 77 so it's visible to both cameras")
        print("   2. Press SPACE to capture calibration frames")
        print("   3. Press 'q' to quit without calibrating")
        print("   4. Multiple captures will improve calibration accuracy")
        print("   NOTE: Using ToF confidence map for ArUco detection")
        
        cv2.namedWindow("Webcam (Detection)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("ToF Camera (Target)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("ToF Confidence (ArUco Detection)", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Confidence Thr", "ToF Camera (Target)", 
                          self.confidence_threshold, 255, 
                          lambda val: setattr(self, 'confidence_threshold', val))
        
        captured_pairs = []
        
        while True:
            webcam_frame, tof_frame, depth_buf, confidence_buf, confidence_aruco_frame = self.capture_frames()
            if webcam_frame is None:
                print("[ERROR] Failed to capture webcam frame")
                break
            
            if tof_frame is None:
                print("[WARNING] ToF frame not available, using placeholder")
                tof_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Use confidence frame for ArUco detection if available, otherwise fall back to depth frame
            tof_aruco_frame = confidence_aruco_frame if confidence_aruco_frame is not None else tof_frame
            
            # Detect ArUco markers in both frames
            webcam_corners, webcam_ids = self.calibration_setup.detect_aruco_marker(webcam_frame)
            tof_corners, tof_ids = self.calibration_setup.detect_aruco_marker(tof_aruco_frame)
            
            # Draw markers on frames
            webcam_display = self.draw_aruco_markers(webcam_frame.copy(), webcam_corners, webcam_ids)
            tof_display = self.draw_aruco_markers(tof_frame.copy(), tof_corners, tof_ids)
            
            # Create confidence display with markers for ArUco detection visualization
            if confidence_aruco_frame is not None:
                confidence_display = self.draw_aruco_markers(confidence_aruco_frame.copy(), tof_corners, tof_ids)
                # Add info text to confidence display
                cv2.putText(confidence_display, "Confidence Map (ArUco Detection)", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                confidence_display = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(confidence_display, "No Confidence Data", (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add status text
            marker_detected_both = False
            if (webcam_ids is not None and tof_ids is not None and
                self.calibration_setup.CALIBRATION_MARKER_ID in webcam_ids.flatten() and
                self.calibration_setup.CALIBRATION_MARKER_ID in tof_ids.flatten()):
                marker_detected_both = True
                status_text = "READY - Press SPACE to capture"
                status_color = (0, 255, 0)
            else:
                status_text = "Position marker ID 77 visible to both cameras"
                status_color = (0, 0, 255)
            
            cv2.putText(webcam_display, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(tof_display, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(confidence_display, status_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Add capture count
            cv2.putText(webcam_display, f"Captured: {len(captured_pairs)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(tof_display, f"Captured: {len(captured_pairs)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(confidence_display, f"Captured: {len(captured_pairs)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frames
            cv2.imshow("Webcam (Detection)", webcam_display)
            cv2.imshow("ToF Camera (Target)", tof_display)
            cv2.imshow("ToF Confidence (ArUco Detection)", confidence_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and marker_detected_both:
                # Capture calibration pair - store webcam frame and confidence frame for ArUco detection
                captured_pairs.append((webcam_frame.copy(), tof_aruco_frame.copy()))
                print(f"[SUCCESS] Captured calibration pair {len(captured_pairs)} (using confidence map for ToF ArUco detection)")
                
                if len(captured_pairs) >= 5:
                    print("[INFO] Sufficient calibration pairs captured. You can continue capturing or press 'c' to compute calibration.")
            
            elif key == ord('c') and len(captured_pairs) >= 3:
                # Compute calibration
                print(f"[LOG] Computing calibration from {len(captured_pairs)} pairs...")
                if self.compute_calibration(captured_pairs):
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("[ERROR] Calibration computation failed")
            
            elif key == ord('q'):
                print("[INFO] Calibration cancelled by user")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return False
    
    def compute_calibration(self, captured_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """
        Compute calibration from captured frame pairs.
        
        Args:
            captured_pairs: List of (webcam_frame, tof_confidence_frame) tuples
            
        Returns:
            bool: True if calibration computed successfully
        """
        try:
            webcam_points = []
            tof_points = []
            
            for webcam_frame, tof_frame in captured_pairs:
                # Detect markers in both frames
                webcam_corners, webcam_ids = self.calibration_setup.detect_aruco_marker(webcam_frame)
                tof_corners, tof_ids = self.calibration_setup.detect_aruco_marker(tof_frame)
                
                # Find calibration marker in both frames
                webcam_marker_center = None
                tof_marker_center = None
                
                if webcam_ids is not None:
                    for i, marker_id in enumerate(webcam_ids.flatten()):
                        if marker_id == self.calibration_setup.CALIBRATION_MARKER_ID:
                            corner = webcam_corners[i][0]
                            webcam_marker_center = np.mean(corner, axis=0)
                            break
                
                if tof_ids is not None:
                    for i, marker_id in enumerate(tof_ids.flatten()):
                        if marker_id == self.calibration_setup.CALIBRATION_MARKER_ID:
                            corner = tof_corners[i][0]
                            tof_marker_center = np.mean(corner, axis=0)
                            break
                
                if webcam_marker_center is not None and tof_marker_center is not None:
                    webcam_points.append(webcam_marker_center)
                    tof_points.append(tof_marker_center)
            
            if len(webcam_points) < 4:
                print(f"[ERROR] Need at least 4 point pairs for calibration, got {len(webcam_points)}")
                return False
            
            # Compute homography
            webcam_points = np.array(webcam_points, dtype=np.float32)
            tof_points = np.array(tof_points, dtype=np.float32)
            
            homography, mask = cv2.findHomography(webcam_points, tof_points, 
                                                 cv2.RANSAC, 5.0)
            
            if homography is None:
                print("[ERROR] Failed to compute homography")
                return False
            
            # Save calibration
            calibration_data = {
                'homography_matrix': homography.tolist(),
                'webcam_points': webcam_points.tolist(),
                'tof_points': tof_points.tolist(),
                'num_points': len(webcam_points),
                'calibration_type': 'webcam_to_tof_confidence',
                'tof_detection_method': 'confidence_map',
                'timestamp': time.time()
            }
            
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"[SUCCESS] Calibration saved to {self.calibration_file}")
            print(f"   Points used: {len(webcam_points)}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Calibration computation failed: {e}")
            return False
    
    def extract_pose_landmarks(self, frame: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], Optional[Any]]:
        """
        Extract pose landmarks from frame using MediaPipe.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            tuple: (landmarks_list, mediapipe_results) or (None, None)
        """
        if self.pose is None:
            return None, None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return None, None
        
        # Extract landmark coordinates
        landmarks = []
        h, w = frame.shape[:2]
        
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        return landmarks, results
    
    def draw_pose_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw pose skeleton on frame using MediaPipe results.
        
        Args:
            frame: Input frame
            results: MediaPipe pose detection results
            
        Returns:
            Frame with skeleton drawn
        """
        if results is None or results.pose_landmarks is None:
            return frame
        
        # Make frame writable
        frame.flags.writeable = True
        
        # Draw the pose landmarks and connections
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return frame
    
    def draw_transformed_points(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Draw transformed landmark points on frame.
        
        Args:
            frame: Input frame
            points: Transformed landmark points
            
        Returns:
            Frame with points drawn
        """
        if points is None or len(points) == 0:
            return frame
        
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Yellow points
                cv2.circle(frame, (x, y), 7, (0, 0, 255), 2)     # Red outline
        
        return frame
    
    def phase2_realtime_pose_mapping(self) -> None:
        """
        Phase 2: Real-time pose detection and mapping.
        """
        print("\nPhase 2: Real-time Pose Mapping")
        print("=" * 50)
        print("Instructions:")
        print("   - Stand in front of webcam for pose detection")
        print("   - Watch transformed pose appear on ToF camera")
        print("   - Press 'q' to quit")
        
        cv2.namedWindow("Webcam (Pose Detection)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("ToF Camera (Transformed Pose)", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Confidence Thr", "ToF Camera (Transformed Pose)", 
                          self.confidence_threshold, 255, 
                          lambda val: setattr(self, 'confidence_threshold', val))
        
        fps_time = cv2.getTickCount()
        
        while True:
            webcam_frame, tof_frame, depth_buf, confidence_buf, confidence_aruco_frame = self.capture_frames()
            if webcam_frame is None:
                print("[ERROR] Failed to capture webcam frame")
                break
            
            if tof_frame is None:
                # Create placeholder if ToF frame not available
                tof_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Extract pose landmarks from webcam
            landmarks, pose_results = self.extract_pose_landmarks(webcam_frame)
            
            # Draw original skeleton on webcam frame
            webcam_display = webcam_frame.copy()
            if pose_results:
                webcam_display = self.draw_pose_skeleton(webcam_display, pose_results)
            
            # Transform landmarks to ToF camera coordinates
            tof_display = tof_frame.copy()
            if landmarks and self.calibration_manager and self.calibration_manager.is_loaded:
                try:
                    # Convert landmarks to numpy array
                    landmarks_array = np.array(landmarks, dtype=np.float32)
                    
                    # Transform using homography
                    if self.calibration_manager.homography is not None:
                        ones = np.ones((landmarks_array.shape[0], 1), dtype=np.float32)
                        landmarks_homogeneous = np.hstack([landmarks_array, ones])
                        
                        transformed_homogeneous = self.calibration_manager.homography @ landmarks_homogeneous.T
                        transformed_points = (transformed_homogeneous[:2] / transformed_homogeneous[2]).T
                        
                        # Draw transformed points on ToF frame
                        tof_display = self.draw_transformed_points(tof_display, transformed_points)
                        
                        # Add info text
                        cv2.putText(tof_display, f"Mapped points: {len(transformed_points)}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"[WARNING] Transformation error: {e}")
            
            # Calculate and display FPS
            current_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_time - fps_time)
            fps_time = current_time
            
            cv2.putText(webcam_display, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(tof_display, f"FPS: {int(fps)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add status information
            pose_status = "Pose detected" if landmarks else "No pose detected"
            cv2.putText(webcam_display, pose_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frames
            cv2.imshow("Webcam (Pose Detection)", webcam_display)
            cv2.imshow("ToF Camera (Transformed Pose)", tof_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[END] Pose mapping stopped by user")
                break
        
        cv2.destroyAllWindows()



    def capture_train_data(self) -> None:
        """
        Capture training data for machine learning model.
        """
        print("=" * 50)
        print("Instructions:")
        print("   - Stand in front of webcam for pose detection")
        print("   - Watch transformed pose appear on ToF camera")
        print("   - Press 'r' to record pose data")
        print("   - Press 's' to stop recording")
        print("   - Press 'q' to quit")
        
        cv2.namedWindow("Webcam (Pose Detection)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("ToF Camera (Transformed Pose)", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Confidence Thr", "ToF Camera (Transformed Pose)", 
                          self.confidence_threshold, 255, 
                          lambda val: setattr(self, 'confidence_threshold', val))
        
        fps_time = cv2.getTickCount()

        RECORDING = False
        
        while True:
            webcam_frame, tof_frame, depth_buf, confidence_buf, confidence_aruco_frame = self.capture_frames()
            if webcam_frame is None:
                print("[ERROR] Failed to capture webcam frame")
                break
            
            if tof_frame is None:
                # Create placeholder if ToF frame not available
                tof_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Extract pose landmarks from webcam
            landmarks, pose_results = self.extract_pose_landmarks(webcam_frame)
            
            # Draw original skeleton on webcam frame
            webcam_display = webcam_frame.copy()
            if pose_results:
                webcam_display = self.draw_pose_skeleton(webcam_display, pose_results)
            
            # Transform landmarks to ToF camera coordinates
            tof_display = tof_frame.copy()
            if landmarks and self.calibration_manager and self.calibration_manager.is_loaded:
                try:
                    # Convert landmarks to numpy array
                    landmarks_array = np.array(landmarks, dtype=np.float32)
                    
                    # Transform using homography
                    if self.calibration_manager.homography is not None:
                        ones = np.ones((landmarks_array.shape[0], 1), dtype=np.float32)
                        landmarks_homogeneous = np.hstack([landmarks_array, ones])
                        
                        transformed_homogeneous = self.calibration_manager.homography @ landmarks_homogeneous.T
                        transformed_points = (transformed_homogeneous[:2] / transformed_homogeneous[2]).T
                        
                        # Draw transformed points on ToF frame
                        tof_display = self.draw_transformed_points(tof_display, transformed_points)

                        # If recording, save the transformed points
                        if RECORDING:
                            print('[INFO] Recording pose data...')
                            # Save original time-of-flight frame to data/tof/[timestamp].jpg
                            timestamp = int(time.time())
                            tof_frame_path = f"data/tof/{timestamp}.jpg"
                            os.makedirs(os.path.dirname(tof_frame_path), exist_ok=True)
                            original_tof_frame = tof_frame.copy()
                            cv2.imwrite(tof_frame_path, original_tof_frame)

                            # Save transformed points to data/pose/[timestamp].json
                            pose_data = {
                                'timestamp': timestamp,
                                'transformed_points': transformed_points.tolist(),
                                'original_tof_frame': tof_frame_path
                            }
                            pose_data_path = f"data/pose/{timestamp}.json"
                            os.makedirs(os.path.dirname(pose_data_path), exist_ok=True)
                            with open(pose_data_path, 'w') as f:
                                json.dump(pose_data, f, indent=2)
                            print(f"[SUCCESS] Recorded pose data to {pose_data_path}")

                            # Save depth data to data/depth/[timestamp].json
                            depth_data_path = f"data/depth/{timestamp}.json"
                            os.makedirs(os.path.dirname(depth_data_path), exist_ok=True)
                            depth_data = depth_buf.tolist() if depth_buf is not None else []
                            with open(depth_data_path, 'w') as f:
                                json.dump(depth_data, f, indent=2)
                            print(f"[SUCCESS] Recorded depth data to {depth_data_path}")

                            # Save confidence data to data/confidence/[timestamp].json
                            confidence_data_path = f"data/confidence/{timestamp}.json"
                            os.makedirs(os.path.dirname(confidence_data_path), exist_ok=True)
                            confidence_data = confidence_buf.tolist() if confidence_buf is not None else []
                            with open(confidence_data_path, 'w') as f:
                                json.dump(confidence_data, f, indent=2)
                            print(f"[SUCCESS] Recorded confidence data to {confidence_data_path}")

                           # Save original webcam frame to data/webcam/[timestamp].jpg
                            webcam_frame_path = f"data/webcam/{timestamp}.jpg"
                            os.makedirs(os.path.dirname(webcam_frame_path), exist_ok=True)
                            original_webcam_frame = webcam_frame.copy()
                            cv2.imwrite(webcam_frame_path, original_webcam_frame)
                            print(f"[SUCCESS] Recorded webcam frame to {webcam_frame_path}")

                            # Save original pose data to data/original_pose/[timestamp].json 
                            original_pose_data_path = f"data/original_pose/{timestamp}.json"
                            os.makedirs(os.path.dirname(original_pose_data_path), exist_ok=True)
                            original_pose_data = {
                                'timestamp': timestamp,
                                'landmarks': landmarks,
                                'original_webcam_frame': webcam_frame_path
                            }
                            with open(original_pose_data_path, 'w') as f:
                                json.dump(original_pose_data, f, indent=2)
                            print(f"[SUCCESS] Recorded original pose data to {original_pose_data_path}")


                           
                        
                        # Add info text
                        cv2.putText(tof_display, f"Mapped points: {len(transformed_points)}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"[WARNING] Transformation error: {e}")
            
            # Calculate and display FPS
            current_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_time - fps_time)
            fps_time = current_time
            
            cv2.putText(webcam_display, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(tof_display, f"FPS: {int(fps)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add status information
            pose_status = "Pose detected" if landmarks else "No pose detected"
            cv2.putText(webcam_display, pose_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frames
            cv2.imshow("Webcam (Pose Detection)", webcam_display)
            cv2.imshow("ToF Camera (Transformed Pose)", tof_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[END] Pose mapping stopped by user")
                break
            if key == ord('r'):
                if not RECORDING:
                    print("[INFO] Starting recording pose data...")
                    RECORDING = True

            if key == ord('s'):
                if RECORDING:
                    print("[INFO] Stopping recording pose data...")
                    RECORDING = False

        
        cv2.destroyAllWindows()
    
    def validate_calibration_realtime(self) -> bool:
        """
        Real-time validation of calibration accuracy.
        
        Returns:
            bool: True if validation passed, False if user wants to recalibrate
        """
        print("\n[LOG] Calibration Validation")
        print("=" * 50)
        print("Instructions:")
        print("   - Position marker ID 77 visible to both cameras")
        print("   - Check if the green circles align on both cameras")
        print("   - Press 'y' if calibration looks good")
        print("   - Press 'n' to recalibrate")
        print("   - Press 'q' to quit")
        
        cv2.namedWindow("Webcam (Validation)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("ToF Camera (Validation)", cv2.WINDOW_AUTOSIZE)
        
        while True:
            webcam_frame, tof_frame, depth_buf, confidence_buf, confidence_aruco_frame = self.capture_frames()
            if webcam_frame is None:
                break
            
            if tof_frame is None:
                tof_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Use confidence frame for ArUco detection if available, otherwise fall back to depth frame
            tof_aruco_frame = confidence_aruco_frame if confidence_aruco_frame is not None else tof_frame
            
            # Detect ArUco markers (using confidence frame for ToF detection)
            webcam_corners, webcam_ids = self.calibration_setup.detect_aruco_marker(webcam_frame)
            tof_corners, tof_ids = self.calibration_setup.detect_aruco_marker(tof_aruco_frame)
            
            # Draw markers
            webcam_display = self.draw_aruco_markers(webcam_frame.copy(), webcam_corners, webcam_ids)
            tof_display = self.draw_aruco_markers(tof_frame.copy(), tof_corners, tof_ids)
            
            # Transform webcam marker to ToF coordinates if found
            if (webcam_ids is not None and 
                self.calibration_setup.CALIBRATION_MARKER_ID in webcam_ids.flatten() and
                self.calibration_manager and self.calibration_manager.homography is not None):
                
                for i, marker_id in enumerate(webcam_ids.flatten()):
                    if marker_id == self.calibration_setup.CALIBRATION_MARKER_ID:
                        corner = webcam_corners[i][0]
                        center = np.mean(corner, axis=0).astype(np.float32)
                        
                        # Transform to ToF coordinates
                        center_homogeneous = np.array([center[0], center[1], 1.0], dtype=np.float32)
                        transformed = self.calibration_manager.homography @ center_homogeneous
                        transformed_point = (transformed[:2] / transformed[2]).astype(int)
                        
                        # Draw transformed point on ToF frame
                        cv2.circle(tof_display, tuple(transformed_point), 15, (255, 0, 255), 3)
                        cv2.putText(tof_display, "PREDICTED", 
                                  (transformed_point[0] - 40, transformed_point[1] - 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        break
            
            # Add instructions
            cv2.putText(webcam_display, "y=accept, n=recalibrate, q=quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(tof_display, "Check alignment of markers", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Webcam (Validation)", webcam_display)
            cv2.imshow("ToF Camera (Validation)", tof_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                print("[SUCCESS] Calibration validated by user")
                cv2.destroyAllWindows()
                return True
            elif key == ord('n'):
                print("[INFO] User requested recalibration")
                cv2.destroyAllWindows()
                return False
            elif key == ord('q'):
                print("[INFO] Validation cancelled")
                cv2.destroyAllWindows()
                return True
    
    def run(self) -> None:
        """
        Main execution method that runs the complete pipeline.
        """
        print("Starting Webcam-ToF Pose Mapper")
        print("=" * 50)
        
        # Initialize cameras
        if not self.initialize_cameras():
            return
        
        # Initialize MediaPipe
        if not self.initialize_mediapipe():
            return
        
        # Try to load existing calibration
        if not self.load_existing_calibration():
            # Run live calibration if no existing calibration
            if not self.phase1_live_calibration():
                print("[ERROR] Calibration failed. Cannot proceed with pose mapping.")
                return
            
            # Load the newly created calibration
            if not self.load_existing_calibration():
                print("[ERROR] Failed to load newly created calibration")
                return
        
        # Validate calibration before proceeding
        print("\nValidating calibration accuracy...")
        while True:
            if self.validate_calibration_realtime():
                break  # Validation passed, continue to pose mapping
            else:
                # User chose to recalibrate
                print("[LOG] Recalibrating...")
                if not self.phase1_live_calibration():
                    print("[ERROR] Recalibration failed. Cannot proceed with pose mapping.")
                    return
                
                # Load the newly created calibration
                if not self.load_existing_calibration():
                    print("[ERROR] Failed to load newly created calibration")
                    return
        
        # Run real-time pose mapping
        self.phase2_realtime_pose_mapping()
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        print("\n[LOG] Cleaning up...")
        
        if self.webcam is not None:
            self.webcam.release()
            print("[LOG] Webcam released")
        
        if self.pose is not None:
            self.pose.close()
            print("[LOG] MediaPipe pose closed")
        
        if self.tof_cam is not None:
            try:
                self.tof_cam.stop()
                self.tof_cam.close()
                print("[LOG] ToF camera closed")
            except:
                pass
        
        cv2.destroyAllWindows()
        print("[LOG] All windows closed")



def main():
    """
    Main execution function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Webcam-ToF Pose Mapper")
    parser.add_argument("--webcam-id", type=int, default=0, 
                       help="Webcam ID for pose detection (default: 0)")
    parser.add_argument("--tof-range", type=float, default=4000.0,
                       help="ToF camera range in millimeters (default: 4000)")
    parser.add_argument("--no-rotate", action="store_true",
                       help="Don't rotate ToF image 180 degrees")
    
    args = parser.parse_args()
    
    try:
        mapper = WebcamToFPoseMapper(
            webcam_id=args.webcam_id,
            camera_range_mm=args.tof_range,
            rotate_tof=not args.no_rotate
        )
        mapper.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
