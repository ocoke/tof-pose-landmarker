#!/usr/bin/env python3
"""
This code is using two webcams (instead of one webcam and one Time-of-Flight camera) as a test prototype.
The script implements a complete pipeline for real-time pose detection and mapping between two webcams using ArUco marker calibration.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import sys
import os
from pathlib import Path
from typing import Tuple, Optional, List

# fix import by using parent folder
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

from camera_calibration import CameraCalibrationManager, ArUcoCalibrationSetup

class DualWebcamPoseMapper:
    # dual camera
    
    def __init__(self, detection_camera_id: int = 1, target_camera_id: int = 0):
        """
        Initialize with two camera ids
        
        detection: the one runs Mediapipe
        target: the one shows the result
        """
        self.detection_camera_id = detection_camera_id
        self.target_camera_id = target_camera_id
        
        # Camera objects
        self.detection_cam = None
        self.target_cam = None
        
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
        self.window_width = 640
        self.window_height = 480
        
        print("[SUCCESS] Dual Webcam Pose Mapper initialized")
        print(f"   Detection camera: {detection_camera_id}")
        print(f"   Target camera: {target_camera_id}")
    
    def initialize_cameras(self) -> bool:
        """
        Initialize both webcams.
        
        Returns True if both cameras initialized successfully
        """
        print("[LOG] Initializing cameras...")
        
        self.detection_cam = cv2.VideoCapture(self.detection_camera_id)
        if not self.detection_cam.isOpened():
            print(f"[ERROR] Failed to open detection camera (ID: {self.detection_camera_id})")
            return False
        
        self.target_cam = cv2.VideoCapture(self.target_camera_id)
        if not self.target_cam.isOpened():
            print(f"[ERROR] Failed to open target camera (ID: {self.target_camera_id})")
            return False
        
        # Set camera properties
        self.detection_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
        self.detection_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
        self.target_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
        self.target_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
        
        print("[SUCCESS] Both cameras initialized successfully")
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
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture frames from both cameras.
        
        Returns:
            tuple: (detection_frame, target_frame) or (None, None) if capture fails
        """
        if self.detection_cam is None or self.target_cam is None:
            return None, None
        
        ret1, detection_frame = self.detection_cam.read()
        ret2, target_frame = self.target_cam.read()
        
        if not ret1 or not ret2:
            return None, None
        
        return detection_frame, target_frame
    
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
    
    def phase1_live_calibration(self) -> bool:
        """
        Phase 1: Interactive ArUco calibration setup.
        
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
        
        cv2.namedWindow("Detection Camera (Live)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Target Camera (Live)", cv2.WINDOW_AUTOSIZE)
        
        captured_pairs = []
        
        while True:
            detection_frame, target_frame = self.capture_frames()
            if detection_frame is None or target_frame is None:
                print("[ERROR] Failed to capture frames")
                break
            
            # Detect ArUco markers in both frames
            detection_corners, detection_ids = self.calibration_setup.detect_aruco_marker(detection_frame)
            target_corners, target_ids = self.calibration_setup.detect_aruco_marker(target_frame)
            
            # Draw markers on frames
            detection_display = self.draw_aruco_markers(detection_frame.copy(), detection_corners, detection_ids)
            target_display = self.draw_aruco_markers(target_frame.copy(), target_corners, target_ids)
            
            # Add status text
            marker_detected_both = False
            if (detection_ids is not None and target_ids is not None and
                self.calibration_setup.CALIBRATION_MARKER_ID in detection_ids.flatten() and
                self.calibration_setup.CALIBRATION_MARKER_ID in target_ids.flatten()):
                marker_detected_both = True
                status_text = "READY - Press SPACE to capture"
                status_color = (0, 255, 0)
            else:
                status_text = "Position marker ID 77 visible to both cameras"
                status_color = (0, 0, 255)
            
            cv2.putText(detection_display, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(target_display, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show capture count
            cv2.putText(detection_display, f"Captures: {len(captured_pairs)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(target_display, f"Captures: {len(captured_pairs)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Detection Camera (Live)", detection_display)
            cv2.imshow("Target Camera (Live)", target_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and marker_detected_both:
                # Capture calibration frames
                captured_pairs.append((detection_frame.copy(), target_frame.copy()))
                print(f"[SUCCESS] Captured calibration pair {len(captured_pairs)}")
                
                # Flash effect
                flash_frame = np.ones_like(detection_frame) * 255
                cv2.imshow("Detection Camera (Live)", flash_frame)
                cv2.imshow("Target Camera (Live)", flash_frame)
                cv2.waitKey(100)
                
            elif key == ord('q'):
                print("[END] Calibration cancelled by user")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('c') and len(captured_pairs) >= 1:
                # Start calibration computation
                break
        
        cv2.destroyAllWindows()
        
        if len(captured_pairs) == 0:
            print("[ERROR] No calibration frames captured")
            return False
        
        # Compute calibration from captured frames
        print(f"\n[LOG] Computing calibration from {len(captured_pairs)} frame pairs...")
        
        # Use the best frame pair (or average multiple if needed)
        # For simplicity, we'll use the last captured pair
        detection_frame, target_frame = captured_pairs[-1]
        
        homography = self.calibration_setup.calculate_homography_from_frames(
            detection_frame, target_frame
        )
        
        if homography is None:
            print("[ERROR] Failed to compute homography")
            return False
        
        # Save calibration
        if self.calibration_setup.save_calibration(homography, self.calibration_file):
            print("[SUCCESS] Calibration completed and saved!")
            self.is_calibrated = True
            return True
        else:
            print("[ERROR] Failed to save calibration")
            return False
    
    def load_existing_calibration(self) -> bool:
        """
        Try to load existing calibration.
        
        Returns:
            bool: True if calibration loaded successfully
        """
        self.calibration_manager = CameraCalibrationManager(self.calibration_file)
        if self.calibration_manager.is_loaded:
            self.is_calibrated = True
            print("[SUCCESS] Existing calibration loaded successfully")
            return True
        else:
            print("⚠️  No existing calibration found")
            return False
    
    def extract_pose_landmarks(self, frame: np.ndarray) -> Tuple[Optional[List[Tuple[float, float]]], Optional[object]]:
        """
        Extract 2D pose landmarks from frame using MediaPipe.
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Tuple of (landmark coordinates, MediaPipe results) or (None, None) if no pose detected
        """
        if self.pose is None:
            return None, None
        
        # Convert BGR to RGB for MediaPipe
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
        print("   - Stand in front of detection camera for pose detection")
        print("   - Watch transformed pose appear on target camera")
        print("   - Press 'q' to quit")
        
        cv2.namedWindow("Detection Camera (Pose)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Target Camera (Transformed)", cv2.WINDOW_AUTOSIZE)
        
        fps_time = cv2.getTickCount()
        
        while True:
            detection_frame, target_frame = self.capture_frames()
            if detection_frame is None or target_frame is None:
                print("[ERROR] Failed to capture frames")
                break
            
            # Extract pose landmarks from detection camera
            landmarks, pose_results = self.extract_pose_landmarks(detection_frame)
            
            # Draw original skeleton on detection frame
            detection_display = detection_frame.copy()
            if pose_results:
                detection_display = self.draw_pose_skeleton(detection_display, pose_results)
            
            # Transform landmarks to target camera coordinates
            target_display = target_frame.copy()
            if landmarks and self.calibration_manager:
                # Convert landmarks to numpy array
                landmarks_array = np.array(landmarks, dtype=np.float32)
                
                # Transform using calibration
                transformed_landmarks, valid_indices = self.calibration_manager.transform_landmarks_rgb_to_tof(landmarks_array)
                
                # Draw transformed points on target frame
                if len(transformed_landmarks) > 0:
                    target_display = self.draw_transformed_points(target_display, transformed_landmarks)
                    
                    # Add info text
                    cv2.putText(target_display, f"Mapped points: {len(transformed_landmarks)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate and display FPS
            current_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_time - fps_time)
            fps_time = current_time
            
            cv2.putText(detection_display, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(target_display, f"FPS: {int(fps)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add status information
            pose_status = "Pose detected" if landmarks else "No pose detected"
            cv2.putText(detection_display, pose_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add debug information
            if landmarks:
                detection_display = self.add_debug_info_to_frame(detection_display, landmarks, transformed_landmarks, valid_indices)
            target_display = self.add_debug_info_to_frame(target_display, landmarks, transformed_landmarks, valid_indices)
            
            # Display frames
            cv2.imshow("Detection Camera (Pose)", detection_display)
            cv2.imshow("Target Camera (Transformed)", target_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[END] Pose mapping stopped by user")
                break
        
        cv2.destroyAllWindows()
    
    def run(self) -> None:
        """
        Main execution method that runs the complete pipeline.
        """
        print("Starting Dual Webcam Pose Mapper")
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
        print("\n Validating calibration accuracy...")
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
        
        if self.detection_cam:
            self.detection_cam.release()
        
        if self.target_cam:
            self.target_cam.release()
        
        if self.pose:
            self.pose.close()
        
        cv2.destroyAllWindows()
        print("[SUCCESS] Cleanup complete")
    
    def validate_calibration_realtime(self) -> bool:
        """
        Show a real-time validation of the calibration using ArUco marker.
        This helps verify that the transformation is working correctly.
        """
        print("\nCalibration Validation Mode")
        print("=" * 50)
        print("Instructions:")
        print("   1. Hold ArUco marker ID 77 visible to both cameras")
        print("   2. The system will show detected marker positions")
        print("   3. Verify that transformed markers align correctly")
        print("   4. Press 'v' to continue to pose mapping, 'r' to recalibrate, 'q' to quit")
        
        cv2.namedWindow("Detection Camera (Validation)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Target Camera (Validation)", cv2.WINDOW_AUTOSIZE)
        
        while True:
            detection_frame, target_frame = self.capture_frames()
            if detection_frame is None or target_frame is None:
                continue
            
            # Detect ArUco markers in both frames
            detection_corners, detection_ids = self.calibration_setup.detect_aruco_marker(detection_frame)
            target_corners, target_ids = self.calibration_setup.detect_aruco_marker(target_frame)
            
            # Draw markers on frames
            detection_display = self.draw_aruco_markers(detection_frame.copy(), detection_corners, detection_ids)
            target_display = self.draw_aruco_markers(target_frame.copy(), target_corners, target_ids)
            
            # If calibration marker is detected in detection camera, transform it
            if (detection_ids is not None and 
                self.calibration_setup.CALIBRATION_MARKER_ID in detection_ids.flatten()):
                
                # Get marker center from detection camera
                marker_idx = np.where(detection_ids.flatten() == self.calibration_setup.CALIBRATION_MARKER_ID)[0][0]
                marker_corners = detection_corners[marker_idx][0]
                marker_center = np.mean(marker_corners, axis=0).astype(np.float32)
                
                # Transform marker center using calibration
                if self.calibration_manager:
                    transformed_points, valid_indices = self.calibration_manager.transform_landmarks_rgb_to_tof([marker_center])
                    
                    if len(transformed_points) > 0:
                        # Draw transformed marker center on target frame
                        tx, ty = int(transformed_points[0][0]), int(transformed_points[0][1])
                        cv2.circle(target_display, (tx, ty), 15, (0, 255, 255), 3)
                        cv2.circle(target_display, (tx, ty), 5, (0, 0, 255), -1)
                        cv2.putText(target_display, "TRANSFORMED", (tx + 20, ty),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Calculate error if actual marker is detected in target camera
                        if (target_ids is not None and 
                            self.calibration_setup.CALIBRATION_MARKER_ID in target_ids.flatten()):
                            
                            target_marker_idx = np.where(target_ids.flatten() == self.calibration_setup.CALIBRATION_MARKER_ID)[0][0]
                            target_marker_corners = target_corners[target_marker_idx][0]
                            target_marker_center = np.mean(target_marker_corners, axis=0).astype(np.float32)
                            
                            # Calculate error
                            error = np.linalg.norm(transformed_points[0] - target_marker_center)
                            
                            # Draw error line
                            cv2.line(target_display, (tx, ty), tuple(target_marker_center.astype(int)), (255, 0, 0), 2)
                            cv2.putText(target_display, f"Error: {error:.1f}px", (10, 90),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
                            # Color code the accuracy
                            if error < 10:
                                accuracy_color = (0, 255, 0)  # Green - Good
                                accuracy_text = "GOOD"
                            elif error < 25:
                                accuracy_color = (0, 255, 255)  # Yellow - OK
                                accuracy_text = "OK"
                            else:
                                accuracy_color = (0, 0, 255)  # Red - Poor
                                accuracy_text = "POOR"
                            
                            cv2.putText(target_display, f"Accuracy: {accuracy_text}", (10, 120),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, accuracy_color, 2)
            
            # Add status text
            cv2.putText(detection_display, "Validation Mode - Hold ArUco marker ID 77", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(target_display, "Validation Mode - Check transformation accuracy", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(detection_display, "Press: 'v' continue, 'r' recalibrate, 'q' quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(target_display, "Press: 'v' continue, 'r' recalibrate, 'q' quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Detection Camera (Validation)", detection_display)
            cv2.imshow("Target Camera (Validation)", target_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('v'):
                cv2.destroyAllWindows()
                return True  # Continue to pose mapping
            elif key == ord('r'):
                cv2.destroyAllWindows()
                return False  # Recalibrate
            elif key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)
    
    def add_debug_info_to_frame(self, frame: np.ndarray, landmarks: List[Tuple[float, float]], 
                               transformed_landmarks: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
        """
        Add debugging information to frame showing transformation details.
        """
        # Show transformation statistics
        info_y = frame.shape[0] - 100
        
        if landmarks:
            cv2.putText(frame, f"Original landmarks: {len(landmarks)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(transformed_landmarks) > 0:
            cv2.putText(frame, f"Transformed points: {len(transformed_landmarks)}", 
                       (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Valid indices: {len(valid_indices)}", 
                       (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show calibration info
        if self.calibration_manager and self.calibration_manager.is_loaded:
            calib_info = self.calibration_manager.get_calibration_info()
            if calib_info and 'calibration_quality' in calib_info:
                quality = calib_info['calibration_quality']
                if 'reprojection_error' in quality:
                    cv2.putText(frame, f"Calib error: {quality['reprojection_error']:.2f}px", 
                               (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        return frame


def main():
    """
    Main function to run the dual webcam pose mapper.
    """
    print("🎯 Dual Webcam Pose Mapping System")
    print("=" * 50)
    print("This system will:")
    print("1. Calibrate two webcams using ArUco marker")
    print("2. Detect poses on one camera")
    print("3. Map pose coordinates to the other camera")
    print()
    
    # Check for force recalibration flag
    force_recalibrate = len(sys.argv) > 1 and '--recalibrate' in sys.argv
    if force_recalibrate:
        calibration_file = "camera_calibration.json"
        if os.path.exists(calibration_file):
            os.remove(calibration_file)
            print("[LOG]  Removed existing calibration file for fresh calibration")
    
    # Create and run the pose mapper
    try:
        mapper = DualWebcamPoseMapper(
            detection_camera_id=1,  # Camera for pose detection
            target_camera_id=0      # Camera for displaying transformed poses
        )
        mapper.run()
        
    except KeyboardInterrupt:
        print("\n[END] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
