#!/usr/bin/env python3
"""
Production Camera Calibration Manager for ToF-RGB Camera System
This module handles the calibration between Time-of-Flight and RGB cameras
using pre-computed ArUco-based homography transformation.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, List, Optional, Union

class CameraCalibrationManager:
    """
    Manages calibration between ToF and RGB cameras for landmark transformation.
    Uses pre-computed homography from ArUco marker calibration.
    """
    
    def __init__(self, calibration_file: str = "camera_calibration.json"):
        """
        Initialize the calibration manager.
        
        Args:
            calibration_file: Path to the calibration JSON file
        """
        self.calibration_file = calibration_file
        self.homography = None
        self.valid_region = None
        self.calibration_data = None
        self.is_loaded = False
        
        # Try to load calibration automatically
        self.load_calibration()
    
    def load_calibration(self) -> bool:
        """
        Load calibration data from file.
        
        Returns:
            bool: True if calibration loaded successfully
        """
        try:
            # Check multiple possible locations
            possible_paths = [
                self.calibration_file,
                f"../../{self.calibration_file}",  # Main project directory
                f"src/data_capture/{self.calibration_file}",
                f"data_capture/{self.calibration_file}"
            ]
            
            calibration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    calibration_path = path
                    break
            
            if calibration_path is None:
                print(f"❌ Calibration file not found in any expected location:")
                for path in possible_paths:
                    print(f"   - {path}")
                return False
            
            with open(calibration_path, 'r') as f:
                self.calibration_data = json.load(f)
            
            self.homography = np.array(self.calibration_data['homography_matrix'], dtype=np.float32)
            self.valid_region = self.calibration_data.get('valid_region_image1', None)
            self.is_loaded = True
            
            print(f"✅ Calibration loaded from {calibration_path}")
            
            # Print quality metrics if available
            if 'calibration_quality' in self.calibration_data:
                quality = self.calibration_data['calibration_quality']
                print(f"📊 Calibration Quality:")
                print(f"   - Reprojection error: {quality.get('reprojection_error', 'N/A'):.3f} pixels")
                print(f"   - Coverage: {quality.get('coverage_percentage', 'N/A'):.1f}%")
                print(f"   - Valid points: {quality.get('num_valid_points', 'N/A')}")
            
            return True
            
        except FileNotFoundError:
            print(f"⚠️  Calibration file {self.calibration_file} not found")
            print(f"   Please run calibration setup first")
            return False
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def is_point_in_valid_region(self, x: float, y: float) -> bool:
        """
        Check if a point is within the valid mapping region.
        
        Args:
            x, y: Point coordinates in the RGB camera image
            
        Returns:
            bool: True if point is in valid calibration region
        """
        if not self.is_loaded or self.valid_region is None:
            # If no valid region defined, assume all points are valid
            return True
        
        return (self.valid_region['min_x'] <= x <= self.valid_region['max_x'] and
                self.valid_region['min_y'] <= y <= self.valid_region['max_y'])
    
    def filter_landmarks_by_region(self, landmarks: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter landmarks to only include those in the valid mapping region.
        
        Args:
            landmarks: List of (x, y) tuples or numpy array of shape (N, 2)
            
        Returns:
            tuple: (valid_landmarks, valid_indices)
        """
        if not self.is_loaded:
            return np.array([]), np.array([])
        
        landmarks = np.array(landmarks)
        if landmarks.size == 0:
            return np.array([]), np.array([])
        
        # Reshape if needed
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 2)
        
        # Check which landmarks are in valid region
        valid_mask = np.array([
            self.is_point_in_valid_region(x, y) 
            for x, y in landmarks
        ])
        
        valid_landmarks = landmarks[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        return valid_landmarks, valid_indices
    
    def transform_landmarks_rgb_to_tof(self, rgb_landmarks: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform landmarks from RGB camera coordinates to ToF camera coordinates.
        
        Args:
            rgb_landmarks: List of (x, y) tuples or numpy array of shape (N, 2)
            
        Returns:
            tuple: (transformed_landmarks, valid_indices)
        """
        if not self.is_loaded:
            print("❌ Calibration not loaded")
            return np.array([]), np.array([])
        
        rgb_landmarks = np.array(rgb_landmarks, dtype=np.float32)
        if rgb_landmarks.size == 0:
            return np.array([]), np.array([])
        
        # Ensure proper shape
        if rgb_landmarks.ndim == 1:
            rgb_landmarks = rgb_landmarks.reshape(-1, 2)
        
        # Filter to valid region first
        valid_landmarks, valid_indices = self.filter_landmarks_by_region(rgb_landmarks)
        
        if len(valid_landmarks) == 0:
            return np.array([]), np.array([])
        
        # Transform valid landmarks
        try:
            transformed_landmarks = cv2.perspectiveTransform(
                valid_landmarks.reshape(1, -1, 2), 
                self.homography
            )[0]
            
            return transformed_landmarks, valid_indices
            
        except Exception as e:
            print(f"❌ Error transforming landmarks: {e}")
            return np.array([]), np.array([])
    
    def get_calibration_info(self) -> Optional[dict]:
        """Get calibration information for debugging."""
        if not self.is_loaded:
            return None
        
        info = {
            'is_loaded': self.is_loaded,
            'calibration_file': self.calibration_file,
            'homography_condition_number': float(np.linalg.cond(self.homography))
        }
        
        if self.valid_region:
            info['valid_region'] = self.valid_region
            
        if 'calibration_quality' in self.calibration_data:
            info['calibration_quality'] = self.calibration_data['calibration_quality']
            
        return info

class ArUcoCalibrationSetup:
    """
    Handles the initial ArUco-based calibration setup between cameras.
    This should be run once to establish the camera-to-camera transformation.
    """
    
    def __init__(self):
        """Initialize ArUco calibration setup."""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters()
        
        self.CALIBRATION_MARKER_ID = 77
        self.MARKER_SIZE_METERS = 0.18
        
        print("🎯 ArUco Calibration Setup initialized")
        print(f"   Looking for marker ID: {self.CALIBRATION_MARKER_ID}")
    
    def detect_aruco_marker(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ArUco markers in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            tuple: (corners, ids) or (None, None) if no markers found
        """
        if image is None:
            return None, None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect markers
        try:
            # Try newer OpenCV API
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        except AttributeError:
            # Try older OpenCV API
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)
        
        return corners, ids
    
    def calculate_homography_from_frames(self, rgb_frame: np.ndarray, tof_confidence_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate homography from RGB frame and ToF confidence frame.
        
        Args:
            rgb_frame: RGB camera frame
            tof_confidence_frame: ToF confidence map frame
            
        Returns:
            numpy.ndarray: Homography matrix or None if calculation fails
        """
        print("🔍 Detecting ArUco markers in both frames...")
        
        # Detect markers in both frames
        rgb_corners, rgb_ids = self.detect_aruco_marker(rgb_frame)
        tof_corners, tof_ids = self.detect_aruco_marker(tof_confidence_frame)
        
        if rgb_ids is None or tof_ids is None:
            print("❌ ArUco markers not detected in one or both frames")
            return None
        
        # Find common marker IDs
        common_ids = set(rgb_ids.flatten()).intersection(set(tof_ids.flatten()))
        
        if self.CALIBRATION_MARKER_ID not in common_ids:
            print(f"❌ Calibration marker ID {self.CALIBRATION_MARKER_ID} not found in both frames")
            print(f"   RGB frame markers: {rgb_ids.flatten()}")
            print(f"   ToF frame markers: {tof_ids.flatten()}")
            return None
        
        # Get marker corners
        rgb_idx = np.where(rgb_ids.flatten() == self.CALIBRATION_MARKER_ID)[0][0]
        tof_idx = np.where(tof_ids.flatten() == self.CALIBRATION_MARKER_ID)[0][0]
        
        rgb_points = rgb_corners[rgb_idx][0].astype(np.float32)
        tof_points = tof_corners[tof_idx][0].astype(np.float32)
        
        # Calculate homography
        try:
            H, status = cv2.findHomography(rgb_points, tof_points, cv2.RANSAC, 5.0)
            if H is not None:
                # Calculate reprojection error
                reproj_points = cv2.perspectiveTransform(rgb_points.reshape(1, -1, 2), H)
                error = np.mean(np.sqrt(np.sum((reproj_points[0] - tof_points)**2, axis=1)))
                
                print(f"✅ Homography calculated successfully!")
                print(f"   Reprojection error: {error:.3f} pixels")
                
                return H
            else:
                print("❌ Homography calculation failed")
                return None
                
        except Exception as e:
            print(f"❌ Error calculating homography: {e}")
            return None
    
    def save_calibration(self, homography: np.ndarray, output_path: str = "camera_calibration.json") -> bool:
        """
        Save calibration data to file.
        
        Args:
            homography: Computed homography matrix
            output_path: Path to save calibration file
            
        Returns:
            bool: True if saved successfully
        """
        try:
            calibration_data = {
                "homography_matrix": homography.tolist(),
                "calibration_marker_id": self.CALIBRATION_MARKER_ID,
                "marker_size_meters": self.MARKER_SIZE_METERS,
                "aruco_dictionary": "DICT_6X6_250",
                "calibration_type": "rgb_to_tof_homography",
                "setup_instructions": [
                    "This calibration maps RGB camera coordinates to ToF camera coordinates",
                    "Use CameraCalibrationManager to load and apply this calibration",
                    "Ensure ArUco marker ID 77 is visible to both cameras during initial setup"
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"✅ Calibration saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False

# Convenience functions for easy integration
def load_calibration(calibration_file: str = "camera_calibration.json") -> Optional[CameraCalibrationManager]:
    """
    Load calibration manager with error handling.
    
    Args:
        calibration_file: Path to calibration file
        
    Returns:
        CameraCalibrationManager instance or None if loading fails
    """
    manager = CameraCalibrationManager(calibration_file)
    return manager if manager.is_loaded else None

def transform_landmarks(rgb_landmarks: Union[List, np.ndarray], 
                       calibration_file: str = "camera_calibration.json") -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick function to transform landmarks using calibration.
    
    Args:
        rgb_landmarks: Landmarks in RGB camera coordinates
        calibration_file: Path to calibration file
        
    Returns:
        tuple: (transformed_landmarks, valid_indices)
    """
    manager = load_calibration(calibration_file)
    if manager is None:
        return np.array([]), np.array([])
    
    return manager.transform_landmarks_rgb_to_tof(rgb_landmarks)
