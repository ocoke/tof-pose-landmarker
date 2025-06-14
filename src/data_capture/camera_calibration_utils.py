#!/usr/bin/env python3
"""
Camera calibration utilities for the ToF pose landmarker project.
This module provides functions to load and use the ArUco-based camera calibration.
"""

import cv2
import numpy as np
import json
from pathlib import Path

class CameraCalibrationManager:
    def __init__(self, calibration_file="camera_calibration.json"):
        """
        Initialize the calibration manager
        
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
    
    def load_calibration(self):
        """Load calibration data from file"""
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            self.homography = np.array(self.calibration_data['homography_matrix'], dtype=np.float32)
            self.valid_region = self.calibration_data['valid_region_image1']
            self.is_loaded = True
            
            print(f"✅ Calibration loaded from {self.calibration_file}")
            print(f"📊 Quality metrics:")
            quality = self.calibration_data['calibration_quality']
            print(f"   - Valid points: {quality['num_valid_points']}")
            print(f"   - Coverage: {quality['coverage_percentage']:.1f}%")
            print(f"   - Reprojection error: {quality['reprojection_error']:.3f} pixels")
            
            return True
            
        except FileNotFoundError:
            print(f"⚠️  Calibration file {self.calibration_file} not found")
            print(f"   Please run calibration first")
            return False
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def is_point_in_valid_region(self, x, y):
        """
        Check if a point is within the valid mapping region
        
        Args:
            x, y: Point coordinates in the source image
            
        Returns:
            bool: True if point is in valid region
        """
        if not self.is_loaded:
            return False
        
        return (self.valid_region['min_x'] <= x <= self.valid_region['max_x'] and
                self.valid_region['min_y'] <= y <= self.valid_region['max_y'])
    
    def filter_landmarks_by_region(self, landmarks):
        """
        Filter landmarks to only include those in the valid mapping region
        
        Args:
            landmarks: List of (x, y) tuples or numpy array of shape (N, 2)
            
        Returns:
            tuple: (valid_landmarks, valid_indices)
        """
        if not self.is_loaded:
            return [], []
        
        landmarks = np.array(landmarks)
        if landmarks.size == 0:
            return np.array([]), np.array([])
        
        # Check which landmarks are in valid region
        valid_mask = np.array([
            self.is_point_in_valid_region(x, y) 
            for x, y in landmarks
        ])
        
        valid_landmarks = landmarks[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        return valid_landmarks, valid_indices
    
    def transform_landmarks(self, landmarks):
        """
        Transform landmarks from source camera to target camera coordinates
        
        Args:
            landmarks: List of (x, y) tuples or numpy array of shape (N, 2)
            
        Returns:
            numpy.ndarray: Transformed landmarks, or empty array if transformation fails
        """
        if not self.is_loaded:
            print("❌ Calibration not loaded")
            return np.array([])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        if landmarks.size == 0:
            return np.array([])
        
        # Filter to valid region first
        valid_landmarks, valid_indices = self.filter_landmarks_by_region(landmarks)
        
        if len(valid_landmarks) == 0:
            print("⚠️  No landmarks in valid calibration region")
            return np.array([])
        
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
    
    def get_calibration_info(self):
        """Get calibration information for debugging"""
        if not self.is_loaded:
            return None
        
        return {
            'is_loaded': self.is_loaded,
            'valid_region': self.valid_region,
            'calibration_quality': self.calibration_data['calibration_quality'],
            'homography_condition_number': float(np.linalg.cond(self.homography))
        }

def integrate_with_existing_calibration():
    """
    Example of how to integrate this with the existing calibration.py
    """
    print("🔧 Camera Calibration Integration Example")
    print("=" * 50)
    
    # Initialize calibration manager
    calib_manager = CameraCalibrationManager()
    
    if not calib_manager.is_loaded:
        print("❌ No calibration available. Please run calibration first.")
        return
    
    # Example: Simulate MediaPipe landmarks
    print("\n🧪 Testing with simulated MediaPipe landmarks...")
    
    # These would be real MediaPipe landmarks in your application
    # Format: [(x, y), (x, y), ...] in RGB camera coordinates
    simulated_landmarks = [
        (1000, 1000),   # Example face landmark
        (2000, 1500),   # Example shoulder landmark  
        (3000, 2000),   # Example elbow landmark
        (4000, 2500),   # Example wrist landmark
        (100, 100),     # This one might be outside valid region
        (6000, 4000),   # This one might be outside valid region
    ]
    
    print(f"📍 Original landmarks: {len(simulated_landmarks)} points")
    
    # Transform landmarks
    transformed_landmarks, valid_indices = calib_manager.transform_landmarks(simulated_landmarks)
    
    if len(transformed_landmarks) > 0:
        print(f"✅ Successfully transformed {len(transformed_landmarks)} landmarks")
        print(f"📊 Transformation results:")
        
        for i, (orig_idx, transformed) in enumerate(zip(valid_indices, transformed_landmarks)):
            original = simulated_landmarks[orig_idx]
            print(f"   Landmark {orig_idx}: ({original[0]:.0f},{original[1]:.0f}) → ({transformed[0]:.0f},{transformed[1]:.0f})")
        
        # Show which landmarks were filtered out
        all_indices = set(range(len(simulated_landmarks)))
        filtered_indices = all_indices - set(valid_indices)
        if filtered_indices:
            print(f"⚠️  Filtered out {len(filtered_indices)} landmarks outside valid region:")
            for idx in filtered_indices:
                orig = simulated_landmarks[idx]
                print(f"   Landmark {idx}: ({orig[0]:.0f},{orig[1]:.0f}) - outside valid region")
    
    else:
        print("❌ No landmarks could be transformed")
    
    # Show calibration info
    info = calib_manager.get_calibration_info()
    print(f"\n📋 Calibration Info:")
    print(f"   Valid region: {info['valid_region']['width']:.0f}x{info['valid_region']['height']:.0f} pixels")
    print(f"   Quality: {info['calibration_quality']['coverage_percentage']:.1f}% coverage")

def create_calibration_integration_code():
    """
    Generate code snippet for integrating with calibration.py
    """
    code_snippet = '''
# Add this to your DualCameraPoseMapper class in calibration.py

from camera_calibration_utils import CameraCalibrationManager

class DualCameraPoseMapper:
    def __init__(self):
        # ... existing initialization code ...
        
        # Add calibration manager
        self.calibration_manager = CameraCalibrationManager("camera_calibration.json")
        
        # Remove the old homography calculation code and use the pre-computed one
        # self.homography_rgb_to_tof = None  # Remove this line
        
    def run(self):
        # ... existing code until MediaPipe landmark extraction ...
        
        if mp_results.pose_landmarks:
            # ... existing MediaPipe drawing code ...
            
            # Collect landmarks for transformation
            mp_landmarks_2d_rgb_pixels = []
            for landmark_idx, landmark in enumerate(mp_results.pose_landmarks.landmark):
                if landmark.visibility < 0.5:
                    continue
                lx = landmark.x * self.webcam_width
                ly = landmark.y * self.webcam_height
                mp_landmarks_2d_rgb_pixels.append((lx, ly))
            
            # Use the pre-computed calibration instead of real-time ArUco detection
            if len(mp_landmarks_2d_rgb_pixels) > 0 and self.calibration_manager.is_loaded:
                transformed_landmarks, valid_indices = self.calibration_manager.transform_landmarks(
                    mp_landmarks_2d_rgb_pixels
                )
                
                # Draw transformed landmarks on ToF display
                for pt_tof in transformed_landmarks:
                    x, y = int(pt_tof[0]), int(pt_tof[1])
                    if 0 <= x < tof_display_bgr.shape[1] and 0 <= y < tof_display_bgr.shape[0]:
                        cv2.circle(tof_display_bgr, (x, y), 3, (0, 255, 0), -1)
'''
    
    with open("calibration_integration_example.py", "w") as f:
        f.write(code_snippet)
    
    print(f"📝 Integration code saved to: calibration_integration_example.py")

def main():
    print("🎯 Camera Calibration Utils")
    print("=" * 40)
    
    # Test the integration
    integrate_with_existing_calibration()
    
    # Create integration example
    print(f"\n📝 Creating integration example...")
    create_calibration_integration_code()
    
    print(f"\n✅ Calibration utilities ready!")
    print(f"📁 Files created:")
    print(f"   - camera_calibration.json (calibration data)")
    print(f"   - calibration_integration_example.py (integration code)")
    print(f"   - valid_region_image1.jpg, valid_region_image2.jpg (visualizations)")

if __name__ == "__main__":
    main()
