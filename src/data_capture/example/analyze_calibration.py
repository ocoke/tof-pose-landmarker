#!/usr/bin/env python3
"""
Calibration Analysis Tool
This script analyzes existing calibration data and helps diagnose issues.
"""

import json
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path to import from src
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_capture.camera_calibration import CameraCalibrationManager

def analyze_calibration(calibration_file="camera_calibration.json"):
    """Analyze the calibration file and provide detailed information."""
    
    print("🔍 Calibration Analysis Tool")
    print("=" * 50)
    
    if not os.path.exists(calibration_file):
        print(f"[ERROR] Calibration file '{calibration_file}' not found")
        print("   Run the main script first to create calibration data")
        return
    
    # Load and analyze calibration data
    try:
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
        
        print(f"[SUCCESS] Loaded calibration from: {calibration_file}")
        print()
        
        # Basic information
        print("📊 Basic Information:")
        print(f"   Calibration type: {calib_data.get('calibration_type', 'Unknown')}")
        print(f"   Marker ID: {calib_data.get('calibration_marker_id', 'Unknown')}")
        print(f"   Marker size: {calib_data.get('marker_size_meters', 'Unknown')} meters")
        print(f"   ArUco dictionary: {calib_data.get('aruco_dictionary', 'Unknown')}")
        print()
        
        # Homography matrix analysis
        if 'homography_matrix' in calib_data:
            H = np.array(calib_data['homography_matrix'])
            print("🧮 Homography Matrix Analysis:")
            print(f"   Matrix shape: {H.shape}")
            print(f"   Matrix condition number: {np.linalg.cond(H):.2f}")
            print(f"   Determinant: {np.linalg.det(H):.6f}")
            
            # Check if matrix is reasonable
            if np.linalg.cond(H) > 1000:
                print("   [WARNING]  High condition number - calibration may be unstable")
            else:
                print("   [SUCCESS] Condition number looks good")
            
            if abs(np.linalg.det(H)) < 1e-6:
                print("   [WARNING]  Very small determinant - matrix may be singular")
            else:
                print("   [SUCCESS] Determinant looks reasonable")
            
            print()
            print("   Homography Matrix:")
            for i, row in enumerate(H):
                print(f"   [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
            print()
        
        # Quality metrics
        if 'calibration_quality' in calib_data:
            quality = calib_data['calibration_quality']
            print("📈 Calibration Quality Metrics:")
            
            if 'reprojection_error' in quality:
                error = quality['reprojection_error']
                print(f"   Reprojection error: {error:.3f} pixels")
                if error < 2.0:
                    print("   [SUCCESS] Excellent accuracy (< 2 pixels)")
                elif error < 5.0:
                    print("   [SUCCESS] Good accuracy (< 5 pixels)")
                elif error < 10.0:
                    print("   [WARNING]  Moderate accuracy (< 10 pixels)")
                else:
                    print("   [ERROR] Poor accuracy (>= 10 pixels) - consider recalibrating")
            
            if 'coverage_percentage' in quality:
                coverage = quality['coverage_percentage']
                print(f"   Coverage: {coverage:.1f}%")
                if coverage > 80:
                    print("   [SUCCESS] Good coverage")
                elif coverage > 60:
                    print("   [WARNING]  Moderate coverage")
                else:
                    print("   [ERROR] Poor coverage - capture more calibration points")
            
            if 'num_valid_points' in quality:
                points = quality['num_valid_points']
                print(f"   Valid points: {points}")
                if points >= 4:
                    print("   [SUCCESS] Sufficient points for calibration")
                else:
                    print("   [ERROR] Insufficient points - need at least 4")
            
            print()
        
        # Valid region analysis
        if 'valid_region_image1' in calib_data:
            region = calib_data['valid_region_image1']
            print("📍 Valid Region Information:")
            print(f"   X range: {region.get('min_x', 0):.1f} to {region.get('max_x', 0):.1f}")
            print(f"   Y range: {region.get('min_y', 0):.1f} to {region.get('max_y', 0):.1f}")
            
            width = region.get('max_x', 0) - region.get('min_x', 0)
            height = region.get('max_y', 0) - region.get('min_y', 0)
            print(f"   Region size: {width:.1f} x {height:.1f} pixels")
            print()
        
        # Test transformation with sample points
        print("🧪 Sample Transformation Test:")
        manager = CameraCalibrationManager(calibration_file)
        if manager.is_loaded:
            # Test a few sample points
            test_points = [
                (320, 240),  # Center
                (100, 100),  # Top-left area
                (540, 380),  # Bottom-right area
                (160, 360),  # Bottom-left area
                (480, 120),  # Top-right area
            ]
            
            print("   Input → Transformed:")
            for point in test_points:
                transformed, valid_idx = manager.transform_landmarks_rgb_to_tof([point])
                if len(transformed) > 0:
                    tx, ty = transformed[0]
                    print(f"   ({point[0]:3d}, {point[1]:3d}) → ({tx:6.1f}, {ty:6.1f})")
                else:
                    print(f"   ({point[0]:3d}, {point[1]:3d}) → (invalid)")
        else:
            print("   [ERROR] Could not load calibration for testing")
        
        print()
        
        # Recommendations
        print("💡 Recommendations:")
        
        # Check calibration age
        file_age_hours = (time.time() - os.path.getmtime(calibration_file)) / 3600
        if file_age_hours > 24:
            print(f"   [WARNING]  Calibration is {file_age_hours:.1f} hours old - consider recalibrating")
        
        # Check if recalibration is needed based on quality
        needs_recalibration = False
        if 'calibration_quality' in calib_data:
            quality = calib_data['calibration_quality']
            if quality.get('reprojection_error', 0) > 10:
                needs_recalibration = True
                print("   🔄 High reprojection error - recalibration recommended")
            if quality.get('coverage_percentage', 0) < 60:
                needs_recalibration = True
                print("   🔄 Low coverage - capture more calibration points")
        
        if H is not None and np.linalg.cond(H) > 1000:
            needs_recalibration = True
            print("   🔄 Unstable homography matrix - recalibration recommended")
        
        if not needs_recalibration:
            print("   [SUCCESS] Calibration appears to be good")
        else:
            print("   🔄 Run script with --recalibrate flag to create new calibration")
        
        print()
        print("🎯 To recalibrate:")
        print("   python src/data_capture/example/webcam_pair_example.py --recalibrate")
        
    except Exception as e:
        print(f"[ERROR] Error analyzing calibration: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    calibration_file = "camera_calibration.json"
    
    # Check for custom calibration file
    if len(sys.argv) > 1:
        calibration_file = sys.argv[1]
    
    analyze_calibration(calibration_file)

if __name__ == "__main__":
    main()
