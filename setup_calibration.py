#!/usr/bin/env python3
"""
ArUco Calibration Setup Script
Run this once to establish calibration between RGB and ToF cameras.
"""

import sys
import os
sys.path.append('src/data_capture')

from camera_calibration import ArUcoCalibrationSetup
import cv2
import numpy as np

def print_instructions():
    """Print setup instructions."""
    print("=" * 60)
    print("📷 DUAL CAMERA ARUCO CALIBRATION SETUP")
    print("=" * 60)
    print()
    print("REQUIREMENTS:")
    print("• ArUco marker ID 77 (from DICT_6X6_250 dictionary)")
    print("• Both RGB and ToF cameras connected and working")
    print("• Marker visible to both cameras simultaneously")
    print()
    print("INSTRUCTIONS:")
    print("1. Print the ArUco marker ID 77 (you can generate it online)")
    print("2. Place marker where both cameras can see it clearly")
    print("3. Press SPACE when marker is well-positioned")
    print("4. Press 'q' to quit")
    print()
    print("=" * 60)

def generate_aruco_marker():
    """Generate and save ArUco marker for convenience."""
    print("📝 Generating ArUco marker ID 77...")
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_size = 200  # pixels
    
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 77, marker_size)
    
    # Add border and instructions
    border = 50
    final_img = np.ones((marker_size + 2*border, marker_size + 2*border), dtype=np.uint8) * 255
    final_img[border:border+marker_size, border:border+marker_size] = marker_img
    
    # Add text
    cv2.putText(final_img, "ArUco Marker ID: 77", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(final_img, "For ToF-RGB Calibration", (10, marker_size + border + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    filename = "aruco_marker_77_calibration.png"
    cv2.imwrite(filename, final_img)
    print(f"[SUCCESS]  ArUco marker saved as: {filename}")
    print(f"   Print this marker and place it where both cameras can see it")
    
    return filename

def main():
    print_instructions()
    
    # Generate marker if it doesn't exist
    marker_file = "aruco_marker_77_calibration.png"
    if not os.path.exists(marker_file):
        generate_aruco_marker()
        print()
        input("Press ENTER after printing the marker and setting it up...")
        print()
    
    # Initialize calibration setup
    calibration_setup = ArUcoCalibrationSetup()
    
    # For now, we'll work with test images
    # In a real setup, this would capture from actual cameras
    print("🚧 DEVELOPMENT MODE:")
    print("   This script is configured for test images.")
    print("   For production use, integrate with actual camera capture.")
    print()
    
    # Check for test images
    test_images = [
        "test_src/1.jpg",  # RGB test image
        "test_src/2.jpg",  # ToF test image
    ]
    
    missing_images = [img for img in test_images if not os.path.exists(img)]
    
    if missing_images:
        print("❌ Test images not found:")
        for img in missing_images:
            print(f"   {img}")
        print()
        print("For production use, you would:")
        print("1. Connect RGB camera (webcam)")
        print("2. Connect ToF camera (Arducam)")
        print("3. Capture frames from both cameras")
        print("4. Run calibration on live frames")
        
        # Create sample calibration data
        print()
        print("📁 Creating sample calibration data...")
        
        # Sample homography matrix (replace with real calibration)
        sample_homography = np.array([
            [0.8745, 0.1234, -15.2345],
            [-0.0987, 0.9234, 8.7654],
            [0.0001, 0.0003, 1.0000]
        ], dtype=np.float32)
        
        success = calibration_setup.save_calibration(sample_homography, "camera_calibration.json")
        if success:
            print("[SUCCESS]  Sample calibration created for testing")
            print("   Replace with real calibration data when cameras are available")
        
        return
    
    # Load test images
    print("📸 Loading test images...")
    rgb_frame = cv2.imread(test_images[0])
    tof_frame = cv2.imread(test_images[1])
    
    if rgb_frame is None or tof_frame is None:
        print("❌ Failed to load test images")
        return
    
    print("[SUCCESS]  Test images loaded")
    
    # Calculate homography
    homography = calibration_setup.calculate_homography_from_frames(rgb_frame, tof_frame)
    
    if homography is not None:
        # Save calibration
        success = calibration_setup.save_calibration(homography, "camera_calibration.json")
        if success:
            print()
            print("🎉 CALIBRATION COMPLETE!")
            print("   You can now run the main calibration.py script")
            print("   The system will use pre-computed calibration for landmark transformation")
        else:
            print("❌ Failed to save calibration")
    else:
        print("❌ Calibration failed")
        print("   Check that ArUco marker ID 77 is visible in both images")

if __name__ == "__main__":
    main()
