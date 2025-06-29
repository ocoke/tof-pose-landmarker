"""
Main script for capturing data for model training.
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
import ArducamDepthCamera as ac

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

from camera_calibration import CameraCalibrationManager, ArUcoCalibrationSetup

from webcam_tof_pose_mapper import WebcamToFPoseMapper

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
        

        # Initialize cameras
        if not mapper.initialize_cameras():
            return
        
        # Initialize MediaPipe
        if not mapper.initialize_mediapipe():
            return
        
        # Try to load existing calibration
        if not mapper.load_existing_calibration():
            # Run live calibration if no existing calibration
            if not mapper.phase1_live_calibration():
                print("[ERROR] Calibration failed. Cannot proceed with pose mapping.")
                return
            
            # Load the newly created calibration
            if not mapper.load_existing_calibration():
                print("[ERROR] Failed to load newly created calibration")
                return
        
        # Validate calibration before proceeding
        print("\nValidating calibration accuracy...")
        while True:
            if mapper.validate_calibration_realtime():
                break  # Validation passed, continue to pose mapping
            else:
                # User chose to recalibrate
                print("[LOG] Recalibrating...")
                if not mapper.phase1_live_calibration():
                    print("[ERROR] Recalibration failed. Cannot proceed with pose mapping.")
                    return
                
                # Load the newly created calibration
                if not mapper.load_existing_calibration():
                    print("[ERROR] Failed to load newly created calibration")
                    return
        
        # Run real-time pose mapping
        mapper.capture_train_data()
        
        # Cleanup
        mapper.cleanup()



    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
