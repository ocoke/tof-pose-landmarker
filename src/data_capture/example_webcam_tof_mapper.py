#!/usr/bin/env python3
"""
Example script to run the Webcam-ToF Pose Mapper
This script demonstrates how to use the WebcamToFPoseMapper class.
"""

import sys
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

def main():
    print("Webcam-ToF Pose Mapper Example")
    print("=" * 40)
    print("This script will:")
    print("1. Initialize webcam and ToF camera")
    print("2. Set up MediaPipe pose detection")
    print("3. Use confidence map for ArUco detection")
    print("4. Calibrate between webcam and ToF camera")
    print("5. Map pose landmarks from webcam to ToF display")
    print()
    
    try:
        from webcam_tof_pose_mapper import WebcamToFPoseMapper
        
        # Create mapper instance
        # Default: webcam_id=0, camera_range_mm=4000.0, rotate_tof=True
        mapper = WebcamToFPoseMapper(
            webcam_id=0,           # Try webcam ID 1 if 0 doesn't work
            camera_range_mm=4000.0, # ToF range in millimeters
            rotate_tof=True        # Rotate ToF image 180 degrees
        )
        
        print("Starting pose mapping pipeline...")
        print("Note: ArUco detection uses ToF confidence map for better accuracy")
        print()
        
        # Run the complete pipeline
        mapper.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("- opencv-python")
        print("- mediapipe")
        print("- numpy")
        print("- ArducamDepthCamera (for ToF functionality)")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
