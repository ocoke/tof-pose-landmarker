#!/usr/bin/env python3
"""
Camera Detection Test Script
This script helps identify available cameras and their capabilities.
"""

import cv2
import sys

def test_cameras():
    """Test and list available cameras."""
    print("🔍 Detecting available cameras...")
    print("=" * 40)
    
    available_cameras = []
    
    for camera_id in range(10):
        # Test first 10 camera index
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to read a frame to confirm camera works
            ret, frame = cap.read()
            
            if ret:
                available_cameras.append(camera_id)
                print(f"[SUCCESS] Camera {camera_id}:")
                print(f"   Resolution: {width}x{height}")
                print(f"   FPS: {fps}")
                print(f"   Frame captured: {frame.shape if frame is not None else 'Failed'}")
            else:
                print(f"[WARNING]  Camera {camera_id}: Detected but cannot capture frames")
            
            cap.release()
        
    if not available_cameras:
        print("[ERROR] No working cameras found!")
        return False
    
    print(f"\n📊 Summary: {len(available_cameras)} working camera(s) found")
    print(f"Available camera IDs: {available_cameras}")
    
    # Recommendations
    if len(available_cameras) >= 2:
        print(f"   Detection camera ID: {available_cameras[1]}")
        print(f"   Target camera ID: {available_cameras[0]}")
    else:
        print(f"\n[WARNING]  Need at least 2 cameras for dual webcam pose mapping")
        print(f"   Currently found: {len(available_cameras)} camera(s)")
    
    return True

def test_specific_cameras(detection_id=1, target_id=0):
    # Test specific camera IDs for the dual webcam setup
    print(f"\n🔧 Testing specific camera setup:")
    print(f"   Detection camera ID: {detection_id}")
    print(f"   Target camera ID: {target_id}")
    print("=" * 40)
    
    # Test detection camera
    det_cap = cv2.VideoCapture(detection_id)
    if not det_cap.isOpened():
        print(f"[ERROR] Detection camera (ID: {detection_id}) failed to open")
        return False
    
    # Test target camera
    tar_cap = cv2.VideoCapture(target_id)
    if not tar_cap.isOpened():
        print(f"[ERROR] Target camera (ID: {target_id}) failed to open")
        det_cap.release()
        return False
    
    print("[SUCCESS] Both cameras opened successfully")
    
    # Test frame capture
    ret1, frame1 = det_cap.read()
    ret2, frame2 = tar_cap.read()
    
    if ret1 and ret2:
        print("[SUCCESS]  Both cameras can capture frames")
        print(f"   Detection camera frame: {frame1.shape}")
        print(f"   Target camera frame: {frame2.shape}")
    else:
        print("[ERROR] One or both cameras cannot capture frames")
        det_cap.release()
        tar_cap.release()
        return False
    
    # Brief live test
    print("\n🎥 Starting brief live test (5 seconds)...")
    print("   Press any key to stop early")
    
    cv2.namedWindow("Detection Camera Test", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Target Camera Test", cv2.WINDOW_AUTOSIZE)
    
    import time
    start_time = time.time()
    
    while time.time() - start_time < 5:
        ret1, frame1 = det_cap.read()
        ret2, frame2 = tar_cap.read()
        
        if ret1 and ret2:
            cv2.imshow("Detection Camera Test", frame1)
            cv2.imshow("Target Camera Test", frame2)
            
            if cv2.waitKey(1) & 0xFF != 255:  # Any key pressed
                break
        else:
            print("[ERROR] Frame capture failed during test")
            break
    
    # Cleanup
    det_cap.release()
    tar_cap.release()
    cv2.destroyAllWindows()
    
    print("[SUCCESS] Camera test completed successfully")
    return True

def main():
    """Main function."""
    print("🎥 Camera Detection and Testing Tool")
    print("This tool helps prepare for dual webcam pose mapping")
    print()
    
    # Test all available cameras
    if not test_cameras():
        return
    
    # Test specific camera configuration
    print("\n" + "="*50)
    choice = input("Test specific camera configuration? (y/n): ").strip().lower()
    
    if choice == 'y':
        try:
            det_id = int(input("Enter detection camera ID (default: 1): ") or "1")
            tar_id = int(input("Enter target camera ID (default: 0): ") or "0")
            test_specific_cameras(det_id, tar_id)
        except ValueError:
            print("[ERROR] Invalid camera ID entered")
        except KeyboardInterrupt:
            print("\n[END] Test interrupted by user")
    
    print("\n🎯 Ready to run dual webcam pose mapping!")
    print("Use: python src/data_capture/example/webcam_pair_example.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[END] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
