import cv2
import os
import argparse
from datetime import datetime

def detect_cameras(max_cameras=10):
    """Detect all available cameras."""
    available_cameras = []
    
    print("Detecting cameras...")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Set to 1080p
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try to read a frame to verify it works
            ret, _ = cap.read()
            if ret:
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'cap': cap
                })
                print(f"  Camera {i}: {width}x{height}")
            else:
                cap.release()
        else:
            # Stop checking if we get 3 consecutive failures
            if i > 2 and len(available_cameras) == 0:
                break
    
    print(f"\nFound {len(available_cameras)} camera(s)\n")
    return available_cameras

def preview_and_select_camera(cameras):
    """Show preview of each camera and let user select the correct one."""
    
    for cam_info in cameras:
        cap = cam_info['cap']
        index = cam_info['index']
        
        print(f"\n--- Previewing Camera {index} ({cam_info['width']}x{cam_info['height']}) ---")
        print("Press SPACE to skip to next camera")
        print("Press ENTER to select this camera")
        print("Press ESC to exit without saving")
        
        window_name = f"Camera {index} - Press ENTER to select, SPACE to skip, ESC to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        selected = False
        skip = False
        
        # Warm up camera
        for _ in range(10):
            cap.read()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error: Could not read from camera {index}")
                break
            
            # Add text overlay
            display_frame = frame.copy()
            text = f"Camera {index} ({cam_info['width']}x{cam_info['height']})"
            cv2.putText(display_frame, text, (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display_frame, "ENTER=Select  SPACE=Skip  ESC=Exit", (30, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter key
                selected = True
                break
            elif key == 32:  # Space key
                skip = True
                break
            elif key == 27:  # ESC key
                cv2.destroyAllWindows()
                return None, None
        
        cv2.destroyWindow(window_name)
        
        if selected:
            print(f"\n✓ Selected Camera {index}")
            return index, cap
        elif skip:
            print(f"  Skipped Camera {index}")
            continue
    
    print("\nNo camera selected.")
    return None, None

def capture_from_camera(camera_index):
    """Capture image directly from specified camera with live preview."""
    print(f"Using camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return []
    
    # Set to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {width}x{height}")
    
    # Warm up camera
    print("Warming up camera...")
    for _ in range(10):
        cap.read()
    
    print("\n--- Camera Ready ---")
    print("Press ENTER or SPACE to capture image (can capture multiple times)")
    print("Press BACKSPACE to save and exit")
    print("Press Ctrl+C to exit")
    
    window_name = f"Camera {camera_index} - Press ENTER/SPACE to capture, BACKSPACE to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    captured_frames = []
    capture_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error: Could not read from camera {camera_index}")
                break
            
            # Add text overlay
            display_frame = frame.copy()
            text = f"Camera {camera_index} ({width}x{height}) - Captured: {capture_count}"
            cv2.putText(display_frame, text, (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display_frame, "ENTER/SPACE=Capture  BACKSPACE=Exit", (30, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == 32:  # Enter or Space key
                captured_frames.append(frame.copy())
                capture_count += 1
                print(f"✓ Captured image #{capture_count}")
            elif key == 8 or key == 127:  # Backspace or Delete key
                print("\nExiting...")
                break
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
    
    return captured_frames

def capture_image(detect_mode=False, default_camera=4):
    """Detect cameras, show preview, and save selected image."""
    
    # Create images folder if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'data/raw')
    os.makedirs(images_dir, exist_ok=True)
    
    captured_frames = []
    selected_index = default_camera
    
    if detect_mode:
        # Detection mode: show preview and let user select
        cameras = detect_cameras()
        
        if not cameras:
            print("Error: No cameras detected")
            return []
        
        # Preview cameras and let user select
        selected_index, selected_cap = preview_and_select_camera(cameras)
        
        # Release all non-selected cameras
        for cam_info in cameras:
            if cam_info['index'] != selected_index:
                cam_info['cap'].release()
        
        if selected_cap is None:
            print("No camera selected.")
            # Release remaining cameras
            for cam_info in cameras:
                cam_info['cap'].release()
            cv2.destroyAllWindows()
            return []
        
        # Now use the selected camera for multiple captures
        width = int(selected_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(selected_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n--- Camera {selected_index} Ready ---")
        print("Press ENTER or SPACE to capture image (can capture multiple times)")
        print("Press BACKSPACE to save and exit")
        print("Press Ctrl+C to exit")
        
        window_name = f"Camera {selected_index} - Press ENTER/SPACE to capture, BACKSPACE to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        capture_count = 0
        
        try:
            while True:
                ret, frame = selected_cap.read()
                
                if not ret:
                    print(f"Error: Could not read from camera {selected_index}")
                    break
                
                # Add text overlay
                display_frame = frame.copy()
                text = f"Camera {selected_index} ({width}x{height}) - Captured: {capture_count}"
                cv2.putText(display_frame, text, (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(display_frame, "ENTER/SPACE=Capture  BACKSPACE=Exit", (30, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 13 or key == 32:  # Enter or Space key
                    captured_frames.append(frame.copy())
                    capture_count += 1
                    print(f"✓ Captured image #{capture_count}")
                elif key == 8 or key == 127:  # Backspace or Delete key
                    print("\nExiting...")
                    break
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        finally:
            selected_cap.release()
            cv2.destroyAllWindows()
    else:
        # Direct mode: use default camera
        captured_frames = capture_from_camera(default_camera)
        
    if not captured_frames:
        print("No images captured.")
        return []
    
    # Save all captured images
    saved_files = []
    for i, frame in enumerate(captured_frames, 1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_cam{selected_index}_{timestamp}_{i}.jpg"
        filepath = os.path.join(images_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"✓ Image {i} saved: {filepath}")
        saved_files.append(filepath)
    
    print(f"\n✓ Total images saved: {len(saved_files)}")
    return saved_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture image from camera')
    parser.add_argument('-d', '--detect', action='store_true',
                        help='Enable detection mode to preview and select camera')
    args = parser.parse_args()
    
    capture_image(detect_mode=args.detect)
