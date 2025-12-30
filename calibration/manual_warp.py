import cv2
import numpy as np
from pathlib import Path

# ---- config ----
INPUT_IMAGE = "../data/raw/capture_cam4_20251229_001311_1.jpg"
OUT_WARPED = "../data/warped/warped_3_1024.png"

OUTPUT_SIZE = 1024          # final warped size
MARGIN_PX = 136             # padding around board inside output (prevents clipping)

# Set to None to skip undistortion for now
CALIB_FILE = "calibration_standard.npz"  # or None
# ----------------

clicked = []
display = None
base_img = None

def load_and_maybe_undistort(path: str):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read {path}")

    if CALIB_FILE is None:
        return img

    data = np.load(CALIB_FILE)
    K = data["camera_matrix"]
    D = data["dist_coeffs"]

    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0, newImgSize=(w, h))
    undistorted = cv2.undistort(img, K, D, None, newK)
    return undistorted

def redraw():
    global display
    display = base_img.copy()
    for i, (x, y) in enumerate(clicked, start=1):
        cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(display, str(i), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

def mouse_cb(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked) < 4:
            clicked.append((x, y))
            redraw()

def main():
    global base_img
    base_img = load_and_maybe_undistort(INPUT_IMAGE)
    redraw()

    win = "Click corners (TL, TR, BR, BL). r=reset, Enter=warp, q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            return

        if key == ord('r'):
            clicked.clear()
            redraw()

        if key == 13 or key == 10:  # Enter
            if len(clicked) != 4:
                print("Need exactly 4 clicks: TL, TR, BR, BL")
                continue

            src = np.array(clicked, dtype=np.float32)

            m = float(MARGIN_PX)
            s = float(OUTPUT_SIZE - 1)

            # Board corners map into a padded square region
            dst = np.array([
                [m, m],
                [s - m, m],
                [s - m, s - m],
                [m, s - m]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(base_img, H, (OUTPUT_SIZE, OUTPUT_SIZE))

            Path(Path(OUT_WARPED).parent).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(OUT_WARPED, warped)
            print("Wrote:", OUT_WARPED)

            cv2.imshow("warped", warped)
            cv2.waitKey(0)
            return

if __name__ == "__main__":
    main()
