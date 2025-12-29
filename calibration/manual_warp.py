import cv2
import numpy as np
from pathlib import Path

# ---- config ----
INPUT_IMAGE = "../data/raw/capture_cam4_20251229_001311_1.jpg"
OUT_WARPED = "../data/warped/warped_2_1024.png"
SIZE = 1024
# ----------------

clicked = []
display = None

def mouse_cb(event, x, y, flags, param):
    global clicked, display
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(display, str(len(clicked)), (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

def main():
    global display

    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        raise RuntimeError(f"Could not read {INPUT_IMAGE}")

    display = img.copy()
    cv2.namedWindow("Click corners (TL, TR, BR, BL). Press r=reset, Enter=warp, q=quit", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click corners (TL, TR, BR, BL). Press r=reset, Enter=warp, q=quit", mouse_cb)

    while True:
        cv2.imshow("Click corners (TL, TR, BR, BL). Press r=reset, Enter=warp, q=quit", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            return
        if key == ord('r'):
            clicked.clear()
            display = img.copy()
        if key == 13 or key == 10:  # Enter
            if len(clicked) != 4:
                print("Need exactly 4 clicks: TL, TR, BR, BL")
                continue

            src = np.array(clicked, dtype=np.float32)
            dst = np.array([
                [0, 0],
                [SIZE - 1, 0],
                [SIZE - 1, SIZE - 1],
                [0, SIZE - 1]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, H, (SIZE, SIZE))

            Path(Path(OUT_WARPED).parent).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(OUT_WARPED, warped)
            print("Wrote:", OUT_WARPED)

            # show result
            cv2.imshow("warped", warped)
            cv2.waitKey(0)
            return

if __name__ == "__main__":
    main()
