import cv2
import glob
import numpy as np
from pathlib import Path

# ----------------------------
# Settings you can edit
# ----------------------------
IMAGE_GLOB = "../data/raw/*.jpg"
SQUARE_SIZE_M = 0.024
OUTPUT_FILE = "calibration_standard.npz"

# If you're unsure, leave this as None and it will auto-pick the best pattern.
# If you're sure, set e.g. FORCE_PATTERN = (7, 7)
FORCE_PATTERN = None

# Common inner-corner patterns to try
PATTERNS_TO_TRY = [
    (7, 7),   # 8x8 squares
    (8, 8),   # 9x9 squares
    (9, 6),
    (10, 7),
    (11, 8),
]

DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)
# ----------------------------


def try_find_corners(gray, pattern):
    """
    Robust chessboard corner detection.
    Returns: (found: bool, corners: np.ndarray or None)
    """
    # Newer, more robust method
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)

    if found:
        return True, corners

    # Fallback to older method (sometimes helps on certain prints)
    flags_old = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found_old, corners_old = cv2.findChessboardCorners(gray, pattern, flags=flags_old)
    if not found_old:
        return False, None

    # Refine to subpixel corners for the old method
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners_refined = cv2.cornerSubPix(gray, corners_old, (11, 11), (-1, -1), criteria)
    return True, corners_refined


def main():
    images = sorted(glob.glob(IMAGE_GLOB))
    print(f"Found {len(images)} images for calibration.")
    if not images:
        raise RuntimeError(f"No images found for glob: {IMAGE_GLOB}")

    # Load first image to confirm resolution
    first = cv2.imread(images[0])
    if first is None:
        raise RuntimeError(f"Could not read first image: {images[0]}")
    image_size = (first.shape[1], first.shape[0])  # (w, h)

    # Decide which pattern to use
    if FORCE_PATTERN is not None:
        patterns = [FORCE_PATTERN]
        print(f"Forcing pattern: {FORCE_PATTERN}")
    else:
        patterns = PATTERNS_TO_TRY
        print(f"Trying patterns: {patterns}")

    # First pass: detect and count successes for each pattern
    pattern_success = {p: 0 for p in patterns}

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Helpful preprocessing for difficult images:
        # - slight blur reduces noise
        # - histogram equalization helps contrast on matte prints
        gray_proc = cv2.GaussianBlur(gray, (3, 3), 0)
        gray_proc = cv2.equalizeHist(gray_proc)

        for pattern in patterns:
            found, corners = try_find_corners(gray_proc, pattern)
            if found:
                pattern_success[pattern] += 1

                # Save a debug image showing corners for the FIRST time each file succeeds
                out = img.copy()
                cv2.drawChessboardCorners(out, pattern, corners, found)
                out_path = DEBUG_DIR / f"{Path(fname).stem}_found_{pattern[0]}x{pattern[1]}.jpg"
                cv2.imwrite(str(out_path), out)

                break  # don’t double-count this image for multiple patterns

    print("Success counts per pattern:")
    for p, c in pattern_success.items():
        print(f"  {p}: {c}/{len(images)}")

    # Pick best pattern if not forced
    if FORCE_PATTERN is None:
        best_pattern = max(pattern_success, key=lambda p: pattern_success[p])
        if pattern_success[best_pattern] == 0:
            raise RuntimeError(
                "No corners detected for any tried pattern.\n"
                "Check: (1) CHECKERBOARD inner-corners vs squares, (2) print quality/contrast, (3) blur/reflections.\n"
                f"Debug images written to: {DEBUG_DIR.resolve()}"
            )
        pattern = best_pattern
        print(f"Chosen pattern: {pattern}")
    else:
        pattern = FORCE_PATTERN
        if pattern_success[pattern] == 0:
            raise RuntimeError(
                f"Forced pattern {pattern} but detected 0 corners.\n"
                f"Debug images written to: {DEBUG_DIR.resolve()}"
            )

    # Second pass: collect points for calibration using the chosen pattern
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints = []
    imgpoints = []
    good = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_proc = cv2.GaussianBlur(gray, (3, 3), 0)
        gray_proc = cv2.equalizeHist(gray_proc)

        found, corners = try_find_corners(gray_proc, pattern)
        if not found:
            continue

        objpoints.append(objp)
        imgpoints.append(corners)
        good += 1

    if good < 3:
        raise RuntimeError(f"Only found corners in {good} images for pattern {pattern}. Need at least 3, ideally 20+.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("RMS reprojection error:", ret)
    print("Camera matrix:\n", camera_matrix)
    print("Dist coeffs:\n", dist_coeffs.ravel())

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size, dtype=np.int32),
        checkerboard=np.array(pattern, dtype=np.int32),
        square_size_m=np.array([SQUARE_SIZE_M], dtype=np.float32),
    )
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Debug visualizations saved to: {DEBUG_DIR.resolve()}")


if __name__ == "__main__":
    main()
