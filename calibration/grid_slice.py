import cv2
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
WARPED_IMAGE = "../data/warped/warped_3_1024.png"   # <-- change to your file
OUT_DIR = "../data/squares"                      # crops go here
OUT_OVERLAY = "../data/warped/warped_with_grid.png"

OUTPUT_SIZE = 1024
MARGIN_PX = 136          # must match what you used in manual warp
BOARD_N = 8              # 8x8 chessboard

DRAW_LABELS = True       # set False if you only want the grid
LABEL_ORIGIN = "a1"      # assumes bottom-left of warped image is a1 (standard chess view)
# If your warp is rotated/flipped, we can adjust mapping easily.
# ----------------------------------------


def square_name(file_idx: int, rank_idx: int) -> str:
    """
    Corrected mapping based on your observed corner correspondences.

    Input:
      file_idx: 0..7 left->right in the warped image
      rank_idx: 0..7 bottom->top in the warped image

    Output:
      algebraic square name (a1..h8) in REAL board coordinates
    """
    real_file_idx = (BOARD_N - 1) - rank_idx  # 7 - rank_idx
    real_rank_idx = file_idx                  # swap

    file_char = chr(ord('a') + real_file_idx)
    rank_char = str(real_rank_idx + 1)
    return f"{file_char}{rank_char}"



def main():
    img = cv2.imread(WARPED_IMAGE)
    if img is None:
        raise RuntimeError(f"Could not read {WARPED_IMAGE}")

    h, w = img.shape[:2]
    if (w, h) != (OUTPUT_SIZE, OUTPUT_SIZE):
        print(f"Warning: image is {w}x{h}, expected {OUTPUT_SIZE}x{OUTPUT_SIZE}. Continuing anyway.")

    overlay = img.copy()

    # Board region inside the margin
    x0 = MARGIN_PX
    y0 = MARGIN_PX
    x1 = w - MARGIN_PX
    y1 = h - MARGIN_PX

    board_w = x1 - x0
    board_h = y1 - y0

    # Use float step to avoid drift; convert each boundary independently
    step_x = board_w / BOARD_N
    step_y = board_h / BOARD_N

    # Draw outer border
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Draw grid lines
    for i in range(1, BOARD_N):
        xi = int(round(x0 + i * step_x))
        yi = int(round(y0 + i * step_y))
        cv2.line(overlay, (xi, y0), (xi, y1), (0, 255, 0), 1)
        cv2.line(overlay, (x0, yi), (x1, yi), (0, 255, 0), 1)

    # Prep output dir
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slice squares.
    #
    # IMPORTANT:
    # - In image coordinates, y increases downward.
    # - In chess, ranks increase upward.
    #
    # We'll define:
    #   file_idx 0..7 = left->right
    #   rank_idx 0..7 = bottom->top
    #
    # That means when slicing image rows, we invert rank to map bottom->top.
    #
    # Square (file_idx, rank_idx) corresponds to:
    #   x: [x0 + file_idx*step_x, x0 + (file_idx+1)*step_x]
    #   y: [y0 + (7-rank_idx)*step_y, y0 + (8-rank_idx)*step_y]
    #
    for file_idx in range(BOARD_N):  # a..h
        for rank_idx in range(BOARD_N):  # 1..8 (bottom->top)
            # Compute crop bounds using rounded boundaries
            xa = int(round(x0 + file_idx * step_x))
            xb = int(round(x0 + (file_idx + 1) * step_x))

            # invert rank for image y
            img_rank = (BOARD_N - 1) - rank_idx
            ya = int(round(y0 + img_rank * step_y))
            yb = int(round(y0 + (img_rank + 1) * step_y))

            crop = img[ya:yb, xa:xb]

            name = square_name(file_idx, rank_idx)
            out_path = out_dir / f"{name}.png"
            cv2.imwrite(str(out_path), crop)

            if DRAW_LABELS:
                # draw label near the top-left of each square region on overlay
                label_x = xa + 5
                label_y = ya + 20
                cv2.putText(
                    overlay, name, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

    # Save overlay
    Path(Path(OUT_OVERLAY).parent).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUT_OVERLAY, overlay)
    print(f"Saved overlay: {OUT_OVERLAY}")
    print(f"Saved 64 crops to: {out_dir.resolve()}")

    # Optional: show overlay for quick sanity check
    cv2.imshow("Grid Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
