import argparse
import cv2
import numpy as np


def build_zone_string(points):
    return " ".join(f"{x},{y}" for x, y in points)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively create a polygon zone from a video frame."
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to use for zoning (default: 0)",
    )
    parser.add_argument(
        "--save",
        default="zone.txt",
        help="File to save zone coordinates string (default: zone.txt)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    if args.frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read the selected frame from the video.")

    points = []
    window_name = "Zone Mapper"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        vis = frame.copy()

        for i, (x, y) in enumerate(points):
            cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(
                vis,
                str(i + 1),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if len(points) >= 2:
            poly = np.array(points, dtype=np.int32)
            cv2.polylines(
                vis,
                [poly],
                isClosed=len(points) >= 3,
                color=(255, 255, 0),
                thickness=2,
            )

        msg1 = "Left click: add point | R: reset | C: complete | Q/Esc: quit"
        msg2 = f"Points: {len(points)}"
        cv2.putText(
            vis,
            msg1,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            msg2,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("Exited without saving.")
            return

        if key == ord("r"):
            points.clear()

        if key == ord("c"):
            if len(points) < 3:
                print("Need at least 3 points to form a polygon.")
                continue

            zone_str = build_zone_string(points)
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(zone_str + "\n")

            print("\nZone created successfully.")
            print(f"Zone string: {zone_str}")
            print(f"Saved to: {args.save}")
            print(
                "Run with direct coordinates:\n"
                f'python app.py --input "{args.input}" --output out.mp4 --zone "{zone_str}"'
            )
            print(
                "Run with saved zone file:\n"
                f'python app.py --input "{args.input}" --output out.mp4 --zone-file "{args.save}"'
            )
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main()
