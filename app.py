import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def parse_zone(zone_str: str):
    """Parse polygon points from string: "x1,y1 x2,y2 x3,y3 ..."."""
    points = []
    for p in zone_str.strip().split():
        x, y = p.split(",")
        points.append((int(x), int(y)))
    if len(points) < 3:
        raise ValueError("Polygon zone must have at least 3 points.")
    return np.array(points, dtype=np.int32)


def load_zone_from_args(zone: str, zone_file: str):
    if zone:
        return parse_zone(zone)

    with open(zone_file, "r", encoding="utf-8") as f:
        zone_str = f.read().strip()

    if not zone_str:
        raise ValueError(f"Zone file is empty: {zone_file}")

    return parse_zone(zone_str)


def main():
    parser = argparse.ArgumentParser(description="Count people inside a polygon zone in a video.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="output_zone_count.mp4", help="Output video path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config for YOLO tracking")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")

    zone_group = parser.add_mutually_exclusive_group(required=True)
    zone_group.add_argument("--zone", help='Polygon points: "x1,y1 x2,y2 x3,y3 ..."')
    zone_group.add_argument("--zone-file", help="Path to text file containing zone string")

    args = parser.parse_args()

    zone = load_zone_from_args(args.zone, args.zone_file)
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    enter_time_by_id = {}
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now_sec = frame_index / fps
        result = model.track(frame, conf=args.conf, tracker=args.tracker, persist=True, verbose=False)[0]
        people_in_zone = 0

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            for box, cls_id, track_id in zip(boxes, classes, track_ids):
                # COCO class 0 = person
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box[:4])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                inside = cv2.pointPolygonTest(zone, (cx, cy), False) >= 0
                color = (0, 255, 0) if inside else (0, 0, 255)

                if inside:
                    people_in_zone += 1
                    if track_id is not None:
                        if track_id not in enter_time_by_id:
                            enter_time_by_id[track_id] = now_sec
                        dwell_sec = now_sec - enter_time_by_id[track_id]

                        dwell_text = f"Dwell: {dwell_sec:.1f}s"
                        text_y = max(20, y1 - 10)
                        cv2.putText(
                            frame,
                            dwell_text,
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                else:
                    # Once person exits the zone, reset dwell timer for that ID.
                    if track_id is not None and track_id in enter_time_by_id:
                        enter_time_by_id.pop(track_id, None)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)

        cv2.polylines(frame, [zone], isClosed=True, color=(255, 255, 0), thickness=2)

        cv2.putText(
            frame,
            f"People in zone: {people_in_zone}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (50, 255, 50),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        cv2.imshow("Zone People Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_index += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
