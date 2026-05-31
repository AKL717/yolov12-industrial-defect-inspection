import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_SRC = ROOT / "third_party" / "ultralytics"
if ULTRALYTICS_SRC.exists():
    sys.path.insert(0, str(ULTRALYTICS_SRC))

from ultralytics import YOLO


DATA = ROOT / "data" / "datasets" / "NEU-DET" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv12 on NEU-DET.")
    parser.add_argument("--weights", default="yolo12n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="neu_yolo12n")
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=str(DATA),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        device=args.device,
        workers=0,
    )


if __name__ == "__main__":
    main()
