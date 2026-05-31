import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_SRC = ROOT / "third_party" / "ultralytics"
if ULTRALYTICS_SRC.exists():
    sys.path.insert(0, str(ULTRALYTICS_SRC))

from ultralytics import YOLO


DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "neu_yolo12n" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained YOLOv12 weights to ONNX.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "neu_yolo12n.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX with onnxslim when available.")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Cannot find trained weights: {weights}. Run scripts/train_yolo12_neu.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=args.simplify, dynamic=False)
    exported_path = Path(exported)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(exported_path.read_bytes())
    print(f"ONNX model saved to: {target}")


if __name__ == "__main__":
    main()
