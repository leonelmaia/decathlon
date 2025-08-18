import argparse
import logging
import utils as deca
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="video.mp4")
parser.add_argument("--output", default="output_tracked.mp4")
parser.add_argument("--model", default="yolov8n.pt")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--iou", type=float, default=0.5)
args = parser.parse_args()

if __name__ == "__main__":
    model = deca.load_model(args.model)
    logging.info(f"✅ Loaded model {args.model}")

    unique_ids = deca.process_video(args.input, args.output, model, conf=args.conf, iou=args.iou)
    logging.info(f"✅ Saved annotated video to {args.output}")
    logging.info(f"👥 Total unique customers seen: {len(unique_ids)}")
