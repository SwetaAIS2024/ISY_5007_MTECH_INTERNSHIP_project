import argparse
import os
from typing import Dict, Iterable, List, Optional, Set

import cv2
import numpy as np
from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv
from typing import Tuple

from zone_annotation import annotate_zones  # Import the annotation module

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119"])

#COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])




# # Replace your existing ZONE_IN_POLYGONS and ZONE_OUT_POLYGONS with:
# NORMALIZED_ZONE_IN_POLYGONS = [
#     np.array([[0.04, 0.84], [0.14, 0.6], [0.43, 0.63], [0.39, 0.97]]),
#     np.array([[0.41, 0.95], [0.44, 0.63], [0.57, 0.64], [0.58, 0.99]]),
#     np.array([[0.59, 0.94], [0.72, 0.61], [0.85, 0.84], [0.84, 0.99]])
# ]

# NORMALIZED_ZONE_OUT_POLYGONS = [
#     np.array([[0.39, 0.26], [0.42, 0.58], [0.18, 0.53], [0.28, 0.24]]),
#     np.array([[0.43, 0.57], [0.4, 0.26], [0.48, 0.23], [0.59, 0.58]]),
#     np.array([[0.74, 0.56], [0.91, 0.42], [0.97, 0.52], [0.86, 0.77]])
# ]



class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        roboflow_api_key: str,
        model_id: str,
        source_video_path: str,
        target_video_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        normalized_zone_in_polygons: List[np.ndarray] = None,
        normalized_zone_out_polygons: List[np.ndarray] = None,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = get_roboflow_model(model_id=model_id, api_key=roboflow_api_key)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        # Use provided polygons or annotate dynamically
        if normalized_zone_in_polygons is None or normalized_zone_out_polygons is None:
            print("Annotating zones dynamically...")
            annotated_polygons = annotate_zones(source_video_path)
            num_zones = len(annotated_polygons) // 2
            normalized_zone_in_polygons = annotated_polygons[:num_zones]
            normalized_zone_out_polygons = annotated_polygons[num_zones:]

        # Denormalize polygons based on video dimensions
        self.ZONE_IN_POLYGONS = self.denormalize_polygons(
            normalized_zone_in_polygons,
            (self.video_info.width, self.video_info.height),
        )
        self.ZONE_OUT_POLYGONS = self.denormalize_polygons(
            normalized_zone_out_polygons,
            (self.video_info.width, self.video_info.height),
        )

        self.zones_in = initiate_polygon_zones(self.ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(self.ZONE_OUT_POLYGONS, [sv.Position.CENTER])

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    @staticmethod
    def denormalize_polygons(
        normalized_polygons: List[np.ndarray],
        frame_size: Tuple[int, int],
    ) -> List[np.ndarray]:
        """Convert normalized polygons to pixel coordinates"""
        width, height = frame_size
        denormalized = []
        for polygon in normalized_polygons:
            denormalized_poly = polygon.copy()
            denormalized_poly[:, 0] *= width  # x coordinates
            denormalized_poly[:, 1] *= height  # y coordinates
            denormalized.append(denormalized_poly.astype(int))
        return denormalized

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.infer(
            frame, confidence=self.conf_threshold, iou_threshold=self.iou_threshold
        )[0]
        detections = sv.Detections.from_inference(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with Inference and ByteTrack"
    )

    parser.add_argument(
        "--model_id",
        default="vehicle-count-in-drone-video/6",
        help="Roboflow model ID",
        type=str,
    )
    parser.add_argument(
        "--roboflow_api_key",
        default=None,
        help="Roboflow API KEY",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    api_key = args.roboflow_api_key
    api_key = os.environ.get("ROBOFLOW_API_KEY", api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API KEY is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    args.roboflow_api_key = api_key

    processor = VideoProcessor(
        roboflow_api_key=args.roboflow_api_key,
        model_id=args.model_id,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()