# detection.py

import gi
#gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from shapely.geometry import Polygon

# Add these imports for Supervision and ByteTrack
import supervision as sv
from collections import defaultdict, deque

from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)

# ---------------------- ADDITIONAL CODE FOR SPEED ESTIMATION ------------------------------------

# Define the polygon in source perspective
#SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
#SOURCE = SOURCE.astype(float)
# change for using the roboflow api
# using the normalised points

ORIGINAL_WIDTH = 3840
ORIGINAL_HEIGHT = 2160
#SOURCE_NORM = np.array([[0.01, 0.73], [0.74, 0.37], [0.94, 0.98], [0, 0.99]]) #working for the video2 
#SOURCE_NORM = np.array([[0.01, 0.9], [1, 0.97], [1, 0.46], [0.28, 0.06]]) #working for video3

#test with lane1 values for video3  
#left = np.array([[0.28, 0.19], [0.02, 0.88]])
#right = np.array([[0.43, 0.17], [0.58, 0.95]])

#test with lane2 values for video3
#left = np.array([[0.93, 0.42], [0.6, 0.97]])
#right = np.array([[0.98, 0.5], [0.85, 0.98]])
# original frame height 3840
# Target Height 100m away from the bottom line in the polygon
#SOURCE_NORM = np.array([left[0], right[0], right[1], left[1]]) # this logic works if we have the lane values - normalised                           

# three lanes for video3 
LANE1 = np.array([[0.27, 0.08], [0.35, 0.08], [0.35, 0.97], [0.01, 0.89]])
LANE2 = np.array([[0.38, 0.13], [0.45, 0.15], [0.59, 0.98], [0.39, 0.98]])
LANE3 = np.array([[0.6, 0.89], [0.9, 0.4], [0.99, 0.54], [0.85, 0.97]])

SOURCE_NORM_LANES = [ LANE1, LANE2, LANE3 ]


#SOURCE_NORM_LANES = [lane.reshape((-1,1,2)) for lane in SOURCE_NORM_LANES]

SOURCE_LANES = SOURCE_NORM_LANES * np.array([ORIGINAL_WIDTH, ORIGINAL_HEIGHT])
SOURCE = SOURCE_LANES.astype(float)
#SOURCE_LANES = list(SOURCE_LANES)

TARGET_WIDTH = 22 #25
TARGET_HEIGHT = 80 #250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Additional fields for speed estimation
        self.coordinates = None
        self.byte_track = None
        self.polygon_zone = None
        self.view_transformer = None
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None
        self.resolution_wh = (3840, 2160)  # Adjust if needed
        # Assume a fixed FPS if not known. If your pipeline's FPS differs, adjust accordingly.
        self.fps = 25  
        self.confidence_threshold = 0.4
        self.iou_threshold = 0.7

    def new_function(self):  # Just to keep original example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data: user_app_callback_class):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
        
    # Increment frame count
    user_data.increment()

    # Get caps to retrieve frame size and format
    format, width, height = get_caps_from_pad(pad)

    # Get video frame (if enabled)
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        # Convert to BGR for annotation (supervision expects BGR)
        if format == "RGB":
            # Already in RGB, just convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # For NV12/YUYV, you'd need proper color conversion to BGR.

    # Extract detections from hailo ROI
    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Convert hailo detections to supervision.Detections
    # hailo bbox format: let's assume it's xywh or xyxy as needed.
    # According to Hailo docs, bbox might be (x_min, y_min, width, height). We need xyxy:
    xyxy = []
    confidences = []
    class_ids = []
    class_labels = []

    for d in hailo_detections:
        label = d.get_label()
        bbox = d.get_bbox()  # (x, y, w, h)
        confidence = d.get_confidence()

        #x, y, w, h = bbox
        #xmin = x
        #ymin = y
        #xmax = x + w
        #ymax = y + h
        xmin = bbox.xmin() * width
        ymin = bbox.ymin() * height
        xmax = bbox.xmax() * width
        ymax = bbox.ymax() * height

        xyxy.append([xmin, ymin, xmax, ymax])
        confidences.append(confidence)
        # If needed, map label to class_id. If no classes known, you can set class_id=0 or map from a label dictionary
        class_ids.append(0)  # or a mapping if you have multiple classes
        class_labels.append(label)

    if len(xyxy) > 0:
        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )
    else:
        detections = sv.Detections.empty()

    mask = detections.confidence > user_data.confidence_threshold
    detections = detections[mask]
    class_labels = [class_labels[i] for i in np.where(mask)[0]]

    # Filter by polygon zone
    if user_data.polygon_zone is not None:
        if len(detections) > 0:
            try:
                inside_mask = [x.trigger(detections) for x in user_data.polygon_zone]
                detections = [detections[i] for i in inside_mask]
                class_labels = [class_labels[i] for i in np.where(inside_mask)[0]]
            except e:
                print("no class labels", e )
                pass

    # Run NMS
    detections_NMS = [i.with_nms(threshold=user_data.iou_threshold) for i in detections]
    print(detections_NMS)
    # Track detections with ByteTrack
    detections = [user_data.byte_track.update_with_detections(detections=dt) for dt in detections_NMS]

    # Speed estimation
    # Transform points (anchor = bottom_center)
    if len(detections) > 0:
        points = [dts.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) for dts in detections]
        #points = user_data.view_transformer.transform_points(points=points).astype(int)
        points = [[x.transform_points(points=pt).astype(int) for x in user_data.view_transformer] for pt in points]
        #print(get_list_depth(points))
        print(points)
        y_points = [arr[:,1] for arr in points]
        print(y_points)
        #for tracker_id, [_, y] in zip(detections.tracker_id, points):
        #    user_data.coordinates[tracker_id].append(y)
        for tracker_id, y in zip(detections.tracker_id, y_points):
            user_data.coordinates[tracker_id].append(y)

        # Prepare labels with speed info
        labels = []
        for i, tracker_id in enumerate(detections.tracker_id):
            current_class_label = class_labels[i]
            if len(user_data.coordinates[tracker_id]) < user_data.fps / 2:
                #labels.append(f"{tracker_id} {current_class_label}")
                labels.append(f"{current_class_label}")
            else:
                coordinate_start = user_data.coordinates[tracker_id][-1]
                coordinate_end = user_data.coordinates[tracker_id][0]
                l_start = len(coordinate_start)
                l_end = len(coordinate_end)
                max_len_coord = max(l_start, l_end)
                delta_pad_coord_start = int(max_len_coord - l_start)
                delta_pad_coord_end = int(max_len_coord - l_end)
                coordinate_start = np.pad(coordinate_start, (0, delta_pad_coord_start))
                coordinate_end = np.pad(coordinate_end, (0, delta_pad_coord_end))
                distance = abs(coordinate_start - coordinate_end)
                time = len(user_data.coordinates[tracker_id]) / user_data.fps
                speed = distance / time * 3.6
                mean_speed = np.mean(speed)
                print(speed.shape)
                print(speed)
                print(mean_speed)
                labels.append(f"{current_class_label} {int(mean_speed)} km/h")
    else:
        labels = []

    # Annotate frame
    if user_data.use_frame and frame is not None:
        # Trace, boxes, labels
        annotated_frame = frame.copy()
        annotated_frame = user_data.trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = user_data.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = user_data.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Show detection count or additional info if you wish
        cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        polygon_points = SOURCE.astype(int)
        #polygon_points = SOURCE

        cv2.polylines(
                annotated_frame,
                polygon_points, # polylines need a list of array of points
                isClosed=True,
                color=(0,0,255), # red color
                thickness=2
                )
        # Push annotated frame to user_data frame queue
        user_data.set_frame(annotated_frame)

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# GStreamerDetectionApp class
# -----------------------------------------------------------------------------------------------
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        
        # Initialize parameters for your model and postprocessing if needed
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        '''
        original_width, original_height = 3840, 2160
        new_width, new_height = 640, 640
        scaling_x = new_width / original_width
        scaling_y = new_height / original_height

        SOURCE = SOURCE * [scaling_x, scaling_y]
        '''
        original_width, original_height = 3840, 2160
        new_width, new_height = self.network_width, self.network_height
        aspect_ratio = original_width / original_height  # ~1.777... for 16:9

        # Compute the scaled height that maintains the aspect ratio at new_width
        scaled_height = int(new_width / aspect_ratio)  # ~360 for 16:9 content in 640 width
        scaling_x = new_width / original_width   # 640/3840 = 1/6
        scaling_y = scaled_height / original_height  # 360/2160 = 1/6

        # Calculate the top black bar offset
        top_black_bar = (new_height - scaled_height) / 2  # (640 - 360)/2 = 140 pixels

        #SOURCE = SOURCE.astype(float)
        # Adjust SOURCE coordinates:
        SOURCE[:, 0] *= scaling_x
        SOURCE[:, 1] = SOURCE[:, 1] * scaling_y + top_black_bar
        
        # Path to custom HEF if provided
        if args.hef_path is not None:
            self.hef_path = args.hef_path
        else:
            # Default networks
            if args.network == "yolov6n":
                self.hef_path = os.path.join(self.current_path, '../resources/yolov6n.hef')
            elif args.network == "yolov8s":
                self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
            elif args.network == "yolox_s_leaky":
                self.hef_path = os.path.join(self.current_path, '../resources/yolox_s_leaky_h8l_mz.hef')
            else:
                assert False, "Invalid network type"

        # Check for custom labels
        if args.labels_json is not None:
            self.labels_config = f' config-path={args.labels_json} '
            new_postprocess_path = os.path.join(self.current_path, '../resources/libyolo_hailortpp_post.so')
            if not os.path.exists(new_postprocess_path):
                print("Missing required custom postprocess so file.")
                exit(1)
            self.default_postprocess_so = new_postprocess_path
        else:
            self.labels_config = ''
            # Default postprocess
            self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolo_hailortpp_post.so')

        self.app_callback = app_callback
        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )
        
        setproctitle.setproctitle("Hailo Detection App")

        # --------------------- SUPER VISION SETUP ---------------------
        # Initialize speed estimation and annotation objects after pipeline creation
        # We assume 30 fps, but if known, adjust user_data.fps accordingly.
        user_data.fps = 25
        user_data.confidence_threshold = 0.4
        user_data.iou_threshold = 0.7

        # Polygon zone and view transformer
        
        #user_data.polygon_zone = sv.PolygonZone(polygon=SOURCE)
        user_data.polygon_zone = [sv.PolygonZone(polygon=poly) for poly in SOURCE]
        user_data.view_transformer = [ViewTransformer(source=poly, target=TARGET) for poly in SOURCE]


        # ByteTrack initialization
        user_data.byte_track = sv.ByteTrack(frame_rate=user_data.fps, track_activation_threshold=user_data.confidence_threshold)
        
        user_data.coordinates = defaultdict(lambda: deque(maxlen=int(user_data.fps))) # store past positions

        # Annotators
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=(self.network_width, self.network_height))
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(self.network_width, self.network_height))
        
        user_data.box_annotator = sv.BoundingBoxAnnotator(
                thickness=1,
                color=sv.ColorPalette(colors=[sv.Color(r=0,g=255,b=0)])
                )
        
        user_data.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        
        # Override default annotation label
         
        user_data.label_annotator = sv.LabelAnnotator(
                text_scale=0.4,
                text_thickness=1,
                text_padding=4,
                color=sv.ColorPalette(colors=[sv.Color(r=0,g=255,b=0)]),
                text_color=sv.Color.BLACK,
                text_position=sv.Position.BOTTOM_CENTER,
                )
        

        user_data.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            color=sv.ColorPalette(colors=[sv.Color(r=0,g=255,b=0)]),
            trace_length=user_data.fps * 1.5,
            position=sv.Position.BOTTOM_CENTER,
        )
        
        self.create_pipeline()

    def get_pipeline_string(self):
        if self.source_type == "rpi":
            source_element = (
                "libcamerasrc name=src_0 auto-focus-mode=2 ! "
                f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
                + QUEUE("queue_src_scale")
                + "videoscale ! "
                f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "
            )
        elif self.source_type == "usb":
            source_element = (
                f"v4l2src device={self.video_source} name=src_0 ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
            )
        else:
            source_element = (
                f"filesrc location={self.video_source} name=src_0 ! "
                + QUEUE("queue_dec264")
                + " qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
                " video/x-raw, format=I420 ! "
            )
        source_element += QUEUE("queue_scale")
        source_element += "videoscale n-threads=2 ! "
        source_element += QUEUE("queue_src_convert")
        source_element += f"videoconvert n-threads=3 name=src_convert qos=false ! video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "

        pipeline_string = (
            "hailomuxer name=hmux "
            + source_element
            + "tee name=t ! "
            + QUEUE("bypass_queue", max_size_buffers=20)
            + "hmux.sink_0 "
            + "t. ! "
            + QUEUE("queue_hailonet")
            + "videoconvert n-threads=3 ! "
            f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} {self.thresholds_str} force-writable=true ! "
            + QUEUE("queue_hailofilter")
            + f"hailofilter so-path={self.default_postprocess_so} {self.labels_config} qos=false ! "
            + QUEUE("queue_hmuc")
            + "hmux.sink_1 "
            + "hmux. ! "
            + QUEUE("queue_hailo_python")
            + QUEUE("queue_user_callback")
            + "identity name=identity_callback ! "
            + QUEUE("queue_hailooverlay")
            + "hailooverlay ! "
            + QUEUE("queue_videoconvert")
            + "videoconvert n-threads=3 qos=false ! "
            + QUEUE("queue_hailo_display")
            + f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
            #+ "fakesink"
        )
        print(pipeline_string)
        return pipeline_string


if __name__ == "__main__":
    user_data = user_app_callback_class()
    parser = get_default_parser()
    parser.add_argument(
        "--network",
        default="yolov6n",
        choices=['yolov6n', 'yolov8s', 'yolox_s_leaky'],
        help="Which Network to use, default is yolov6n",
    )
    parser.add_argument(
        "--hef-path",
        default=None,
        help="Path to HEF file",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="Path to custom labels JSON file",
    )
    args = parser.parse_args()
    app = GStreamerDetectionApp(args, user_data)
    #start = time.time() 
    app.run()
    #end = time.time()
    #total_time = end - start
    #print(f"TIME TAKEN: {total_time}")
