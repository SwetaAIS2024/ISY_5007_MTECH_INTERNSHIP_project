# detection.py

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo

# Add these imports for Supervision and ByteTrack
import supervision as sv
from collections import defaultdict, deque
# from utils import HailoAsyncInference
from utils.hailo_rpi_common import (
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,

)

from utils.zone_annotation import annotate_zones


# ---------------------- ADDITIONAL CODE FOR SPEED ESTIMATION ------------------------------------

TARGET_HEIGHT = 80 #250
TARGET_WIDTH = 22 #25
SOURCE = np.array([[1252, 1200], [2298, 1200], [5039, 2159], [-550, 2159]]) 
SOURCE = SOURCE.astype(float)

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
    def __init__(self, args):
        super().__init__()
        # Additional fields for speed estimation
        self.coordinates = None
        self.byte_track = None
        self.polygon_zone = None
        self.view_transformer = None
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None
        self.resolution_wh = (0,0)  # Adjust if needed
        # Assume a fixed FPS if not known. If your pipeline's FPS differs, adjust accordingly.
        self.fps = 0  # this should be dynamic and should be extracted from the video itself
        self.confidence_threshold = 0.4
        self.iou_threshold = 0.7
        self.binary_mask = None 
        self.traffic_changed = True # To store the binary mask based on the ODs
        self.SOURCE = None # To store the polygon points
        self.TARGET = None # To store the target points
        self.current_frame = None

        video_source = args.input
        self.all_lane_polygons = self._initialize_lanes(video_source) # To store the lane data

    def new_function(self):  # Just to keep original example
        return "The meaning of life is: "
    
    def extract_frame(self, buffer, pad): 
        format, width, height = get_caps_from_pad(pad)
        if not self.use_frame or format is None or width is None or height is None:
            print(f"Invalid frame parameters: use_frame={self.use_frame}, format={format}, width={width}, height={height}")
            return None
        
        frame = get_numpy_from_buffer(buffer, format, width, height)

        if frame is None:
            print("Failed to extract frame from buffer.")
            return None

        if format == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        print(f"Extracted frame with shape: {frame.shape}")
        return frame
    
    def _initialize_lanes(self, video_source: str = None) -> dict[str, np.ndarray]:
        """ Initialize thelane polygons either from the annoatations or from the defaults. """
        if video_source:
            return self._get_annotated_lanes(video_source)
        else:
            return self._get_default_lanes()
    
    def _get_annotated_lanes(self, video_source: str) -> dict[str, np.ndarray]:
        frame = self._get_first_frame(video_source)
        if frame is not None and frame.size > 0:
            print("Getting the super lane zone for each lane...")
            super_lane_zone_polygons = annotate_zones(frame)
            print("Getting the queue zone for each lane...")
            queue_lane_zone_polygons = annotate_zones(frame)

            lane_polygons = {}
            for i, (super_lane_zone_polygons, queue_lane_zone_polygons) in enumerate(zip(super_lane_zone_polygons, queue_lane_zone_polygons)):
                lane_polygons[f"lane_{i + 1}"] = {
                    "super_lane_polygon": np.array(super_lane_zone_polygons),
                    "queue_zone_polygon": np.array(queue_lane_zone_polygons),
                }
            return lane_polygons
        else:
            print("Failed to get the first frame for lane annotation.")
            return {}
    
    def _get_first_frame(self, source:str) -> np.ndarray:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error in opening the video source")
            return None
        else:
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
    
    def set_frame(self, frame):
        self.current_frame = frame

    def get_frame(self):
        return self.current_frame
        
# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# od_callback
# ld_callback
# These two for the parallel pipeline
 
def od_callback(args, pad, info, user_data: user_app_callback_class):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    # placeholder
    return Gst.PadProbeReturn.OK

def ld_callback(args, pad, info, user_data: user_app_callback_class):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    # placeholder
    return Gst.PadProbeReturn.OK

# this one for the sequential pipeline

def app_callback(pad, info, user_data: user_app_callback_class):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
        
    # Increment frame count
    user_data.increment()

    # Get caps to retrieve frame size and format
    #format, width, height = get_caps_from_pad(pad)

    # Extract the frame using the helper method
    frame = user_data.extract_frame(buffer, pad)
    if frame is None:
        print("Extracted frame is None.")
        return Gst.PadProbeReturn.OK
    
    user_data.set_frame(frame)
    print("Frame sucessfully stored in user_data")

    # Extract detections from hailo ROI
    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Convert hailo detections to supervision.Detections
    xyxy = []  # [[xmin, ymin, xmax, ymax], ...]
    confidences = []
    class_ids = []
    class_labels = []

    for d in hailo_detections:
        label = d.get_label()
        bbox = d.get_bbox()  # (x, y, w, h)
        confidence = d.get_confidence()

        xmin = bbox.xmin() * frame.shape[1]
        ymin = bbox.ymin() * frame.shape[0]
        xmax = bbox.xmax() * frame.shape[1]
        ymax = bbox.ymax() * frame.shape[0]

        xyxy.append([xmin, ymin, xmax, ymax])
        confidences.append(confidence)
        class_ids.append(0)  # Default class ID
        class_labels.append(label)

    if len(xyxy) > 0:
        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )
    else:
        detections = sv.Detections.empty()

    # Apply confidence threshold
    mask = detections.confidence > user_data.confidence_threshold
    detections = detections[mask]
    class_labels = [class_labels[i] for i in np.where(mask)[0]]

    # Process each lane independently
    for lane_name, polygon in user_data.all_lane_polygons.items():
        super_lane_polygon = polygon["super_lane_polygon"]
        queue_zone_polygon = polygon["queue_zone_polygon"]
        
        # Denormalize super lane polygon points to pixel coordinates
        pixel_super_lane_polygon = (super_lane_polygon * [frame.shape[1], frame.shape[0]]).astype(int)

        # Create a mask for the super lane
        super_lane_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(super_lane_mask, [pixel_super_lane_polygon], 255)

        # Filter detections for the super lane
        if len(detections) > 0:
            try:
                super_lane_zone = sv.PolygonZone(polygon=pixel_super_lane_polygon)
                inside_mask = super_lane_zone.trigger(detections)
                super_lane_detections = detections[inside_mask]
                super_lane_class_labels = [class_labels[i] for i in np.where(inside_mask)[0]]
            except Exception as e:
                print(f"Error filtering detections for super lane {lane_name}: {e}")
                super_lane_detections = sv.Detections.empty()
                super_lane_class_labels = []
        else:
            super_lane_detections = sv.Detections.empty()
            super_lane_class_labels = []

        # Run NMS for the super lane
        super_lane_detections = super_lane_detections.with_nms(threshold=user_data.iou_threshold)

        # Track detections with ByteTrack for the super lane
        super_lane_detections = user_data.byte_tracks[lane_name].update_with_detections(detections=super_lane_detections)

        # Annotate the super lane on the main frame
        cv2.polylines(
            frame,
            [pixel_super_lane_polygon],
            isClosed=True,
            color=(0, 255, 0),  # Green color for super lane
            thickness=2
        )

        # Process the queue zone for the current lane
        pixel_queue_zone_polygon = (queue_zone_polygon * [frame.shape[1], frame.shape[0]]).astype(int)

        # Create a mask for the queue zone
        queue_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(queue_zone_mask, [pixel_queue_zone_polygon], 255)

        # Filter detections for the queue zone
        if len(detections) > 0:
            try:
                queue_zone = sv.PolygonZone(polygon=pixel_queue_zone_polygon)
                queue_inside_mask = queue_zone.trigger(detections)
                queue_detections = detections[queue_inside_mask]
                queue_class_labels = [class_labels[i] for i in np.where(queue_inside_mask)[0]]
            except Exception as e:
                print(f"Error filtering detections for queue zone in lane {lane_name}: {e}")
                queue_detections = sv.Detections.empty()
                queue_class_labels = []
        else:
            queue_detections = sv.Detections.empty()
            queue_class_labels = []

        # Process queue zone detections (e.g., counting vehicles)
        if len(queue_detections) > 0:
            print(f"Queue Zone in {lane_name}: {len(queue_detections)} vehicles detected.")

        # --- Queue length calculation logic ---
        QUEUE_LENGTH_THRESHOLD = 3  # Minimum vehicles to trigger queue length calculation
        if len(queue_detections) >= QUEUE_LENGTH_THRESHOLD:
            # Calculate queue length using bottom-center points of bounding boxes
            queue_points = queue_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            # Optionally, transform to real-world coordinates if needed
            # queue_points = user_data.view_transformer.transform_points(points=queue_points)
            if len(queue_points) > 0:
                # Sort by y (vertical) position (assuming y increases downwards)
                sorted_points = sorted(queue_points, key=lambda pt: pt[1])
                # Queue length: distance from first to last vehicle in the queue (vertical axis)
                queue_length_pixels = abs(sorted_points[-1][1] - sorted_points[0][1])
                print(f"[Queue Length] Lane {lane_name}: {queue_length_pixels:.1f} pixels (vehicles: {len(queue_detections)})")
                # --- Draw queue length on frame ---
                # Use centroid of queue zone polygon as text position
                centroid = np.mean(pixel_queue_zone_polygon, axis=0).astype(int)
                text = f"Queue: {queue_length_pixels:.0f}px ({len(queue_detections)} veh)"
                cv2.putText(
                    frame,
                    text,
                    tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,  # font scale
                    (0, 0, 255),  # red color
                    2,  # thickness
                    cv2.LINE_AA
                )

    user_data.set_frame(frame)
    print("Frame sucessfully stored in user_data")

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# GStreamerDetectionApp class
# -----------------------------------------------------------------------------------------------
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)

        self.app_callback = app_callback
        #self.app_callback_od = od_callback
        #self.app_callback_ld = ld_callback
        self.ld_hef_path = args.hef_path_ld
        self.od_hef_path = args.hef_path_od

        
        # instead of hardcoding the values, we can get the values from the model itself 
        # so that we can use the same code for multiple models
        # self.network_height, self.network_width, _ =  hailo_inference.get_input_shape()

        # Initialize parameters for your model and postprocessing if needed
        self.od_batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.od_network_width = 640
        self.od_network_height = 640
        self.od_network_format = "RGB"
        od_nms_score_threshold = 0.3
        od_nms_iou_threshold = 0.45
        
        self.ld_batch_size = 2
        self.ld_nms_score_threshold = 0.3 
        self.ld_network_width = 640
        self.ld_network_height = 288
        self.ld_network_format = "RGB"
        ld_nms_score_threshold = 0.3
        ld_nms_iou_threshold = 0.45
        
        self.all_lane_polygons = user_data.all_lane_polygons
        frame = self.app_callback.get_frame()
        if frame is None:
            print("Frame is None. Skipping...")
            return
        
        frame_height, frame_width = frame.shape[:2]
        
        '''
        original_width, original_height = 3840, 2160
        new_width, new_height = 640, 640 # now input size directly from the model
        scaling_x = new_width / original_width
        scaling_y = new_height / original_height

        SOURCE = SOURCE * [scaling_x, scaling_y]
        '''
        
        original_width, original_height = 3840, 2160 # this part also can be extracted from 
        # the input video itself 
        new_width, new_height = self.network_width_od, self.network_height_od
        aspect_ratio = original_width / original_height  # ~1.777... for 16:9

        # Compute the scaled height that maintains the aspect ratio at new_width
        scaled_height = int(new_width / aspect_ratio)  # ~360 for 16:9 content in 640 width
        scaling_x = new_width / original_width   # 640/3840 = 1/6
        scaling_y = scaled_height / original_height  # 360/2160 = 1/6

        # Calculate the top black bar offset
        top_black_bar = (new_height - scaled_height) / 2  # (640 - 360)/2 = 140 pixels

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
            self.default_postprocess_so = os.path.join(self.postprocess_dir_od, 'libyolo_hailortpp_post.so')

        

        self.od_thresholds_str = (
            f"nms-score-threshold={od_nms_score_threshold} "
            f"nms-iou-threshold={od_nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        self.ld_thresholds_str = (
            f"nms-score-threshold={ld_nms_score_threshold} "
            f"nms-iou-threshold={ld_nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )
        
        setproctitle.setproctitle("Hailo Detection App")

        # --------------------- SUPER VISION SETUP ---------------------
        # Initialize speed estimation and annotation objects after pipeline creation
        # We assume 30 fps, but if known, adjust user_data.fps accordingly.
        user_data.fps = 25
        user_data.confidence_threshold = 0.4
        user_data.iou_threshold = 0.7
        
        # Initialize lane-specific data
        user_data.all_lane_polygons = self.all_lane_polygons  # Assume `self.lane_polygons` contains normalized polygons for all lanes

        # Initialize ByteTrack for each lane
        user_data.byte_tracks = {
            lane_name: sv.ByteTrack(frame_rate=user_data.fps, track_activation_threshold=user_data.confidence_threshold)
            for lane_name in user_data.all_lane_polygons
        }

        # Initialize annotators for each lane
        user_data.annotators = {}
        for lane_name in user_data.all_lane_polygons:
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=(self.network_width, self.network_height))
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=(self.network_width, self.network_height))
            
            user_data.annotators[lane_name] = {
                "box_annotator": sv.BoxAnnotator(
                    thickness=1,
                    color=sv.ColorPalette(colors=[sv.Color(r=0, g=255, b=0)])
                ),
                "label_annotator": sv.LabelAnnotator(
                    text_scale=text_scale,
                    text_thickness=thickness,
                    text_position=sv.Position.BOTTOM_CENTER,
                ),
                "trace_annotator": sv.TraceAnnotator(
                    thickness=thickness,
                    color=sv.ColorPalette(colors=[sv.Color(r=0, g=255, b=0)]),
                    trace_length=user_data.fps * 1.5,
                    position=sv.Position.BOTTOM_CENTER,
                ),
            }

        # Initialize speed tracking data for each lane
        user_data.coordinates = {lane_name: defaultdict(list) for lane_name in user_data.all_lane_polygons}

        # Process each lane independently
        for lane_name, polygons in user_data.all_lane_polygons.items():
            
            super_lane_polygon = polygons["super_lane_polygon"]
            # Denormalize polygon points to pixel coordinates
            pixel_polygon = (super_lane_polygon * [self.network_width, self.network_height]).astype(int)

            # Create a mask for the lane
            #mask = np.zeros((self.network_height, self.network_width), dtype=np.uint8)
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(mask, [pixel_polygon], 255)

            # Extract lane-specific region
            lane_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Process lane-specific detections
            detections = user_data.byte_tracks[lane_name].update(lane_frame)

            # Speed tracking for each detection
            lane_labels = []
            if len(detections) > 0:
                # Get bottom-center points of bounding boxes
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points = user_data.view_transformer.transform_points(points=points).astype(int)

                # Update coordinates for speed tracking
                for tracker_id, [_, y] in zip(detections.tracker_id, points):
                    user_data.coordinates[lane_name][tracker_id].append(y)

                # Calculate speed for each tracked object
                for i, tracker_id in enumerate(detections.tracker_id):
                    current_class_label = detections.class_id[i]
                    if len(user_data.coordinates[lane_name][tracker_id]) < user_data.fps / 2:
                        lane_labels.append(f"{current_class_label}")
                    else:
                        coordinate_start = user_data.coordinates[lane_name][tracker_id][-1]
                        coordinate_end = user_data.coordinates[lane_name][tracker_id][0]
                        distance = abs(coordinate_start - coordinate_end)
                        time = len(user_data.coordinates[lane_name][tracker_id]) / user_data.fps
                        speed = distance / time * 3.6  # Convert to km/h
                        lane_labels.append(f"{current_class_label}, {int(speed)} km/h")

            # Annotate the lane-specific frame
            lane_frame = user_data.annotators[lane_name]["box_annotator"].annotate(scene=lane_frame, detections=detections)
            lane_frame = user_data.annotators[lane_name]["label_annotator"].annotate(scene=lane_frame, detections=detections, labels=lane_labels)
            lane_frame = user_data.annotators[lane_name]["trace_annotator"].annotate(scene=lane_frame, detections=detections)

            # Display the lane-specific frame (optional)
            cv2.imshow(f"{lane_name} - Lane View", lane_frame)
        
        
        self.create_pipeline()

    def get_pipeline_string(self):
        if self.options_menu.pipeline_type == "parallel":
            return self.get_parallel_pipeline_string()
        elif self.options_menu.pipeline_type == "sequential":
            return self.get_sequential_pipeline_string()
        else:
            raise ValueError(f"Invalid pipeline type. {self.options_menu.pipeline_type} Choose 'parallel' or 'sequential'.")

    def get_parallel_pipeline_string(self):
        source_element = "hailomuxer name=hmux "
        if self.source_type == "rpi":
            source_element += (
                "libcamerasrc name=src_0 auto-focus-mode=2 ! "
                f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
                + QUEUE("queue_src_scale")
                + "videoscale ! "
                f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "
            )
        elif self.source_type == "usb":
            source_element += (
                f"v4l2src device={self.video_source} name=src_0 ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
            )
        else:
            source_element += (
                f"filesrc location={self.video_source} name=src_0 ! "
                + QUEUE("queue_dec264")
                + " qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
                + " video/x-raw, format=I420 ! "
            )

        source_element += "tee name=t "

        pipeline_string_OD = ( 
              f"t. !"
            + QUEUE("queue_od", max_size_buffers=3, leaky="no")
            + QUEUE("queue_scale_od", max_size_buffers=3, leaky="no")
            + "videoscale n-threads=2 ! "
            + QUEUE("queue_src_convert_od", max_size_buffers=3, leaky="no")
            + f"videoconvert n-threads=3 name=src_convert_od qos=false ! video/x-raw, format={self.network_format_od}, width={self.network_width_od}, height={self.network_height_od}, pixel-aspect-ratio=1/1 ! "
            + QUEUE("queue_hailonet_od", max_size_buffers=3, leaky="no")
            + f"hailonet hef-path={self.od_hef_path} batch-size={self.od_batch_size} {self.od_thresholds_str} force-writable=true vdevice-group-id=1 scheduling-algorithm=HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN ! "
            + QUEUE("queue_hailofilter_od", max_size_buffers=3, leaky="no")
            + f"hailofilter so-path={self.default_postprocess_so_od} {self.labels_config} qos=false ! "
            + f"hmux.sink_0 "
        )
        
        pipeline_string_LD = (
              f"t. !"
            + QUEUE("queue_ld", max_size_buffers=3, leaky="no")
            + QUEUE("queue_scale_ld", max_size_buffers=3, leaky="no")
            + "videoscale n-threads=2 ! " 
            + QUEUE("queue_src_convert_ld", max_size_buffers=3, leaky="no")
            + f"videoconvert n-threads=3 name=src_convert_ld qos=false ! video/x-raw, format={self.network_format_ld}, width={self.network_width_ld}, height={self.network_height_ld}, pixel-aspect-ratio=1/1 ! "
            + QUEUE("queue_hailonet_ld", max_size_buffers=3, leaky="no")
            + f"hailonet hef-path={self.ld_hef_path} batch-size={self.ld_batch_size} force-writable=true vdevice-group-id=1 scheduling-algorithm=HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN ! "
            + QUEUE("queue_hailofilter_ld", max_size_buffers=3,leaky="no")
            #+ f"hailopython module = {self.default_postprocess_so_ld} function=post_process_lane_detections qos=false ! "
            + f"hmux.sink_1 "
        )    

        source_element += pipeline_string_OD
        source_element += pipeline_string_LD

        # BRANCH 1 + BRANCH 2  
        source_element += "hmux. ! "

        # now we need to send the output of the muxer to the hailooverlay
        pipeline_string_parallel = (
            source_element
            + QUEUE("queue_hailo_python", leaky="no")
            + QUEUE("queue_user_callback", leaky="no")
            + "identity name=identity_callback ! " # the callback fundtion, i think this needs to be common, this is like the processing part after the inference at frmae is done.
            + QUEUE("queue_hailooverlay", leaky="no")
            + "hailooverlay ! "
            + QUEUE("queue_videoconvert", leaky="no")
            + "videoconvert n-threads=3 qos=false ! "
            + QUEUE("queue_hailo_display", leaky="no")
            + f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
            #+ "fakesink"
        )
        print(pipeline_string_parallel)
        
        return pipeline_string_parallel
    
    def get_sequential_pipeline_string(self):
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

        pipeline_string_scale_vconv_1 = (
            QUEUE("queue_scale_1")
            + "videoscale n-threads=2 name=vscale_1 ! "
            + QUEUE("queue_src_convert_1")
            + f"videoconvert n-threads=3 name=src_convert_1 qos=false ! video/x-raw, format={self.network_format_od}, width={self.network_width_od}, height={self.network_height_od}, pixel-aspect-ratio=1/1 ! "
        )

        pipeline_string_scale_vconv_2 = (
            QUEUE("queue_scale_2")
            + "videoscale n-threads=2 name=vscale_2 ! "
            + QUEUE("queue_src_convert_2")
            + f"videoconvert n-threads=3 name=src_convert_2 qos=false ! video/x-raw, format={self.network_format_od}, width={self.network_width_od}, height={self.network_height_od}, pixel-aspect-ratio=1/1 ! "
        )

        pipeline_string_OD = ( 
            QUEUE("queue_od", max_size_buffers=3, leaky="no")
            + QUEUE("queue_scale_od", max_size_buffers=3, leaky="no")
            + "videoscale n-threads=2 ! "
            + QUEUE("queue_src_convert_od", max_size_buffers=3, leaky="no")
            + f"videoconvert n-threads=3 name=src_convert_od qos=false ! video/x-raw, format={self.network_format_od}, width={self.network_width_od}, height={self.network_height_od}, pixel-aspect-ratio=1/1 ! "
            + QUEUE("queue_hailonet_od", max_size_buffers=3, leaky="no")
            + f"hailonet hef-path={self.od_hef_path} batch-size={self.od_batch_size} {self.od_thresholds_str} force-writable=true vdevice-group-id=1 scheduling-algorithm=HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN ! "
            + QUEUE("queue_hailofilter_od", max_size_buffers=3, leaky="no")
            + f"hailofilter so-path={self.default_postprocess_so_od} {self.labels_config} qos=false ! "
        )
        
        pipeline_string_LD = (
            QUEUE("queue_ld", max_size_buffers=3, leaky="no")
            + QUEUE("queue_scale_ld", max_size_buffers=3, leaky="no")
            + "videoscale n-threads=2 ! " 
            + QUEUE("queue_src_convert_ld", max_size_buffers=3, leaky="no")
            + f"videoconvert n-threads=3 name=src_convert_ld qos=false ! video/x-raw, format={self.network_format_ld}, width={self.network_width_ld}, height={self.network_height_ld}, pixel-aspect-ratio=1/1 ! "
            #+ QUEUE("queue_hailonet_ld", max_size_buffers=3, leaky="no")
            #+ f"hailonet hef-path={self.ld_hef_path} batch-size={self.ld_batch_size} force-writable=true vdevice-group-id=1 scheduling-algorithm=HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN ! "
            #+ QUEUE("queue_hailofilter_ld", max_size_buffers=3,leaky="no")
            #+ f"hailopython module = {self.default_postprocess_so_ld} function=post_process_lane_detections qos=false ! "
        )   

        pipeline_string_after_aggr = (
            QUEUE("queue_user_callback", leaky="no")
            + "identity name=identity_callback ! " # the callback fundtion, i think this needs to be common, this is like the processing part after the inference at frmae is done.
            + QUEUE("queue_hailooverlay", leaky="no")
            + "hailooverlay ! "
            + QUEUE("queue_videoconvert", leaky="no")
            + "videoconvert n-threads=3 qos=false ! "
            + QUEUE("queue_hailo_display", leaky="no")
            + f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
            #+ "fakesink"
        )

        pipeline_string_sequential = (
            source_element
            + pipeline_string_scale_vconv_1
            + "tee name=t "
            + "hailomuxer name=hmux "
            
            + "t. ! "
            + QUEUE("queue_bypass_OD", leaky="no")
            + "hmux.sink_0 "
            
            + "t. ! "
            + pipeline_string_OD
            + "hmux.sink_1 "
            + "hmux. ! "

            + "tee name=thm "
            + "hailomuxer name=hm "

            + "thm. ! "
            + QUEUE("queue_bypass_LD", leaky="no")
            + "hm.sink_0 "

            + "thm. ! "
            + pipeline_string_LD
            + "hm.sink_1 "
            + "hm. ! "

            + pipeline_string_scale_vconv_2
            + pipeline_string_after_aggr
        )
        

        
        print(pipeline_string_sequential)
        return pipeline_string_sequential

