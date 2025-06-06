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
from func.hailo_rpi_common import (
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)


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
        self.resolution_wh = (0,0)  # Adjust if needed
        # Assume a fixed FPS if not known. If your pipeline's FPS differs, adjust accordingly.
        self.fps = 0  # this should be dynamic and should be extracted from the video itself
        self.confidence_threshold = 0.4
        self.iou_threshold = 0.7
        self.binary_mask = None 
        self.traffic_changed = True # To store the binary mask based on the ODs
        self.SOURCE = None # To store the polygon points
        self.TARGET = None # To store the target points
        self.lane_data = None # To store the lane data

    def new_function(self):  # Just to keep original example
        return "The meaning of life is: "

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
    xyxy = [] # [[xmin, ymin, xmax, ymax], ...] distances are in pixels, diagonal distance
    confidences = []
    class_ids = []
    class_labels = []

    for d in hailo_detections:
        #print(f"hailo detections:{d.get_label()}") # debugging print
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
                inside_mask = user_data.polygon_zone.trigger(detections)
                detections = detections[inside_mask]
                class_labels = [class_labels[i] for i in np.where(inside_mask)[0]]
            except:
                pass

    # Run NMS
    detections = detections.with_nms(threshold=user_data.iou_threshold)

    # Track detections with ByteTrack
    detections = user_data.byte_track.update_with_detections(detections=detections)

    # Speed estimation
    # Transform points (anchor = bottom_center)
    if len(detections) > 0:
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = user_data.view_transformer.transform_points(points=points).astype(int)

        for tracker_id, [_, y] in zip(detections.tracker_id, points):
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
                distance = abs(coordinate_start - coordinate_end)
                time = len(user_data.coordinates[tracker_id]) / user_data.fps
                speed = distance / time * 3.6
                labels.append(f"{current_class_label}, {int(distance)} km, {int(speed)} km/h")
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
        
        cv2.polylines(
                annotated_frame,
                [polygon_points], # polylines need a list of array of points
                isClosed=True,
                color=(0,0,255), # red color
                thickness=1
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

        self.app_callback = app_callback
        #self.app_callback_od = od_callback
        #self.app_callback_ld = ld_callback

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
        
        # Polygon zone and view transformer
        user_data.polygon_zone = sv.PolygonZone(polygon=SOURCE)
        user_data.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

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
    
