import numpy as np
import supervision as sv

from Lane_Cogestion_Monitoring.func.hailo_functional import (
    user_app_callback_class,
    ViewTransformer,
    GStreamerDetectionApp,
)
from func.hailo_rpi_common import get_default_parser
from Lane_Cogestion_Monitoring.utils.hailo_utils import (
    define_source_polygon, 
    get_video_info
)


# Define the polygon in source perspective
#SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
#TARGET_HEIGHT = 80 #250
#SOURCE = define_source_polygon(left_lane, right_lane, frame_height, TARGET_HEIGHT)
#SOURCE = SOURCE.astype(float)
#TARGET_WIDTH = 22 #25

#TARGET = np.array(
#    [
#        [0, 0],
#        [TARGET_WIDTH - 1, 0],
#        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
#        [0, TARGET_HEIGHT - 1],
#    ]
#)


# the main function to be shifted to the main.py file later 
if __name__ == "__main__":
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

    parser.add_argument(
        "--hef_path_ld",
        default="ufld_v2.hef",
        help="Path to HEF file for lane detection",
    )

    parser.add_argument(
        "--hef_path_od",
        default="yolov6n.hef",
        help="Path to HEF file for object detection",
    )

    args = parser.parse_args()

    try:
        original_frame_width, original_frame_height, total_frames = get_video_info(args.input_video)
    except ValueError as e:
        print(e)
        exit(1)
    
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

    user_data = user_app_callback_class()
    user_data.SOURCE = SOURCE
    user_data.TARGET = TARGET

    user_data.polygon_zone = sv.PolygonZone(polygon=SOURCE)
    user_data.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    app = GStreamerDetectionApp(args, user_data)
    app.run()