import numpy as np
import supervision as sv

from func.object_detection_tracking.detection_supervision_tappas328 import (
    user_app_callback_class,
    ViewTransformer,
    GStreamerDetectionApp,
)
from func.object_detection_tracking.hailo_rpi_common import get_default_parser
from func.object_detection_tracking.utils import (
    define_source_polygon, 
    get_video_info
)
from func.lane_detection.lane_detection import infer as lane_infer, UFLDProcessing



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
    args = parser.parse_args()

    # Lane detection 
    try:
        original_frame_width, original_frame_height, total_frames = get_video_info(args.input_video)
    except ValueError as e:
        print(e)
        exit(1)
    ufld_processing = UFLDProcessing(num_cell_row=100,
                                      num_cell_col=100,
                                      num_row=56,
                                      num_col=41,
                                      num_lanes=4,
                                      crop_ratio=0.8,
                                      original_frame_width=original_frame_width,
                                      original_frame_height=original_frame_height,
                                      total_frames=total_frames,
                                      )
    # Run the lane detection pipeline
    lane_data = lane_infer(
        video_path=args.input_video,
        net_path=args.net,
        batch_size=1,
        output_video_path=args.output_video,
        ufld_processing=ufld_processing,
    )

    # Dynamically compute the SOURCE polygon based on detected lanes
    left_lane = lane_data[0][0]  # Example: First frame, first lane
    right_lane = lane_data[0][-1]  # Example: First frame, last lane
    TARGET_HEIGHT = 80 #250
    TARGET_WIDTH = 22 #25
    SOURCE = define_source_polygon(left_lane, right_lane, original_frame_height, TARGET_HEIGHT)
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
    user_data.lane_data = lane_data
    user_data.polygon_zone = sv.PolygonZone(polygon=SOURCE)
    user_data.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    app = GStreamerDetectionApp(args, user_data)
    app.run()