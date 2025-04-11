import numpy as np
import supervision as sv

from func.hailo_functional import (
    user_app_callback_class,
    GStreamerDetectionApp,
)

from func.hailo_rpi_common import get_default_parser


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

    app = GStreamerDetectionApp(args, user_data)
    app.run()
