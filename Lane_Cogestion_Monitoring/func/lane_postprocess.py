

def process_lane_detections(lane_detections, ld_frame_width, ld_frame_height, model_type):
    """
    Process lane detections to filter and format the data.

    Args:
        lane_detections (list): List of lane detection results.
        ld_frame_width (int): Width of the frame.
        ld_frame_height (int): Height of the frame.
        model_type (str): Type of the lane detection model used.

    Returns:
        list: Processed lane detections.
    """
    processed_detections = []

    match model_type:
        case "UFLD":
            # UFLD processing - detecttion results are in the format [(x, y), confidence scores, may be Lane IDs]
            for detection in lane_detections:
                # Filter out detections with low confidence
                if detection['confidence'] < 0.5:
                    continue

                # Normalize coordinates to frame size
                detection['x1'] = int(detection['x1'] * ld_frame_width)
                detection['y1'] = int(detection['y1'] * ld_frame_height)
                detection['x2'] = int(detection['x2'] * ld_frame_width)
                detection['y2'] = int(detection['y2'] * ld_frame_height)

                processed_detections.append(detection)

        case "Hailo":
            # Hailo processing
            for detection in lane_detections:
                # Filter out detections with low confidence
                if detection['confidence'] < 0.5:
                    continue

                # Normalize coordinates to frame size
                detection['x1'] = int(detection['x1'] * ld_frame_width)
                detection['y1'] = int(detection['y1'] * ld_frame_height)
                detection['x2'] = int(detection['x2'] * ld_frame_width)
                detection['y2'] = int(detection['y2'] * ld_frame_height)

                processed_detections.append(detection)

    return processed_detections