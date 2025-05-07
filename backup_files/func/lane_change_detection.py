class LaneChangeModel:
    """
    A class for detecting lane changes by comparing current lane detections
    with a stored "first correct instance" of lane detections.
    """

    def __init__(self):
        # Initialize the first correct instance of lane detections
        self.first_correct_instance = None

    def set_first_correct_instance(self, lane_detections):
        """
        Set the first correct instance of lane detections.
        Args:
            lane_detections (list): The initial correct lane detections.
        """
        self.first_correct_instance = lane_detections

    def detect(self, current_lane_detections):
        """
        Detect lane changes by comparing current lane detections with the first correct instance.
        Args:
            current_lane_detections (list): The current lane detections.
        Returns:
            bool: True if a lane change is detected, False otherwise.
        """
        if self.first_correct_instance is None:
            raise ValueError("First correct instance of lane detections is not set.")

        # Compare the current lane detections with the first correct instance
        lane_change_detected = self.compare_lanes(self.first_correct_instance, current_lane_detections)
        return lane_change_detected

    def compare_lanes(self, first_instance, current_instance):
        """
        Compare two sets of lane detections to detect lane changes.
        Args:
            first_instance (list): The first correct instance of lane detections.
            current_instance (list): The current lane detections. This will be periodically updated.
        The first instance is the reference for comparison.
        The current instance is the latest lane detection result.
        The comparison logic can be customized based on the specific requirements.
        Returns:
            bool: True if a lane change is detected, False otherwise.
        """
        # Example comparison logic: Check if the number of lanes or their positions differ
        if len(first_instance) != len(current_instance):
            return True  # Lane change detected due to a difference in the number of lanes

        for first_lane, current_lane in zip(first_instance, current_instance):
            # Compare lane positions (e.g., using a threshold for position differences)
            if not self.is_lane_similar(first_lane, current_lane):
                return True  # Lane change detected due to position differences

        return False  # No lane change detected

    def is_lane_similar(self, lane1, lane2, position_threshold=20):
        """
        Check if two lanes are similar based on their positions.
        Args:
            lane1 (dict): The first lane detection (e.g., {'points': [(x1, y1), ...]}).
            lane2 (dict): The second lane detection (e.g., {'points': [(x1, y1), ...]}).
            position_threshold (int): The maximum allowed difference in positions.
        Returns:
            bool: True if the lanes are similar, False otherwise.
        """
        points1 = lane1['points']
        points2 = lane2['points']

        if len(points1) != len(points2):
            return False  # Lanes are not similar if they have a different number of points

        for p1, p2 in zip(points1, points2):
            if abs(p1[0] - p2[0]) > position_threshold or abs(p1[1] - p2[1]) > position_threshold:
                return False  # Lanes are not similar if any point differs beyond the threshold

        return True  # Lanes are similar

def lane_change_detection(video_frames, lane_detections_per_frame):
    """
    Detect lane changes in the given video frames by comparing lane detections
    with the first correct instance.
    Args:
        video_frames (list): List of video frames.
        lane_detections_per_frame (list): List of lane detections for each frame.
    Returns:
        list: List of lane change detection results for each frame.
    """
    # Initialize the lane change detection model
    lane_change_model = LaneChangeModel()

    # Set the first correct instance of lane detections (e.g., from the first frame)
    if len(lane_detections_per_frame) > 0:
        lane_change_model.set_first_correct_instance(lane_detections_per_frame[0])

    # Initialize the list to store lane change detections
    lane_change_detections = []

    # Process each frame and its corresponding lane detections
    for frame, current_lane_detections in zip(video_frames, lane_detections_per_frame):
        # Detect lane changes in the current frame
        lane_change_detected = lane_change_model.detect(current_lane_detections)

        # Append the detection result to the list
        lane_change_detections.append(lane_change_detected)

    return lane_change_detections