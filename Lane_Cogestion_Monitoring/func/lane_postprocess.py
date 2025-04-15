# import hailo
from gsthailo import VideoFrame
# from gi.repository import Gst

# may be needs to be added in the LD callback, not in the gstsreamer.
def process_lane_detections(video_frame: VideoFrame, model_type):
    """
    Process lane detections to filter and format the data.

    Args:
        video_frame (VideoFrame): The video frame containing lane detections.
        model_type (str): Type of the lane detection model used.

    Returns:
        list: Processed lane detections.
    """
    roi = video_frame.roi
    tensrs = roi.get_tensors()
    print("Checking the LD post processing pipeline")

    match model_type:
        case "anchor":
            # anchor based model - UFLDv2 processing - detection results are in the format [(x, y), confidence scores, may be Lane IDs]

            for tensor in tensrs:
                print(tensor)
                print(tensor.name())         
            
        # case "segmentation":
        #     # segmentation based model - LaneNet, more accuracte SCNN or ENet-SAD processing
        #     for tensor in tensrs:
        #         print(tensor)
        #         print(tensor.name())
                    


        # case "keypoint":
        #     # Keypoint based model - FOLOLane processing
        #     for tensor in tensrs:
        #         print(tensor)
        #         print(tensor.name()) 


        # case "parameter":
        #     # Parameter based model - example - PolyLANENET processing
        #     for tensor in tensrs:
        #         print(tensor)
        #         print(tensor.name())

                