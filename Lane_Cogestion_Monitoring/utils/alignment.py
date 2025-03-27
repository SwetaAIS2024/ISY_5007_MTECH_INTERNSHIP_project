# why alignment is needed?
# The coordinate system of the lanes and the objects detected could be different.
# The resized frame or orginal frame of both the detection system could be different
# In other words (W_lane x H_lane) could be different from  (W_object x H_object) 
# this code should be in the utils.


#Od pipeline input is reframed to 640 x 640 - model input shape for example here for yolo models input size is 640x640
# For the OD pipeline it is hardcoded to 640 x 640
# need to change this - done - now dynamic input shape update is done in hailo_inference.get_input_shape()

#Lane detection pipeline input is reframed to 1280 x 720 - 
# model input shape , here the algo direcltly gets the info from hailo_inference.get_input_shape()



import cv2


def post_processing_alignment_coord(frame_size, target_frame_size):
    """
    Alignment of the coordinate system of the lanes and the objects detected can be
    become different if the resolution of the input in both the pipelines are different

    
    Args:
    target_frame_size: Tuple[int, int]: Target resolution to resize the frame to.
    frame_size: Tuple[int, int]: Original frame size.
    aligned_resized_frame: np.ndarray - Frame to be resized. 
        
        TO BE DONE.
    
    Returns:
        aligned_resized_frame: np.ndarray - Frame to be resized. 

    """
    aligned_resized_frame = cv2.resize(frame_size, target_frame_size)


    return aligned_resized_frame