# why alignment is needed?
# The coordinate system of the lanes and the objects detected could be different.
# The resized frame or orginal frame of both the detection system could be different
# In other words (W_lane x H_lane) could be different from  (W_object x H_object) 
# this code should be in the utils.


import cv2


def alignment_coord(frame_size, target_frame_size):
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