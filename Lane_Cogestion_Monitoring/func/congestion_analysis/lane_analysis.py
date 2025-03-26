# Description: This file contains the functions to analyze the congestion 
# in the lanes.

def lane_to_object_mapping(lane_coodinates, object_coordinates):
    """
    Map objects detected in the OD pipeline to lanes.
    
    Args:
        lane_coodinates(Dict[int, List[Tuple[int,int]]]): Lane id map with Coordinates of the lanes detected.
        object_coordinates(Dict[int, List[Tuple[int,int]]]): Object id map with Coordinates of the objects detected.

        using "" some mapping strategy to map objects to lanes.
        TO BE DONE.
    
    Returns:
        Dict[int, List[int]]: Mapping of lane indices to object IDs.
    """
    
    lane_id_to_object_id = {}
    
    for lane_id, lane in enumerate(lane_coodinates):
        lane_id_to_object_id[lane_id] = []
        
        for object_id, object in enumerate(object_coordinates):
            #if object in lane: # Some strategy tp map the lanes.
                lane_id_to_object_id[lane_id].append(object_id)
    
    return lane_id_to_object_id



def congestion_level_per_lane(lane_id_to_object_id):
    """
    Count the number of unique objects detected in the OD pipeline.
    
    Args:
        lane_id_to_object_id(Dict[int, List[int]]): Mapping of lane indices to object IDs.
    
    Returns:
        Dict[int, float]: MApping of lane indices to congestion levels.
    """
    
    lane_id_congestion_level = {}
    
    for lane, objects in lane_id_to_object_id.items():
        lane_id_congestion_level[lane] = len(set(objects)) 

    return lane_id_congestion_level


