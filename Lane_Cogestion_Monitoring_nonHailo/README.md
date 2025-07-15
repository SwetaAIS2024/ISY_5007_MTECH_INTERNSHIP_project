# Lane_Congestion_Monitoring

## Multi Target Multi Tracking system

1. Manual annotation of zones
(Assume zone_annotation.py provides annotation tools for queue, entry, exit, and lane zones)

2. Object detection
(Assume detection is performed using a YOLO/other detector, e.g., in func/hailo_functional.py)

3. ID assignment
(Each detection is assigned a unique ID using a tracker, e.g., ByteTrack/DeepSORT)

4. Unique tracker for each OD in each lane for each camera
(Trackers are instantiated per lane per camera, maintaining separate tracklets)

5. Send the metadata to buffer and the aggregator
(Metadata for each detection/track is pushed to a buffer, then sent to an aggregator module)

6. Association and Matching at the aggregator
(Aggregator receives metadata from all lanes/cameras, performs association/matching using appearance features, timestamps, and spatial logic)

7. Final count update at the aggregator
(Aggregator updates ingoing/outgoing counts for each lane based on association results)

8. Overlay all the data on the frame
(Overlay vehicle IDs, counts, and zone info on the frame using OpenCV or Supervision)

9. Update the metadata for each lane and send to a dummy traffic controller
(After aggregation, send per-lane metadata to a dummy controller for simulation/testing)

The above steps are implemented in the following modules:

- func/hailo_functional.py: Core detection, tracking, zone logic, overlay
- utils/zone_annotation.py: Manual annotation tools
- main.py: Pipeline setup, aggregator, and controller communication

For simulation/testing, run main.py to execute the full pipeline.