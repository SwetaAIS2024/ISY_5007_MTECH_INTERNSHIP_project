# Lane_Congestion_Monitoring

- Lane Detection with the UFLDv2 : Lane detection using the Ultra Fast Lane Detection v2 model
- Object Detection with yolov5m and speed estimation : Already done - used the hailo's example + Roboflow speed estimation , we will leverage the existing work done for the current project.
Roboflow example already incorporates the ByteTrack for tracking the detected objects.

- Source : https://blog.roboflow.com/estimate-speed-computer-vision/ 

Three options :

1. First vehicle OD -> Lane detection
2. First lane detection -> then vehicle OD
3. Parallelly do both 

Need to evaluate which is better for Real Time traffic analysis.

Model to choose from (model inference throughput):
1. yolov6n (single context) - 1251.78 FPS
2. yolov5m (single context) - 
3. 