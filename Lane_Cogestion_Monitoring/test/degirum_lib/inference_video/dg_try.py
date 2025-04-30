import degirum as dg, degirum_tools
import cv2
import numpy as np

zoo_url_l = '/home/william-stengg/Desktop/sweta_LCM/Lane_Cogestion_Monitoring/test/degirum_lib/model_zoo' 
address = '@local'
# Verify the model zoo contents
zoo_contents = dg.list_models(address, zoo_url=zoo_url_l)
print("Available models in the model zoo:", zoo_contents)

# Check if the model is available in the zoo
model_name = 'yolov8n_seg'
if model_name not in zoo_contents:
    raise FileNotFoundError(f"Model '{model_name}' not found in the model zoo.")

# Load the segmentation model
model = dg.load_model(
    model_name=model_name,
    inference_host_address=address,
    zoo_url=zoo_url_l,
    token=""
)

video_source = "/home/william-stengg/Desktop/sweta_LCM/Lane_Cogestion_Monitoring/test/degirum_lib/inference_video/video1.mp4"
#ann_path = "/home/william-stengg/Desktop/sweta_LCM/Lane_Cogestion_Monitoring/test/degirum_lib/inference_video/ann_video1.mp4"
# # Annotate the video using the loaded model
# degirum_tools.annotate_video(model, video_source, ann_path)
# # display the annotated video
# degirum_tools.ipython_display(ann_path)


# run AI inference on video stream
inference_results = degirum_tools.predict_stream(model, video_source)

# display inference results
# Press 'x' or 'q' to stop
with degirum_tools.Display("AI Camera") as display:
    for inference_result in inference_results:
        display.show(inference_result)