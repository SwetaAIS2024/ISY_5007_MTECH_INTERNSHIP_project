import degirum as dg, degirum_tools

# choose inference host address
#inference_host_address = "@cloud"
# inference_host_address = "@local"
inference_host_address = "@local"


# choose zoo_url
#zoo_url = "degirum/models_hailort"
zoo_url = "home/rubesh/Desktop"
# zoo_url = "../models"

# set token
#token = degirum_tools.get_token()
token = '' # leave empty for local inference

od_model_name = "yolov5n"
ld_model_name = "ufld_v2"
video_source = "/home/rubesh/Desktop/hailo-rpi5-examples/resources/video3.mp4"

# Load face detection and gender detection models
face_det_model = dg.load_model(
    model_name=od_model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=degirum_tools.get_token(),
    overlay_color=[(255,255,0),(0,255,0)]    
)

gender_cls_model = dg.load_model(
    model_name=ld_model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=degirum_tools.get_token(),
)

# # Create a compound cropping model with 30% crop extent
# crop_model = degirum_tools.CroppingAndClassifyingCompoundModel(
#     face_det_model, 
#     gender_cls_model, 
#     30.0
# )

model_combined = degirum_tools.CompoundModel(
    od_model_name,
    ld_model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=degirum_tools.get_token(),
    overlay_color=[(255,255,0),(0,255,0)]
)


#  Run AI inference on video stream and display inference results
# Press 'x' or 'q' to stop
with degirum_tools.Display("Objects and Lanes") as display:
    for inference_result in degirum_tools.predict_stream(model_combined, video_source):
        display.show(inference_result)