# https://community.hailo.ai/t/running-multiple-models-independently/8122
# This script demonstrates how to run multiple models independently on different video sources (video files and webcams) using threading.
# It uses the Degirum SDK to load models and perform inference, displaying results in separate windows.

# https://community.hailo.ai/t/looking-for-more-info-to-work-with-2-streams-and-2-networks/4830/11 

#https://docs.edgeimpulse.com/experts/computer-vision-projects/brainchip-akida-multi-camera-inference



import threading
import degirum as dg
import degirum_tools

# choose inference host address
inference_host_address = "@cloud"
# inference_host_address = "@local"

# choose zoo_url
zoo_url = "degirum/models_hailort"
# zoo_url = "../models"

# set token
token = degirum_tools.get_token()
# token = '' # leave empty for local inference

# Define the configurations for video file and webcam
configurations = [
    {
        "model_name": "yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1",
        "source": "../assets/Traffic.mp4",  # Video file
        "display_name": "Traffic Camera"
    },
    {
        "model_name": "yolov8n_relu6_face--640x640_quant_hailort_hailo8_1",
        "source": 1,  # Webcam index
        "display_name": "Webcam Feed"
    }
]

# Function to run inference on a video stream (video file or webcam)
def run_inference(model_name, source, inference_host_address, zoo_url, token, display_name):
    # Load AI model
    model = dg.load_model(
        model_name=model_name,
        inference_host_address=inference_host_address,
        zoo_url=zoo_url,
        token=token
    )

    with degirum_tools.Display(display_name) as output_display:
        for inference_result in degirum_tools.predict_stream(model, source):
            output_display.show(inference_result)
    print(f"Stream '{display_name}' has finished.")

# Create and start threads
threads = []
for config in configurations:
    thread = threading.Thread(
        target=run_inference,
        args=(
            config["model_name"],
            config["source"],
            inference_host_address,
            zoo_url,
            token,
            config["display_name"]
        )
    )
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All streams have been processed.")