import degirum as dg
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

input_image_path = '/home/william-stengg/Desktop/sweta_LCM/Lane_Cogestion_Monitoring/test/degirum_lib/3_lane.png'
input_image = cv2.imread(input_image_path)
input_image_resized = cv2.resize(input_image, (640, 640))
input_image_uint8 = input_image_resized.astype(np.uint8)
# Run inference on an input image
print(f"size of the input image : {input_image_uint8.shape}")
input_tensor = np.expand_dims(input_image_uint8, axis=0)
print(f"Shape of the input tensor : {input_tensor.shape}")



inference_result = model(input_image_uint8)
print("Inference result:", inference_result.results)
print("Inference result type:", type(inference_result))
print("Inference result attributes:", dir(inference_result))
# Display the segmentation output
cv2.imwrite("Segmentation_Output.png", inference_result.image_overlay)
cv2.imshow("Segmentation Output", inference_result.image_overlay)


# Wait until the user presses 'x' or 'q' to close the window
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('x') or key == ord('q'):
        break

cv2.destroyAllWindows()