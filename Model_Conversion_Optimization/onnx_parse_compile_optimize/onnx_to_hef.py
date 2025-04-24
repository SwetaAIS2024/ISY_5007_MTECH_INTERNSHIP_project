# the code in the hailo developer zone


# # General imports used throughout the tutorial
# import tensorflow as tf
# from IPython.display import SVG

# # import the ClientRunner class from the hailo_sdk_client package
# from hailo_sdk_client import ClientRunner

# chosen_hw_arch = "hailo8"
# # For Hailo-15 devices, use 'hailo15h'
# # For Mini PCIe modules or Hailo-8R devices, use 'hailo8r'
# onnx_model_name = "resnet_v1_18"
# onnx_path = "../models/resnet_v1_18.onnx"


# runner = ClientRunner(hw_arch=chosen_hw_arch)
# hn, npz = runner.translate_onnx_model(
#     onnx_path,
#     onnx_model_name,
#     start_node_names=["input.1"],
#     end_node_names=["192"],
#     net_input_shapes={"input.1": [1, 3, 224, 224]},
# )

# hailo_model_har_name = f"{onnx_model_name}_hailo_model.har"
# runner.save_har(hailo_model_har_name)

# !hailo visualizer {hailo_model_har_name} --no-browser
# SVG("resnet_v1_18.svg")


# # optimisation 
# # General imports used throughout the tutorial
# # file operations
# import json
# import os

# import numpy as np
# import tensorflow as tf
# from IPython.display import SVG
# from matplotlib import patches
# from matplotlib import pyplot as plt
# from PIL import Image
# from tensorflow.python.eager.context import eager_mode

# # import the hailo sdk client relevant classes
# from hailo_sdk_client import ClientRunner, InferenceContext

# %matplotlib inline

# IMAGES_TO_VISUALIZE = 5

# # First, we will prepare the calibration set. Resize the images to the correct size and crop them.
# def preproc(image, output_height=224, output_width=224, resize_side=256):
#     """imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px"""
#     with eager_mode():
#         h, w = image.shape[0], image.shape[1]
#         scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
#         resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h * scale), int(w * scale)])
#         cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)

#         return tf.squeeze(cropped_image)


# images_path = "../data"
# images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] == ".jpg"]

# calib_dataset = np.zeros((len(images_list), 224, 224, 3))
# for idx, img_name in enumerate(sorted(images_list)):
#     img = np.array(Image.open(os.path.join(images_path, img_name)))
#     img_preproc = preproc(img)
#     calib_dataset[idx, :, :, :] = img_preproc.numpy()

# np.save("calib_set.npy", calib_dataset)

# # Second, we will load our parsed HAR from the Parsing Tutorial

# model_name = "resnet_v1_18"
# hailo_model_har_name = f"{model_name}_hailo_model.har"
# assert os.path.isfile(hailo_model_har_name), "Please provide valid path for HAR file"
# runner = ClientRunner(har=hailo_model_har_name)
# # By default it uses the hw_arch that is saved on the HAR. For overriding, use the hw_arch flag.

# # Now we will create a model script, that tells the compiler to add a normalization on the beginning
# # of the model (that is why we didn't normalize the calibration set;
# # Otherwise we would have to normalize it before using it)

# # Batch size is 8 by default
# alls = "normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n"

# # Load the model script to ClientRunner so it will be considered on optimization
# runner.load_model_script(alls)

# # Call Optimize to perform the optimization process
# runner.optimize(calib_dataset)

# # Save the result state to a Quantized HAR file
# quantized_model_har_path = f"{model_name}_quantized_model.har"
# runner.save_har(quantized_model_har_path)



# # compilation 

# Note: This section demonstrates the Python APIs for Hailo Compiler. You could also use the CLI: try hailo compiler --help. More details on Dataflow Compiler User Guide / Building Models / Profiler and other command line tools.

# from hailo_sdk_client import ClientRunner

# Choose the quantized model Hailo Archive file to use throughout the example:

# model_name = "resnet_v1_18"
# quantized_model_har_path = f"{model_name}_quantized_model.har"

# Load the network to the ClientRunner:

# runner = ClientRunner(har=quantized_model_har_path)
# # By default it uses the hw_arch that is saved on the HAR. It is not recommended to change the hw_arch after Optimization.

# Run compilation (This method can take a couple of minutes):

# Note: The hailo compiler CLI tool can also be used.

# hef = runner.compile()

# file_name = f"{model_name}.hef"
# with open(file_name, "wb") as f:
#     f.write(hef)

