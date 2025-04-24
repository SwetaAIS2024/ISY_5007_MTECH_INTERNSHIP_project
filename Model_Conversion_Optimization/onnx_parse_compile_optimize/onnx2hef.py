import os
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner
import tensorflow as tf


class HailoModelProcessor:
    def __init__(self, model_name, onnx_path, hw_arch="hailo8"):
        self.model_name = model_name
        self.onnx_path = onnx_path
        self.hw_arch = hw_arch
        self.runner = ClientRunner(hw_arch=self.hw_arch)
        self.hailo_model_har_name = f"{self.model_name}_hailo_model.har"
        self.quantized_model_har_path = f"{self.model_name}_quantized_model.har"
        self.hef_file_name = f"{self.model_name}.hef"

    def parse_onnx_model(self, start_node_names, end_node_names, net_input_shapes):
        """Parse the ONNX model and save it as a HAR file."""
        print("Parsing ONNX model...")
        hn, npz = self.runner.translate_onnx_model(
            self.onnx_path,
            self.model_name,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
            net_input_shapes=net_input_shapes,
        )
        self.runner.save_har(self.hailo_model_har_name)
        print(f"Model parsed and saved as HAR: {self.hailo_model_har_name}")

    def preprocess_images(self, images_path, output_height=224, output_width=224, resize_side=256):
        """Preprocess images for calibration."""
        print("Preprocessing images for calibration...")
        images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] == ".jpg"]
        calib_dataset = np.zeros((len(images_list), output_height, output_width, 3))

        for idx, img_name in enumerate(sorted(images_list)):
            img = np.array(Image.open(os.path.join(images_path, img_name)))
            img_preproc = self._preproc(img, output_height, output_width, resize_side)
            calib_dataset[idx, :, :, :] = img_preproc.numpy()

        np.save("calib_set.npy", calib_dataset)
        print("Calibration dataset saved as calib_set.npy")
        return calib_dataset

    @staticmethod
    def _preproc(image, output_height, output_width, resize_side):
        """Helper function for image preprocessing."""
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h * scale), int(w * scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        return tf.squeeze(cropped_image)

    def optimize_model(self, calib_dataset):
        """Optimize the model using the calibration dataset."""
        print("Optimizing the model...")
        assert os.path.isfile(self.hailo_model_har_name), "Please provide a valid path for the HAR file"
        self.runner = ClientRunner(har=self.hailo_model_har_name)

        # Add normalization to the model script
        normalization_script = "normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n"
        self.runner.load_model_script(normalization_script)

        # Optimize the model
        self.runner.optimize(calib_dataset)

        # Save the optimized model as a Quantized HAR file
        self.runner.save_har(self.quantized_model_har_path)
        print(f"Optimized model saved as Quantized HAR: {self.quantized_model_har_path}")

    def compile_model(self):
        """Compile the quantized model into a HEF file."""
        print("Compiling the model...")
        assert os.path.isfile(self.quantized_model_har_path), "Please provide a valid path for the Quantized HAR file"
        self.runner = ClientRunner(har=self.quantized_model_har_path)

        # Compile the model
        hef = self.runner.compile()

        # Save the compiled HEF file
        with open(self.hef_file_name, "wb") as f:
            f.write(hef)
        print(f"Compiled model saved as HEF: {self.hef_file_name}")


# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = HailoModelProcessor(
        model_name="yolov8n-seg",
        onnx_path="/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/onnx_parse_compile_optimize/yolov8n-seg_static_outputs_inferred_fixed_inferred.onnx",
        hw_arch="hailo8"
    )

    # Step 1: Parse the ONNX model
    # processor.parse_onnx_model(
    #     start_node_names=["/model.0/conv/Conv"],
    #     end_node_names=["/model.22/Concat_29"],
    #     net_input_shapes={"/model.0/conv/Conv": [1, 3, 224, 224]}
    # )

    processor.parse_onnx_model(
        start_node_names=["/model.0/conv/Conv"],  # Keep the same start node
        end_node_names=[
            "/model.22/cv4.0/cv4.0.2/Conv",
            "/model.22/Concat_4",
            "/model.22/cv4.1/cv4.1.2/Conv",
            "/model.22/cv4.2/cv4.2.2/Conv",
            "/model.22/Concat_5",
            #"/model.22/Unsqueeze_15",
            "/model.22/Concat_6"
        ],
    net_input_shapes={"/model.0/conv/Conv": [1, 3, 224, 224]}
    )
    # Step 2: Preprocess images for calibration
    calib_dataset = processor.preprocess_images(images_path="/home/rubesh/Desktop/sweta/Mtech_internship/others/all_images_LTA")

    # Step 3: Optimize the model
    processor.optimize_model(calib_dataset)

    # Step 4: Compile the model
    processor.compile_model()