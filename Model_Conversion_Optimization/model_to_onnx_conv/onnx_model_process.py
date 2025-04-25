import onnx
from onnx import shape_inference
import os
import logging
from ultralytics import YOLO


class ONNXModelProcessor:
    def __init__(self, model_path, log_file="onnx_model_process.log"):
        self.model_path = model_path
        self.model = None

        # Configure logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()

    def log(self, message, level="info"):
        """Log a message at the specified level."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
        print(message)  # Also print to console for immediate feedback

    def convert_to_onnx(self, trained_model_path, dynamic=True, simplify=False):
        """Convert a trained YOLOv8 model to ONNX format."""
        self.log(f"Converting YOLOv8 model to ONNX format: {trained_model_path}")
        if not os.path.exists(trained_model_path):
            error_message = f"Trained model file not found: {trained_model_path}"
            self.log(error_message, level="error")
            raise FileNotFoundError(error_message)

        # Load the YOLOv8 model
        model = YOLO(trained_model_path)
        self.log(f"Loaded YOLOv8 model from: {trained_model_path}")

        # Export the model to ONNX format
        model.export(format="onnx", dynamic=dynamic, opset=12, simplify=simplify)

    def load_model(self):
        """Load the ONNX model."""
        if not os.path.exists(self.model_path):
            error_message = f"Model file not found: {self.model_path}"
            self.log(error_message, level="error")
            raise FileNotFoundError(error_message)
        self.model = onnx.load(self.model_path)
        self.log(f"Loaded ONNX model from: {self.model_path}")

    def validate_model(self):
        """Validate the ONNX model."""
        try:
            onnx.checker.check_model(self.model)
            self.log("ONNX model is valid!")
        except onnx.checker.ValidationError as e:
            error_message = f"ONNX model validation failed: {e}"
            self.log(error_message, level="error")
            raise

    def infer_shapes(self):
        """Infer shapes for the ONNX model."""
        self.log("Inferring shapes for the ONNX model...")
        self.model = shape_inference.infer_shapes(self.model)
        self.log("Shape inference completed.")

    def fix_dynamic_shapes(self, static_value=1):
        """Fix dynamic or zero dimensions in the ONNX model."""
        self.log("Fixing dynamic shapes...")
        for value_info in self.model.graph.value_info:
            shape = value_info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.dim_param == "dynamic" or dim.dim_value == 0:
                    dim.dim_value = static_value  # Replace with the provided static value
        self.log("Dynamic shapes fixed.")

    def save_model(self, suffix="_fixed"):
        """Save the modified ONNX model."""
        fixed_model_path = self.model_path.replace(".onnx", f"{suffix}.onnx")
        onnx.save(self.model, fixed_model_path)
        self.log(f"Modified model saved to: {fixed_model_path}")
        return fixed_model_path

    def analyze_model(self):
        """Print input and output shapes of the ONNX model."""
        self.log("Analyzing model input and output shapes...")
        print("\nModel Input and Output Shapes:")
        for input_tensor in self.model.graph.input:
            input_shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in input_tensor.type.tensor_type.shape.dim]
            self.log(f"Input: {input_tensor.name}, Shape: {input_shape}")

        for output_tensor in self.model.graph.output:
            output_shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in output_tensor.type.tensor_type.shape.dim]
            self.log(f"Output: {output_tensor.name}, Shape: {output_shape}")

        # Print all nodes in the model
        self.log("Listing all nodes in the model...")
        for node in self.model.graph.node:
            self.log(f"Node: {node.name}, OpType: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and fix ONNX models.")
    parser.add_argument("--model_path", type=str, help="Path to the ONNX model file.")
    parser.add_argument("--trained_model_path", type=str, help="Path to the trained YOLOv8 model (.pt file).")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes for the ONNX model.")
    parser.add_argument("--simplify", action="store_true", help="Enable model simplification during export.")
    args = parser.parse_args()

    processor = ONNXModelProcessor(args.model_path or "")
    try:
        if args.trained_model_path:
            # Convert YOLOv8 model to ONNX
            processor.convert_to_onnx(args.trained_model_path, dynamic=args.dynamic, simplify=args.simplify)
        elif args.model_path:
            # Process the ONNX model
            processor.load_model()
            processor.validate_model()
            processor.infer_shapes()
            processor.fix_dynamic_shapes(static_value=1)
            processor.analyze_model()
            processor.save_model()
        else:
            raise ValueError("Either --model_path or --trained_model_path must be provided.")
    except Exception as e:
        processor.log(f"An error occurred: {e}", level="error")