import onnx

# Load the ONNX model
model_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/yolov8n-seg_static_outputs_inferred_fixed_inferred.onnx"
model = onnx.load(model_path)

# File to save the intermediate tensor shapes
output_file_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/intermediate_tensor_shapes.txt"

# Open the file for writing
with open(output_file_path, "w") as file:
    # Print shapes of all intermediate tensors
    file.write("Intermediate Tensor Shapes:\n")
    print("Intermediate Tensor Shapes:")
    for value_info in model.graph.value_info:
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in value_info.type.tensor_type.shape.dim]
        tensor_info = f"Tensor: {value_info.name}, Shape: {shape}\n"
        print(tensor_info.strip())
        file.write(tensor_info)

print(f"\nIntermediate tensor shapes saved to: {output_file_path}")