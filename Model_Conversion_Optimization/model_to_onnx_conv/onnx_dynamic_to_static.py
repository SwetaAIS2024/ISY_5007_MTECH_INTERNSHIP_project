import onnx

# Load the ONNX model
model_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/yolov8n-seg.onnx"
model = onnx.load(model_path)


# Modify the input shape
for input_tensor in model.graph.input:
    if input_tensor.name == "images":  # Replace with your input tensor name
        input_tensor.type.tensor_type.shape.dim[0].dim_value = 1  # Batch size
        input_tensor.type.tensor_type.shape.dim[2].dim_value = 224  # Height
        input_tensor.type.tensor_type.shape.dim[3].dim_value = 224  # Width

# Manually set static shapes for outputs
for output_tensor in model.graph.output:
    if output_tensor.name == "output0":  # Replace with your actual output name
        output_tensor.type.tensor_type.shape.dim[0].dim_value = 1  # Batch size
        output_tensor.type.tensor_type.shape.dim[1].dim_value = 116  # Channels
        output_tensor.type.tensor_type.shape.dim[2].dim_value = 56  # Height (example)
    elif output_tensor.name == "output1":  # Replace with your actual output name
        output_tensor.type.tensor_type.shape.dim[0].dim_value = 1  # Batch size
        output_tensor.type.tensor_type.shape.dim[1].dim_value = 32  # Channels
        output_tensor.type.tensor_type.shape.dim[2].dim_value = 56  # Height
        output_tensor.type.tensor_type.shape.dim[3].dim_value = 56  # Width

# Save the modified model
modified_model_path = model_path.replace(".onnx", "_static_outputs.onnx")
onnx.save(model, modified_model_path)
print(f"Modified model saved to: {modified_model_path}")
