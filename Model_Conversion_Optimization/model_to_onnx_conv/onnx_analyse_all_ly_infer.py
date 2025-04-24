import onnx
from onnx import shape_inference

#onnx_model_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/yolov8n-seg_static_outputs_inferred_fixed.onnx"
onnx_model_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/best.onnx"

# Load the ONNX model
model = onnx.load(onnx_model_path)

# Validate the ONNX model
try:
    onnx.checker.check_model(model)
    print("ONNX model is valid!")
except onnx.checker.ValidationError as e:
    print(f"ONNX model validation failed: {e}")

# Infer shapes for the model
print("\nInferring shapes for the ONNX model...")
inferred_model = shape_inference.infer_shapes(model)

# Save the inferred model (optional, for debugging purposes)
inferred_model_path = onnx_model_path.replace(".onnx", "_inferred.onnx")
onnx.save(inferred_model, inferred_model_path)
print(f"Inferred model saved to: {inferred_model_path}")

# Print input and output shapes
print("\nModel Input and Output Shapes:")
for input_tensor in inferred_model.graph.input:
    input_shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Input: {input_tensor.name}, Shape: {input_shape}")

for output_tensor in inferred_model.graph.output:
    output_shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in output_tensor.type.tensor_type.shape.dim]
    print(f"Output: {output_tensor.name}, Shape: {output_shape}")

# Print all nodes in the model
print("\nModel Nodes:")
for node in inferred_model.graph.node:
    print(f"Node: {node.name}, OpType: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")