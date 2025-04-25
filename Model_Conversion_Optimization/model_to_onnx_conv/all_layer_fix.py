import onnx

# Load the ONNX model
#model_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/yolov8n-seg_static_outputs_inferred.onnx"
model_path ="/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/best_inferred.onnx"
model = onnx.load(model_path)

# Fix dynamic shapes
for value_info in model.graph.value_info:
    shape = value_info.type.tensor_type.shape
    for dim in shape.dim:
        if dim.dim_param == "dynamic" or dim.dim_value == 0:
            # Replace 'dynamic' or zero dimensions with static values
            dim.dim_value = 1  # Replace with appropriate static value

# Save the modified model
fixed_model_path = model_path.replace(".onnx", "_fixed.onnx")
onnx.save(model, fixed_model_path)
print(f"Modified model saved to: {fixed_model_path}")