# Input file containing intermediate tensor shapes
input_file_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/intermediate_tensor_shapes.txt"

# Output file to save tensors with "dynamic" in their shapes
output_file_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/model_to_onnx_conv/dynamic_tensors.txt"

# Open the input file and filter lines with "dynamic"
with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    output_file.write("Tensors with 'dynamic' in their shapes:\n")
    for line in input_file:
        if "dynamic" in line:
            output_file.write(line)

print(f"Tensors with 'dynamic' shapes saved to: {output_file_path}")