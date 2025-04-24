import numpy as np
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import cv2


# The target can be used as a context manager ("with" statement) to ensure it's released on time.
# Here it's avoided for the sake of simplicity
target = VDevice()

# Loading compiled HEFs to device:
model_name = 'yolov8m_seg'
hef_path = "./yolov8m_seg.hef"
hef = HEF(hef_path)
    
# Configure network groups
configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

# Create input and output virtual streams params
input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)

# Define dataset params
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]
image_height, image_width, channels = input_vstream_info.shape


image_path = "./3_lane.png"
input_size = (640, 640)

im = cv2.imread(image_path)
im_rz = cv2.resize(im, input_size)
im_rgb = cv2.cvtColor(im_rz, cv2.COLOR_BGR2RGB)

ip_tensor = im_rgb.astype(np.uint8).transpose(2, 0, 1)
ip_tensor = np.expand_dims(ip_tensor, axis=0)
dataset = np.array(ip_tensor, dtype=np.float32)

# File paths for saving results
input_shape_file = "input_shape.txt"
input_data_file = "input_data.npy"
output_shape_file = "output_shape.txt"
output_data_file = "output_data.npy"


# Infer 
with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
    input_data = {input_vstream_info.name: dataset}
    with network_group.activate(network_group_params):
        print('Starting inference')
        # Save input shape to a file
        input_shape = input_data[input_vstream_info.name].shape
        with open(input_shape_file, "w") as f:
            f.write(f"Input shape: {input_shape}\n")
        print(f"Input shape is saved to {input_shape_file}")

        # Save input data to a file
        np.save(input_data_file, input_data[input_vstream_info.name])
        print(f"Input data is saved to {input_data_file}")

        # Perform inference
        infer_results = infer_pipeline.infer(input_data)

        # Save output shape to a file
        output_shape = infer_results[output_vstream_info.name].shape
        with open(output_shape_file, "w") as f:
            f.write(f"Output shape: {output_shape}\n")
        print(f"Stream output shape is saved to {output_shape_file}")

        # Save inference results to a file
        np.save(output_data_file, infer_results[output_vstream_info.name])
        print(f"Inference results are saved to {output_data_file}")

def send(configured_network, num_frames):
    configured_network.wait_for_activation(1000)
    vstreams_params = InputVStreamParams.make(configured_network)
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        vstream_to_buffer = {vstream: np.ndarray([1] + list(vstream.shape), dtype=vstream.dtype) for vstream in vstreams}
        for _ in range(num_frames):
            for vstream, buff in vstream_to_buffer.items():
                vstream.send(buff)

def recv(configured_network, vstreams_params, num_frames):
    configured_network.wait_for_activation(1000)
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for _ in range(num_frames):
            for vstream in vstreams:
                data = vstream.recv()

def recv_all(configured_network, num_frames):
    vstreams_params_groups = OutputVStreamParams.make_groups(configured_network)
    recv_procs = []
    for vstreams_params in vstreams_params_groups:
        proc = Process(target=recv, args=(configured_network, vstreams_params, num_frames))
        proc.start()
        recv_procs.append(proc)
    recv_failed = False
    for proc in recv_procs:
        proc.join()
        if proc.exitcode:
            recv_failed = True
            
    if recv_failed:
        raise Exception("recv failed")


num_of_frames = 1

send_process = Process(target=send, args=(network_group, num_of_frames))
recv_process = Process(target=recv_all, args=(network_group, num_of_frames))
recv_process.start()
send_process.start()
print('Starting streaming (hef=\'{}\', num_of_frames={})'.format(model_name, num_of_frames))
with network_group.activate(network_group_params):
    send_process.join()
    recv_process.join()
    
    if send_process.exitcode:
        raise Exception("send process failed")
    if recv_process.exitcode:
        raise Exception("recv process failed")
        
print('Done')

target.release()

