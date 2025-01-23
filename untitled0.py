# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:46:56 2024

@author: nashra
"""

import onnx
import onnxoptimizer

# Load the ONNX model
model_path = 'F:/detection-ship/pytorch-YOLOv4/mobilenetv2-7-infer.onnx'
optimized_model_path = 'F:/detection-ship/pytorch-YOLOv4/mobilenetv2-7-optimized.onnx'

# Load the original model
model = onnx.load(model_path)

# List available optimization passes
print("Available optimization passes:")
print(onnxoptimizer.get_fuse_and_elimination_passes())


passes = onnxoptimizer.get_fuse_and_elimination_passes()
optimized_model = onnxoptimizer.optimize(model, passes)

# Save the optimized model
onnx.save(optimized_model, optimized_model_path)
print(f"Optimized model saved at {optimized_model_path}")

import onnx

try:
    onnx.checker.check_model(optimized_model)
    print("Optimized model is valid.")
except onnx.checker.ValidationError as e:
    print(f"Optimized model is invalid: {e}")


import time
import numpy as np
import onnxruntime as ort

def measure_inference_time(model_path, input_shape, num_runs=100):
    # Create a session
    session = ort.InferenceSession(model_path)
    
    # Generate a dummy input
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Warm-up
    session.run(None, {input_name: dummy_input})
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy_input})
    avg_time = (time.time() - start_time) / num_runs
    print(f"Average inference time for {model_path}: {avg_time:.4f} seconds")

# Measure performance of original and optimized models
original_model_path = 'F:/detection-ship/pytorch-YOLOv4/mobilenetv2-7-infer.onnx'
measure_inference_time(original_model_path, (1, 3, 416, 416))
measure_inference_time(optimized_model_path, (1, 3, 416, 416))

