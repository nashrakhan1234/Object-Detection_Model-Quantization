# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:57:09 2024

@author: nashra
"""
import onnx

# Load the modified quantized model
modified_model_path = "F:/detection-ship/pytorch-YOLOv4/mobilenetv2-7-quantized.onnx"
model = onnx.load(modified_model_path)

# Validate the model
try:
    onnx.checker.check_model(model)
    print(f"The model at {modified_model_path} is valid.")
except onnx.checker.ValidationError as e:
    print(f"The model at {modified_model_path} is invalid. Error: {e}")

