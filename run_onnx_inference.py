# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:17:41 2024

@author: nashra
"""

import cv2
import numpy as np
import onnxruntime
from tool.utils import load_class_names, plot_boxes_cv2, post_processing

def preprocess_image(image_path, input_shape):
    """
    Preprocess the input image for the ONNX model.
    """
    # Load the image
    image_src = cv2.imread(image_path)
    IN_IMAGE_H, IN_IMAGE_W = input_shape[2], input_shape[3]
    
    # Resize and normalize
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)  # Add batch dimension
    img_in /= 255.0  # Normalize pixel values to [0, 1]
    
    return img_in, image_src
def run_inference(onnx_model_path, image_path, namesfile, output_image_path='predictions_onnx.jpg'):
    """
    Perform inference using an ONNX model.
    """
    # Load the ONNX model and specify the execution providers (e.g., CPUExecutionProvider)
    session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    print("Model loaded. Input shape: ", session.get_inputs()[0].shape)

    # Get model input shape
    input_shape = session.get_inputs()[0].shape  # [batch_size, channels, height, width]
    
    # Preprocess the image
    img_in, image_src = preprocess_image(image_path, input_shape)
    print("Shape of the processed input: ", img_in.shape)

    # Perform inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})

    # Post-process the outputs
    boxes = post_processing(img_in, 0.4, 0.6, outputs)  # Pass 'outputs' as a positional argument.

    # Load class names
    class_names = load_class_names(namesfile)

    # Visualize the results
    plot_boxes_cv2(image_src, boxes[0], savename=output_image_path, class_names=class_names)
    print(f"Output saved to {output_image_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 5:
        onnx_model_path = sys.argv[1]
        image_path = sys.argv[2]
        namesfile = sys.argv[3]
        output_image_path = sys.argv[4]
        
        run_inference(onnx_model_path, image_path, namesfile, output_image_path)
    else:
        print("Usage: python run_onnx_inference.py <onnxModelPath> <imagePath> <namesFile> <outputImagePath>")
