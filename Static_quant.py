# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:49:44 2024

@author: nashra
"""
import os
import numpy as np
import cv2
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader,QuantizationMode
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

from onnxruntime.quantization import quantize_static, QuantType
from onnx import load

# Paths for optimized and quantized models
optimized_model_path = 'F:/detection-ship/pytorch-YOLOv4/mobilenetv2-7-pruned.onnx'
quantized_model_path = 'F:/detection-ship/pytorch-YOLOv4/mobilenetv2-7-quantized.onnx'

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    # Rescale image
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Padding
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.0

    if gt_boxes is None:
        return image_paded
    else:
        # Adjust ground truth boxes
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


# Custom CalibrationDataReader for ONNX quantization
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_folder, batch_size=1, input_size=(416, 416)):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.input_size = input_size
        self.image_files = os.listdir(image_folder)
        self.index = 0

    def preprocess_image(self, img_path):
        """Preprocess image by reading and resizing."""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = image_preprocess(np.copy(img), self.input_size)
        img = img.astype(np.float32)  # Ensure float32 for ONNX
        return img

    def get_next(self) -> dict:
        """Generate the next batch of input data for inference."""
        if self.index >= len(self.image_files):
            return None  # End of data

        # Create list of image paths for batch processing
        batch_paths = [os.path.join(self.image_folder, self.image_files[i]) for i in range(self.index, self.index + self.batch_size)]
        
        # Parallelize image preprocessing
        with ThreadPoolExecutor() as executor:
            batch_images = list(executor.map(self.preprocess_image, batch_paths))

        # Stack and transpose images to match model input shape (batch_size, channels, height, width)
        batch_images = np.stack(batch_images, axis=0)
        batch_images = np.transpose(batch_images, (0, 3, 1, 2))

        # Increment index for next batch
        self.index += self.batch_size
        return {"input": batch_images}

    def __len__(self):
        """Calculate the total number of batches."""
        return len(self.image_files) // self.batch_size




# Define calibration data reader (your implementation of MyCalibrationDataReader)
calibration_data_reader = MyCalibrationDataReader(
    image_folder='F:/detection-ship/final-model/tensorflow-yolov4-tflite-master/val2017', 
    batch_size=1
)

# Perform quantization on the optimized model
quantize_static(
    model_input=optimized_model_path,
    model_output=quantized_model_path,
    calibration_data_reader=calibration_data_reader,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8  # You can also use QuantType.QInt8 if supported
)

print(f"Quantized model saved at {quantized_model_path}")



import time

def benchmark_model(session, input_data):
    input_name = session.get_inputs()[0].name
    
    # Warm-up run (to avoid first-run overhead)
    session.run(None, {input_name: input_data})
    
    # Measure inference time
    start_time = time.time()
    for _ in range(100):  # Run multiple iterations for more stable results
        session.run(None, {input_name: input_data})
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 100
    print(f"Average inference time: {avg_inference_time:.6f} seconds")
    return avg_inference_time

# Load original and quantized models
original_session = ort.InferenceSession(optimized_model_path)
quantized_session = ort.InferenceSession(quantized_model_path)

# Benchmark using a sample input
sample_input = np.random.rand(1, 3, 416, 416).astype(np.float32)  # Replace with actual preprocessed input
print("Original model:")
benchmark_model(original_session, sample_input)

print("Quantized model:")
benchmark_model(quantized_session, sample_input)
