import onnx

# Load the quantized ONNX model
quantized_model_path = "F:/detection-ship/pytorch-YOLOv4/yolov4_1_3_416_416_quantized.onnx"
model = onnx.load(quantized_model_path)

# Check the data type of each tensor in the quantized model
for tensor in model.graph.initializer:
    print(f"Tensor Name: {tensor.name}, Data Type: {tensor.data_type}")
    if tensor.data_type == 1:
        print(f"{tensor.name} is of type FLOAT32")
    elif tensor.data_type == 11:
        print(f"{tensor.name} is of type INT8")
