import cv2
import onnxruntime as ort
import numpy as np
import time  # Import time module for sleep and measuring inference time

# Path to the quantized model
quantized_model_path = 'F:/detection-ship/pytorch-YOLOv4/yolov4_1_3_416_416_static_quantized.onnx'

# Load the quantized model into ONNX Runtime for inference
session = ort.InferenceSession(quantized_model_path)

# Define the input tensor (replace with your actual image preprocessing)
def preprocess_image(image):
    # Resize the image to 416x416 (input size for YOLOv4)
    image_resized = cv2.resize(image, (416, 416))
    
    # Normalize the image
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    
    # Convert to CHW format
    image_input = image_normalized.transpose(2, 0, 1).astype(np.float32)  # HWC to CHW
    image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension (1, 3, 416, 416)
    
    return image_resized, image_input

# Load the video
video_path = 'F:/detection-ship/Boat Video/2024-04-26_16-28-49.mp4'
cap = cv2.VideoCapture(video_path)

# Video output settings
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video output
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (1280, 1024))  # Adjust size as per your video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    start_time = time.time()
    image_resized, image_input = preprocess_image(frame)
    inference_times = time.time() - start_time
    
    print(f"Resized Time : {inference_times:.4f} seconds")
    
    

    # Record time before inference
    start_time = time.time()

    # Run inference on the frame
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_input})

    # Record time after inference
    inference_time = time.time() - start_time

    # Extracting boxes and class scores
    boxes = outputs[0]  # Bounding box coordinates [x, y, w, h]
    scores = outputs[1]  # Class probabilities (80 classes for COCO)

    # Flatten the box and score arrays
    boxes = boxes.reshape(-1, 4)  # Reshape to (num_boxes, 4)
    scores = scores.reshape(-1, 80)  # Reshape to (num_boxes, 80)

    # Confidence threshold for displaying boxes
    conf_threshold = 0.65

    # Initialize a list to store valid boxes
    valid_boxes = []

    # Iterate over the boxes and scores
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        
        # Get class probabilities and the highest one
        class_confidence = np.max(score)
        class_id = np.argmax(score)
        
        # If the max class confidence is above the threshold
        if class_confidence > conf_threshold:
            x, y, w, h = box
            x = int(x * image_resized.shape[1])
            y = int(y * image_resized.shape[0])
            w = int(w * image_resized.shape[1])
            h = int(h * image_resized.shape[0])
            
            valid_boxes.append((x, y, w, h, class_id, class_confidence))

    # Draw valid boxes on the frame
    for (x, y, w, h, class_id, conf) in valid_boxes:
        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the class label and confidence score
        cv2.putText(frame, f"Class: {class_id} Conf: {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Display the resulting frame
    cv2.imshow('RESIZED', image_resized)
    
    # Write the frame to output video
    out.write(frame)

    # Display the inference time for the frame
    print(f"Inferece Time for Current Frame: {inference_time:.4f} seconds")

    # Add a 400 milliseconds sleep
    # time.sleep(0.1)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
