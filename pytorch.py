import torch
import cv2
import numpy as np
import time
from tool.utils import load_class_names, plot_boxes_cv2, post_processing  # Assuming these functions exist and work for PyTorch

# Assuming you have a YOLO model class, you need to load the model and its state_dict
from models import Yolov4  # Example: Change this import according to your model's definition

def preprocess_frame(frame, input_shape):
    """
    Preprocess the input frame for the PyTorch model.
    """
    IN_IMAGE_H, IN_IMAGE_W = input_shape[2], input_shape[3]
    
    # Resize and normalize
    resized = cv2.resize(frame, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)  # Add batch dimension
    img_in /= 255.0  # Normalize pixel values to [0, 1]
    
    return img_in

def run_video_inference(pytorch_model, video_path, namesfile):
    """
    Perform inference on a video using a PyTorch model.
    """
    # Set the model to evaluation mode
    pytorch_model.eval()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Define VideoWriter to save output
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Load class names
    class_names = load_class_names(namesfile)

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no frames are left

        # Preprocess the frame
        img_in = preprocess_frame(frame, (1, 3, 416, 416))  # Assuming model input is (1, 3, 416, 416)
        input_tensor = torch.tensor(img_in).float()
        input_tensor = input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Perform inference
        start_time = time.time()
        with torch.no_grad():
            outputs = pytorch_model(input_tensor)

        inference_time = time.time() - start_time

        # Post-process the outputs (you need to define your own post-processing function)
        boxes = post_processing(outputs, 0.4, 0.6)  # Adjust the thresholds as needed

        # Draw the results on the frame
        frame = plot_boxes_cv2(frame, boxes[0], class_names=class_names)
        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the frame to the output video
        # out.write(frame)
        print(f"Inference Time for Current Frame: {inference_time:.4f} seconds")
        # time.sleep(0.1)

    # Release resources
    cap.release()
    # out.release()
    # print(f"Output video saved to {output_video_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 5:
        pytorch_model_path = sys.argv[1]
        video_path = sys.argv[2]
        namesfile = sys.argv[3]
        # output_video_path = sys.argv[4]
        
        # Load PyTorch model class (YOLOv4)
        pytorch_model = Yolov4()  # Initialize the model class
        pytorch_model.load_state_dict(torch.load(pytorch_model_path))  # Load the saved state_dict
        pytorch_model.eval()

        run_video_inference(pytorch_model, video_path, namesfile)
    else:
        print("Usage: python run_pytorch_inference.py <pytorchModelPath> <videoPath> <namesFile> <outputVideoPath>")
