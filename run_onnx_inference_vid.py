

import cv2
import numpy as np
import onnxruntime
from tool.utils import load_class_names, plot_boxes_cv2, post_processing
import time
import psutil
def preprocess_frame(frame, input_shape):
    """
    Preprocess the input frame for the ONNX model.
    """
    IN_IMAGE_H, IN_IMAGE_W = input_shape[2], input_shape[3]
    
    # Resize and normalize
    resized = cv2.resize(frame, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)  # Add batch dimension
    img_in /= 255.0  # Normalize pixel values to [0, 1]
    
    return img_in

def run_video_inference(onnx_model_path, video_path, namesfile):
    """
    Perform inference on a video using an ONNX model.
    """
    # Load the ONNX model and specify the execution providers (e.g., CPUExecutionProvider)
    session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'], 
                                       provider_options=[{'num_threads': 4}])  # Adjust thread count as needed

    # session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    print("Model loaded. Input shape: ", session.get_inputs()[0].shape)

    # Get model input shape
    input_shape = session.get_inputs()[0].shape  # [batch_size, channels, height, width]

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
        img_in = preprocess_frame(frame, input_shape)

        # Perform inference
        start_time = time.time()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_in})
        
        inference_time = time.time() - start_time

        # Post-process the outputs
        boxes = post_processing(img_in, 0.4, 0.6, outputs)

        # Draw the results on the frame
        frame = plot_boxes_cv2(frame, boxes[0], class_names=class_names)
        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the frame to the output video
        # out.write(frame)
        print(f"Inferece Time for Current Frame: {inference_time:.4f} seconds")
        # time.sleep(0.1)

    # Release resources
    cap.release()
    # out.release()
    # print(f"Output video saved to {output_video_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 5:
        onnx_model_path = sys.argv[1]
        video_path = sys.argv[2]
        namesfile = sys.argv[3]
        # output_video_path = sys.argv[4]
        
        run_video_inference(onnx_model_path, video_path, namesfile)
    else:
        print("Usage: python run_onnx_inference.py <onnxModelPath> <videoPath> <namesFile> <outputVideoPath>")
