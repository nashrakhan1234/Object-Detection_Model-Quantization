# Define the file path for val2017.txt
file_path = 'F:/detection-ship/pytorch-YOLOv4/data/val2017.txt'

# Define the base path to replace '/media' with
base_path = 'F:/detection-ship/final-model/tensorflow-yolov4-tflite-master'

# Read the file, modify paths, and write the results back to a new file
with open(file_path, 'r') as file:
    # Read all lines in the file
    lines = file.readlines()

# Modify each line to replace '/media' with the new base path
modified_lines = [line.replace('/media/user/Source/Data/coco_dataset/coco/images', base_path) for line in lines]

# Write the modified lines back to the same file or a new file
with open('modified_val2017.txt', 'w') as file:
    file.writelines(modified_lines)

print("Paths have been modified and saved in 'modified_val2017.txt'.")
