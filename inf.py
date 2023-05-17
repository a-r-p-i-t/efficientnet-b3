import os
import shutil
import torch
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import argparse
import PIL
import glob
import time

print(torch.cuda.is_available())
# Define the transforms to be applied to the input images
parser = argparse.ArgumentParser(description='PyTorch Custom Testing')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
args = parser.parse_args()

if args.advprop:
    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
image_size=640








# ])
test_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


# Set the paths for the test folder, the pre-trained model checkpoint, and the output folder for misidentified images
test_folder = "C:\\Users\\Arpit Mohanty\\ic\\data\\test"
checkpoint = torch.load("C:\\Users\\Arpit Mohanty\\ic\\data\\checkpoint.pth.tar")
output_folder = "C:\\Users\\Arpit Mohanty\\ic\\data\\output"

# Set the device for running the inference
device = torch.device("cuda:0")

# Load the pre-trained model
model = EfficientNet.from_name("efficientnet-b3")
checkpoint = torch.load('C:\\Users\\Arpit Mohanty\\ic\\data\\checkpoint.pth.tar',map_location=torch.device("cuda"))
state_dict = checkpoint['state_dict']
print(state_dict.keys())
model_path = "C:\\Users\\Arpit Mohanty\\ic\\data\\checkpoint.pth.tar"

# Get the size of the file in bytes
model_size_bytes = os.path.getsize(model_path)

# Convert the size to MB
model_size_mb = model_size_bytes / (1024 * 1024)
print(f"Model size: {model_size_mb:.2f} MB")

# for k in list(state_dict.keys()):
#     if 'module' in k:
#         state_dict[k.replace('module.','')] = state_dict[k]
#         del state_dict[k]



model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Define the class names and the tags for misidentified images
class_names = ["empty", "filled"]
tag = "misidentified"

# Create the output folder for misidentified images if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of image filenames in the test folder
image_filenames = glob.glob(os.path.join(test_folder, "**/*.jpg"), recursive=True)


# Initialize the lists to store the predicted labels and true labels
predicted_labels = []
true_labels = []

# Iterate over the images in the test folder
count=0
count1=0
total_inference_time = 0
num_images = len(image_filenames)
start_time1=time.time()
for image_filename in image_filenames:
    count1+=1
    # Load the image and apply the transforms
    image = Image.open(os.path.join(test_folder, image_filename))
    start_time=time.time()
    inputs = test_transforms(image).unsqueeze(0)
    
    # Move the inputs to the device
    inputs = inputs.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    # Get the predicted label
    _, preds = torch.max(outputs, 1)
    predicted_label = preds.item()

    end_time=time.time()
    
    # Append the predicted and true labels to their respective lists
    predicted_labels.append(predicted_label)
    true_label = 0 if "empty" in image_filename else 1
    true_labels.append(true_label)
    
    # If the predicted label is incorrect, move the image to the output folder and tag it
    if predicted_label != true_label:
        count+=1
        output_filename = os.path.join(output_folder, f"{os.path.basename(image_filename).split('.')[0]}_{tag}.jpg")
        shutil.copy(image_filename, output_filename)
        img = Image.open(os.path.join(output_folder, output_filename))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 30) # specify the font and size
        draw.text((10, 10), f"Misclassified as {class_names[predicted_label]}", fill='red',font=font)
        img.save(os.path.join(output_folder, output_filename))
    inference_time = end_time - start_time
    total_inference_time += inference_time
    print(total_inference_time)
       
end_time1=time.time()

# Convert the predicted and true labels to NumPy arrays
predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)
accuracy = np.mean(predicted_labels == true_labels)
avg_inference_time = total_inference_time / num_images
throughput = 1 / avg_inference_time

print(f"Accuracy: {accuracy}")
print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} images per second")
print(f"Total testing time is {end_time1-start_time1} ")
print(f"Total number of test images are {count1}")
print(f"Total number of correctly classified images are {count1-count}")
print(f"Total number of incorrectly classified images are {count}")