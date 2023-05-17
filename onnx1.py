import torch
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import os
from efficientnet_pytorch import EfficientNet
import time
print(torch.cuda.is_available())

class ONNXModel:
    def __init__(self, onnx_model_path):
        self.onnx_model = ort.InferenceSession(onnx_model_path)
        self.device = torch.device('cuda')
        self.image_transforms = transforms.Compose([
            transforms.Resize(640,interpolation=Image.BICUBIC),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        model = EfficientNet.from_name('efficientnet-b3')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        state_dict=checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if 'module' in k:
                state_dict[k.replace('module.','')] = state_dict[k]
                del state_dict[k]

        model.load_state_dict(state_dict=state_dict)
        device = torch.device("cuda")
        model.to(device)
        # model.cuda()
        model.eval()

        input_shape = (3, 640, 640)
        dummy_input = torch.randn((1,3,640,640),device=device)
        input_names = ['input']
        output_names = ['output']
        onnx_model_path = "model.onnx"
        torch.onnx.export(model, dummy_input, onnx_model_path,verbose=False)
        model_size_bytes = os.path.getsize(onnx_model_path)
        # print(f"Model size: {model_size_bytes / (1024 * 1024)} MB")
        


        return cls(onnx_model_path),model_size_bytes

    
    def apply_image_transforms(self, image_path):
        image = Image.open(image_path)
        image_tensor = self.image_transforms(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def run_onnx_inference(self, input_tensor):
        input_name = self.onnx_model.get_inputs()[0].name
        input_tensor = input_tensor.to(self.device)

    
        ort_inputs = {input_name: input_tensor.cpu().numpy().astype(np.float32)}
        output_name = self.onnx_model.get_outputs()[0].name
        ort_outs = self.onnx_model.run([output_name], ort_inputs)
        output_tensor = torch.from_numpy(ort_outs[0]).to(self.device)

        _, predicted_class = torch.max(output_tensor, 1)

        ch_end_time=time.time()
        # print(ch_end_time-ch_strt_time)
        print(1)
       
        return predicted_class.item()
    
    def predict(self, test_folder_path):
        correct_predictions = 0
        total_images = 0
        my_inference_time=0
        count=0
        
        for folder_name in ['empty', 'filled']:
            folder_path = os.path.join(test_folder_path, folder_name)
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                    count+=1
                    test_image_path = os.path.join(folder_path, filename)

                    test_image_tensor = self.apply_image_transforms(test_image_path)
                    my_time_start=time.time()

                    # print('PyTorch version before running inference:', torch.__version__)
                    predicted_class = self.run_onnx_inference(test_image_tensor)
                    my_end_time=time.time()
                    # print('PyTorch version after running inference:', torch.__version__)
                    true_class = 0 if folder_name == 'empty' else 1


                    # print('Image:', filename, 'True class:', true_class, 'Predicted class:', predicted_class)
                    if predicted_class == true_class:
                        correct_predictions += 1
                    total_images += 1

                    my_inference_time+=(my_end_time-my_time_start)
                    print(my_inference_time)
                    # print(torch.__version__)

                


        accuracy = correct_predictions / total_images
        total_testing_time = my_inference_time
        avg_inference_time_per_image1=my_inference_time/total_images
        # print(count)
        # print(total_images)
        # print('Accuracy:', accuracy)
        print('Total testing time:', total_testing_time, 'seconds')
        print('Average inference time per image1:', avg_inference_time_per_image1, 'seconds')

    
model,model_size = ONNXModel.from_checkpoint('C:\\Users\\Arpit Mohanty\\ic\\data\\checkpoint.pth.tar')
model.predict('C:\\Users\\Arpit Mohanty\\ic\\data\\test')
print(model_size)



