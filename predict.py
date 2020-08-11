import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import train
from PIL import Image
import Parser
import json

def load_checkpoint(save_dir):
    checkpoint = torch.load(save_dir)
    model = train.model_selection(checkpoint['model_name'])
    
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    pred_image = Image.open(image)
    
    image_process = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ])
                           
    processed_image = image_process(pred_image)
    return processed_image

def predict(image, model, topk, gpu):
    model.to(gpu)
    model.eval()
    
    image = torch.from_numpy(np.expand_dims(process_image(image), axis=0)).type(torch.FloatTensor).to(gpu)
                   
    
    with torch.no_grad():
        output = model.forward(image)
       
    probability = F.softmax(output.data, dim=1)
    
    return probability.topk(topk)

def main():
    args = Parser.predict_arg()
    image_path = args.image
    category = args.category_names
    model = load_checkpoint(args.checkpoint)
    topk = args.topk
    gpu = args.gpu
    
    with open(category, 'r') as f:
        cat_to_name = json.load(f)
    
    probability = predict(image_path, model, topk, gpu)
    label = [cat_to_name[str(index+1)] for index in np.array(probability[1][0])]
    probs = np.array(probability[0][0])
    for i in range(topk):
        print("Flower {} with a probability of {:.3f}%".format(label[i], probs[i]*100))
              
if __name__ == "__main__":
    main()
    
                                  
    
    
    
    
    
    
        
 