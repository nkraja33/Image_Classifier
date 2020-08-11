# Argument Parser for Train & Predict Scripts
import argparse
#Object Initialization 
parser = argparse.ArgumentParser()
#Defining Arguments for Train model
def train_arg():
    parser.add_argument('data_dir', action="store", default="./flowers")
    parser.add_argument('--gpu', action="store", default="gpu")
    parser.add_argument('--save_dir', dest='save_dir', action="store", default="./img_class2_checkpoint.pth", help="Enter the location to save the files")  
    parser.add_argument('--arch', dest='arch', action="store", default="vgg16", help="Enter the Required Architecture, Default is VGG16")
    parser.add_argument('--lr', dest='learning_rate', action="store", type=float, default=0.001, help="Enter the learning rate in float, Default is 0.001")
    parser.add_argument('--dropout', dest='dropout', action="store", type=float, default=0.5, help="Enter the Dropout Value, Default is 0.5")
    parser.add_argument('--epochs', dest='epochs', action="store", type=int, default=5, help="Enter the number of Epochs to train the model, Default is 5") 
    parser.add_argument('--hidden_layers', dest='hidden_layers', type=int, default=4096, help="Enter the numer of hidden layers, Default is 4096")
    return parser.parse_args()
#Defining Arguments for Predecting
def predict_arg():
    parser.add_argument('image', action="store", default="flowers/test/10/image_07117.jpg", help="Enter the path of the image to predict")
    parser.add_argument('checkpoint', default="./img_class2_checkpoint.pth", action="store", type=str, help="Enter the checkpoint path")
    parser.add_argument('--gpu', action="store", default="cuda")
    parser.add_argument('--topk', dest='topk', action="store", type=int, default=5, help="Enter the Top-K value, Default is 5")
    parser.add_argument('--category_names', dest='category_names', action="store", default="cat_to_name.json", help="Enter the file to check the labels")
    return parser.parse_args()