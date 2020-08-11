import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
import Parser
import Input

def model_selection(arch):
    if(arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
        model.name = arch
    elif(arch =='alexnet'):
        model = models.alexnet(pretrained=True)
        model.name = arch
    else:
        return "Please select either 'vgg16'(Defualt) or 'alexnet'"
    model.name = arch
    return model

def create_classifier(model, hidden, dropout):
    input_feat = model.classifier[0].in_features
    #Freezing the model parameters to avoid backprop
    for param in model.parameters():
        param.requires_grad = False
    # Creating New Untrained FeedForward Classifier

    classifer = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_feat, hidden, bias=True)),
                            ('ReLu1', nn.ReLU()),
                            ('Dropout1', nn.Dropout(p=dropout)),
                            ('fc2', nn.Linear(hidden, 102, bias=True)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifer
    return model

def validation(model, valid_data, criterion, device):
    accuracy = 0
    valid_loss = 0
    for ii, (images, labels) in enumerate(valid_data):
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        valid_loss += criterion(output, labels)
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

def train(model, train_data, valid_data, criterion, optimizer, epochs, device):
    print ("Training Process Started ......\n")
    steps = 0
    print_every = 40
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for i, (images, labels) in enumerate(train_data):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_data, criterion, device)
                print("Iteration : {}".format(i),
                      "Epoch : {}/{}".format(epoch +1, epochs),
                      "Training Loss : {:.3f}".format(running_loss/print_every),
                      "Validation Loss {:.3f}".format(valid_loss/len(valid_data)),
                      "Validation Accuracy : {:.3f}%".format((accuracy/len(valid_data)*100)))
                running_loss =0
                model.train()
    print("\n Training Process completed") 
    return model

def save_checkpoint(model, train_image_data, save_dir):
    model.class_to_idx = train_image_data.class_to_idx
    checkpoint = {'classifier' : model.classifier,
                  'class_to_idx' : model.class_to_idx,
                  'model_name' : model.name,
                  'state_dict' : model.state_dict()}
    torch.save(checkpoint, save_dir)
    
        

def main():
    
    args = Parser.train_arg()
    hidden = args.hidden_layers
    dropout = args.dropout
    source = args.data_dir
    arch = args.arch
    epochs = args.epochs
    lr = args.learning_rate
    save_dir = args.save_dir         
    train_data, valid_data, test_data, train_image_data = Input.load_data(source)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                       
    model = model_selection(arch)
    model = create_classifier(model, hidden, dropout)
    
    model.to(device)
   
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    
    trained_model = train(model, train_data, valid_data, criterion, optimizer, epochs, device)
    save_checkpoint(model, train_image_data, save_dir)
    
if __name__ == "__main__":
    main()