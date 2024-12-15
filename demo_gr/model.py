import torch
import torchvision

from torch import nn

def create_effnet_b3_model(num_classes:int = 3,
                           seed:int = 42):

    # Get pretrained efficientnet model
    model = torchvision.models.efficientnet_b3(weights='DEFAULT')
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT 
    transform = weights.transforms()

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Recreate classifier head
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1536, # Check torch summary for this value
                        out_features=num_classes, # same number of output units as our number of classes
                        bias=True))
    
    return model, transform