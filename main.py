import gradio as gr 

import torch
from torch import nn
import torchvision
from torchvision import transforms
import pathlib
from pathlib import Path

import utils
import engine
from utils import accuracy_fn
from utils import save_model
import data_setup

# Main code
def main():
    
    # Device agnostic code
    device = "cude" if torch.cuda.is_available() else "cpu"

    # Define the path to the images
    current_dir = pathlib.Path().resolve()
    data_dir = current_dir / "pizza_steak_sushi"
    train_dir = data_dir / Path("train")
    test_dir = data_dir / Path("test")

    """ utils.get_data_libraries(data_dir,
                             train_dir,
                             test_dir,
                             clean_data = True,
                             low_image_treshold = 20,
                             split_train_ratio = 0.8,
                             split_experimental_ratio = 0.2) """
    
    # Get pretrained efficientnet model
    model = torchvision.models.efficientnet_b3(weights='DEFAULT').to(device)
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT 
    auto_transform = weights.transforms()

    BATCH_SIZE = 32
    NUM_WORKERS = 1

    # Creating train and test dataloder, getting class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                                   test_dir,
                                                                                   auto_transform,
                                                                                   BATCH_SIZE,
                                                                                   NUM_WORKERS)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)


    # Freeze the effnet layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1536, # Check torch summary for this value
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Train the model
    results = engine.train(model = model,
                          train_dataloader = train_dataloader,
                          test_dataloader = test_dataloader,
                          optimizer = optimizer,
                          loss_fn = loss_fn,
                          epochs = 5,
                          device = device)
    
    save_model(model = model,
               target_dir = current_dir / Path("demo_gr"),
               model_name = "efficientnet_b3.pth")

if __name__ == "__main__":
    main()


