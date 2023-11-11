import torch
# Import nn submodule to subclass our neural network
from torch import nn

def initialize_model(Pytorch_file_path):

    """Initializes the TinyVGG Model for Inference

    Args:
        Pytorch_file_path: A filepath to a pytorch model file (.pt or .pth)

    Returns:
        A pre-trained TinyVGG PyTorch model for inference

    Raises:
        FileNotFound: An error occurred accessing the directory.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # This is a TinyVGG Architecture.
    # Details of this architecture can be found at : https://poloclub.github.io/cnn-explainer/
    class TinyVGG(nn.Module):
        def __init__(self, input_shape, hidden_units, output_shape):
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*7*7,
                          out_features=output_shape)
            )
        def forward(self, x):
            x=self.conv_block_1(x)
            x=self.conv_block_2(x)
            x=self.classifier(x)
            return x

    # Instantiate the model
    model = TinyVGG(input_shape=1,
                        output_shape=10,
                        hidden_units=10).to(device)
    
    # Load the Weights
    model.load_state_dict(torch.load(f=Pytorch_file_path,map_location=torch.device(device)))
    
    return model
                                    
