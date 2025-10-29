import torch #type: ignore
import torch.nn as nn #type: ignore

class C3D(nn.Module):
    """
    C3D (3D Convolutional Network)
    Simple but effective 3D CNN architecture
    """
    def __init__(self, num_classes=2303, dropout=0.5, input_frames=40):
        super(C3D, self).__init__()
        
        self.input_frames = input_frames
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # Conv layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Input: (batch, 3, 40, 112, 112)
        # After all pooling: temporal_dim = 40 -> 40 -> 20 -> 10 -> 5 -> 2
        #                    spatial_dim = 112 -> 56 -> 28 -> 14 -> 7 -> 3
        self.flattened_size = self._get_flattened_size()
        
        # FC layers
        self.fc6 = nn.Linear(self.flattened_size, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
    
    def _get_flattened_size(self):
        """Calculate the flattened size after all conv and pooling layers"""
        # Create a dummy input to calculate output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.input_frames, 112, 112)
            x = self.relu(self.conv1(dummy_input))
            x = self.pool1(x)
            
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            
            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)
            
            return x.view(1, -1).size(1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        
        return x
