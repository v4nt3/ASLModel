import torch
import torch.nn as nn

class C3D(nn.Module):
    """
    usa AdaptiveAvgPool3d para obtener un tamaño fijo antes del classifier.
    Asegúrate de instanciarlo con `C3D(num_classes=num_classes, input_frames=Config.NUM_FRAMES)`.
    """
    def __init__(self, num_classes=1000, dropout=0.5, input_frames=40):
        super(C3D, self).__init__()
        self.input_frames = input_frames

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # Bloques convolucionales (igual que tu versión)
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
        # quitamos el Pool3d agresivo que dejaba dimensiones muy pequeñas si quieres:
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Adaptative pooling para obtener tamaño fijo (1,1,1) independientemente de T,H,W
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Después de adaptive_pool la salida tiene shape (N, 512, 1, 1, 1) -> flattened = 512
        flattened_size = 512

        # Clasificador: puedes mantener 4096 si tienes memoria, o usar tamaños más pequeños.
        # Aquí uso una opción intermedia (512 -> 1024 -> 512 -> num_classes) para ser más eficiente.
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        # Inicialización (opcional, pero recomendable)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Input x expected shape: (N, 3, T, H, W)
        """
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

        # adaptive pool -> (N, 512, 1, 1, 1)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # (N, 512)

        x = self.classifier(x)     # logits (sin softmax)
        return x
