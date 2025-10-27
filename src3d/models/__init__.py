from .c3d import C3D
from .r3d import R3D
from .i3d import I3D

def get_model(arch, num_classes, dropout=0.5, input_frames=40):
    """
    Factory function to get model by architecture name
    
    Args:
        arch: Model architecture ('c3d', 'r3d', 'i3d')
        num_classes: Number of output classes
        dropout: Dropout rate
        input_frames: Number of input frames (for C3D)
    
    Returns:
        Model instance
    """
    arch = arch.lower()
    
    if arch == 'c3d':
        return C3D(num_classes=num_classes, dropout=dropout, input_frames=input_frames)
    elif arch == 'r3d':
        return R3D(num_classes=num_classes)
    elif arch == 'i3d':
        return I3D(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from ['c3d', 'r3d', 'i3d']")
