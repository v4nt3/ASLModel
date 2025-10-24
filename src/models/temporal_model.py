"""
Modelo temporal usando LSTM bidireccional
"""
import torch # type: ignore
import torch.nn as nn # type: ignore

class TemporalLSTM(nn.Module):
    """LSTM bidireccional para modelado temporal de secuencias"""
    
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.4,
                 bidirectional: bool = True):
        """
        Args:
            input_dim: Dimensión de entrada (features de ResNet)
            hidden_size: Tamaño del hidden state de LSTM
            num_layers: Número de capas LSTM
            dropout: Dropout entre capas LSTM
            bidirectional: Usar LSTM bidireccional
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dimensión de salida
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa secuencia temporal
        
        Args:
            x: Tensor de shape (batch_size, seq_len, input_dim)
        
        Returns:
            output: Tensor de shape (batch_size, output_dim)
                   (último hidden state)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        
        if self.bidirectional:
            # Concatenar forward y backward del último layer
            # h_n[-2]: forward del último layer
            # h_n[-1]: backward del último layer
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_size*2)
        else:
            hidden = h_n[-1]  # (batch, hidden_size)
        
        # Layer normalization
        output = self.layer_norm(hidden)
        
        return output


class TemporalAttentionLSTM(nn.Module):
    """LSTM con mecanismo de atención temporal"""
    
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Args:
            input_dim: Dimensión de entrada
            hidden_size: Tamaño del hidden state
            num_layers: Número de capas LSTM
            dropout: Dropout
            bidirectional: Usar LSTM bidireccional
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        self.output_dim = lstm_output_dim
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa secuencia con atención
        
        Args:
            x: Tensor de shape (batch_size, seq_len, input_dim)
        
        Returns:
            output: Tensor de shape (batch_size, output_dim)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * num_directions)
        
        # Calcular attention weights
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Aplicar attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size * num_directions)
        
        # Layer normalization
        output = self.layer_norm(context)
        
        return output
