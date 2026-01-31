import torch
import torch.nn as nn


class Conv1DEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, seq_len, kernel_size=3, stride=1, padding=0, dilation=1, activation=nn.ReLU):
        super(Conv1DEncoder, self).__init__()
        layers = []
        prev_channels = in_channels
        curr_seq_len = seq_len
        
        self.seq_len_tracker = [curr_seq_len]
        for out_channels in hidden_channels:
            layers.append(nn.Conv1d(
                prev_channels, out_channels, kernel_size, 
                stride=stride, padding=padding, dilation=dilation))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation())
            prev_channels = out_channels
            curr_seq_len = (curr_seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1 # Update sequence length after Conv1D
            self.seq_len_tracker.append(curr_seq_len)
            
        self.conv_layers = nn.Sequential(*layers)
        self.flattened_dim = prev_channels * curr_seq_len
        self.fc = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)  # Flatten
        return self.fc(x)
        # print(x.shape)
        # for layer in self.conv_layers:
        #     x = layer(x)
        #     if isinstance(layer, nn.Conv1d):
        #         print(x.shape)
        # x = x.view(x.shape[0], -1) 
        # print(x.shape)
        # return self.fc(x)
        
    
class Conv1DDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, in_channels, seq_len_tracker, kernel_size=3, stride=1, padding=0, dilation=1, activation=nn.ReLU):
        super(Conv1DDecoder, self).__init__()
        self.hidden_channels = hidden_channels[::-1]
        self.seq_len_tracker = seq_len_tracker[::-1]
        self.fc = nn.Linear(latent_dim, self.seq_len_tracker[0] * self.hidden_channels[0])
        
        layers = []
        prev_channels = self.hidden_channels[0]
        
        seq_len_idx = 0
        for out_channels in self.hidden_channels[1:]:
            output_padding = self.seq_len_tracker[seq_len_idx + 1] - 1 - \
                ((self.seq_len_tracker[seq_len_idx] - 1) * stride - 2 * padding + dilation * (kernel_size - 1))
            layers.append(nn.ConvTranspose1d(
                prev_channels, out_channels, kernel_size, 
                stride=stride, padding=padding, dilation=dilation, 
                output_padding=output_padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation())
            prev_channels = out_channels
            seq_len_idx += 1
            
        output_padding = self.seq_len_tracker[seq_len_idx + 1] - 1 - \
                ((self.seq_len_tracker[seq_len_idx] - 1) * stride - 2 * padding + dilation * (kernel_size - 1))    
        layers.append(nn.ConvTranspose1d(prev_channels, in_channels, kernel_size, 
                                         stride=stride, padding=padding, dilation=dilation, 
                                         output_padding=output_padding))
        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z):
        batch_size = z.shape[0]
        x = self.fc(z)
        x = x.view(batch_size, self.hidden_channels[0], self.seq_len_tracker[0])
        return self.deconv_layers(x)
        # print(x.shape)
        # for layer in self.deconv_layers:
        #     x = layer(x)
        #     if isinstance(layer, nn.ConvTranspose1d):
        #         print(x.shape)
        # return x[:, :, :self.seq_len_tracker[-1]]

    
class Conv1DAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, seq_len, kernel_size=3, stride=1, padding=0, dilation=1, activation=nn.ReLU):
        super(Conv1DAutoencoder, self).__init__()
        self.encoder = Conv1DEncoder(
            in_channels, hidden_channels, latent_dim, seq_len, 
            kernel_size, stride, padding, dilation, activation)

        self.decoder = Conv1DDecoder(
            latent_dim, hidden_channels, in_channels, self.encoder.seq_len_tracker, 
            kernel_size, stride, padding, dilation, activation)
    def forward(self, x, downstream=False):
        z = self.encoder(x)
        if downstream:
            return z
        return self.decoder(z)

    
if __name__ == '__main__':
    # Example usage
    batch_size, in_channels, seq_len = 10, 18, 18 * 768
    hidden_channels = [32, 64]  # Conv layer sizes
    latent_dim = 128  # Latent space size
    kernel_size = 5
    stride = 5
    padding = 0
    dilation = 1
    model = Conv1DAutoencoder(in_channels, hidden_channels, latent_dim, seq_len, 
                              kernel_size, stride, padding, dilation)
    x = torch.rand(batch_size, in_channels, seq_len)  # Random input tensor
    x_recon = model(x, downstream=True)

    print("Input shape:", x.shape)
    print("Reconstructed shape:", x_recon.shape)