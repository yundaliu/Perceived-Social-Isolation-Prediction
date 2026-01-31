import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.ReLU):
        super(MLPEncoder, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    
class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super(MLPDecoder, self).__init__()
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):  # Reverse order for symmetry
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))  # Final output projection
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)
    
    
class MLPEncoderDecoder(nn.Module):
    def __init__(self, input_shape, hidden_dims, latent_dim, activation=nn.ReLU):
        super(MLPEncoderDecoder, self).__init__()
        input_dim = input_shape[1] * input_shape[2]
        self.encoder = MLPEncoder(input_dim, hidden_dims, latent_dim, activation)
        self.decoder = MLPDecoder(latent_dim, hidden_dims, input_dim, activation)

    def forward(self, x, downstream=False):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        z = self.encoder(x_flat)
        if downstream:
            return z

        x_recon = self.decoder(z)
        return x_recon.view(x.shape)
    

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=2, activation=nn.ReLU):
        super(Classifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())  # Apply activation function
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
class MLPClassifier(nn.Module):
    def __init__(self, 
                 input_shape, 
                 encoder_hidden_dims, 
                 latent_dim, 
                 classifier_hidden_dims,
                 activation=nn.ReLU):
        super(MLPClassifier, self).__init__()
        input_dim = input_shape[2] * input_shape[3]
        self.latent_dim = latent_dim
        self.encoder = MLPEncoder(input_dim, encoder_hidden_dims, latent_dim, activation)
        self.classifier = Classifier(latent_dim, classifier_hidden_dims, activation=activation)
    
    def forward(self, x):
        # reshape data
        batch_size, n_surveys, n_questions, dim = x.shape
        x_flat = x.reshape(batch_size * n_surveys, n_questions * dim)
        
        z = self.encoder(x_flat)
        z = z.reshape(batch_size, n_surveys, self.latent_dim)
        z = torch.mean(z, dim=1)
        return self.classifier(z)
    

if __name__ == '__main__':
#     # Example usage
#     n_samples, channels, input_dim = 10, 1, 128  # 10 samples, 128 input features
#     input_shape = (n_samples, channels, input_dim)
#     hidden_dims = [256, 128, 64]  # List defining hidden layer sizes
#     latent_dim = 32  # Latent space size

#     model = MLPEncoderDecoder(input_shape, hidden_dims, latent_dim)
#     x = torch.rand(n_samples, channels, input_dim)  # Random input tensor
#     x_recon = model(x)

#     print("Input shape:", x.shape)
#     print("Reconstructed shape:", x_recon.shape)
    
    n_samples, n_surveys, n_questions, dim = 32, 5, 18, 768
    input_shape = (n_samples, n_surveys, n_questions, dim)
    encoder_hidden_dims = [64]
    latent_dim = 32
    classifier_hidden_dims = [16, 8]
    
    # initialize model
    model = MLPClassifier(
        input_shape, encoder_hidden_dims, latent_dim, classifier_hidden_dims)
    
    # initialize input
    x = torch.rand(n_samples, n_surveys, n_questions, dim) 
    
    # apply
    out = model(x)
    print("Input shape:", x.shape)
    print("Reconstructed shape:", out.shape)