class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # complete init function
        
        self.z_size = z_size
        self.conv_dim = conv_dim
     
        self.fc = nn.Linear(z_size, conv_dim*32*1*1)
        # 2x2
        self.convT1 = deconv(conv_dim*32,conv_dim*16, 4)
        # 4x4
        self.convT2 = deconv(conv_dim*16,conv_dim*8, 4)
        # 8x8
        self.convT3 = deconv(conv_dim*8, conv_dim*4, 4)
        # 16x16
        self.convT4 = deconv(conv_dim*4, conv_dim*2, 4)
        # 32x32
        self.convT5 = deconv(conv_dim*2, 3, 4, batch_norm=False)
        
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        
        out = self.fc(x)
        # should be batch_size, channels , width, height
        out = out.view(-1, self.conv_dim*32, 1, 1)
        
        out = F.relu(self.convT1(out))
        out = F.relu(self.convT2(out))
        out = F.relu(self.convT3(out))
        out = F.relu(self.convT4(out))
        out = F.tanh(self.convT5(out))
        
        return out