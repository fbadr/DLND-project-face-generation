class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        
        self.conv_dim = conv_dim
        
        # 16x16
        self.conv1 = conv(3, conv_dim*2, 4, batch_norm=False)
        # 8x8
        self.conv2 = conv(conv_dim*2, conv_dim*4, 4)
        # 4x4
        self.conv3 = conv(conv_dim*4, conv_dim*8, 4)
        # 2x2
        self.conv4 = conv(conv_dim*8, conv_dim*16, 4)
        # 1x1
        self.conv5 = conv(conv_dim*16, conv_dim*32, 4)
        self.fc = nn.Linear(conv_dim*32*1*1, 1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv5(out), negative_slope=0.2)
        
        #flatten the output
        # out = out.view(-1, self.conv_dim*16*4*4)
        out = out.view(out.size(0), -1)
        
        output = self.fc(out)
        
        return output