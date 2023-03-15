import torch.nn as nn
import torch.nn.functional as F
import torch

class Actual_NeRF(nn.Module):
    def __init__(self, pos_embd_size, W, dir_embd_size):
        super(Actual_NeRF, self).__init__()

        self.Layers1 = nn.Sequential(
            nn.Linear(pos_embd_size * 6 + 3, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
        )

        self.Layers2 = nn.Sequential(
            nn.Linear(pos_embd_size * 6 + 3 + W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W + 1),
        )

        self.Layers3 = nn.Sequential(
            nn.Linear(dir_embd_size * 6 + 3 + W, int(W / 2)),
            nn.ReLU(),
        )

        self.Layers4 = nn.Sequential(
            nn.Linear(int(W / 2), 3),
            nn.Sigmoid()
        )

    def forward1(self, pos, dir):
        output1 = self.Layers1(pos)
        input2 = torch.cat((output1, pos), dim=-1)  ###########skipped connection
        output2 = self.Layers2(input2)
        sigma = nn.ReLU(output2[:, -1])
        input3 = torch.cat((output2[:, :-1], dir), dim=-1)
        output3 = self.Layers3(input3)
        colors = self.Layers4(output3)
        return colors, sigma


class tinyNeRF(nn.Module):
    def __init__(self, size, W):
        super(tinyNeRF, self).__init__()

        self.Layers1 = nn.Sequential(
            nn.Linear(size - 6, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
        )

        self.Layers2 = nn.Sequential(
            nn.Linear(size - 6 + W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, 4),
        )

    def forward2(self, pos):
        output1 = self.Layers1(pos)
        input2 = torch.cat((output1, pos), dim=-1)  ###########skipped connection
        output2 = self.Layers2(input2)
        sigma = torch.relu(output2[:, -1])
        colors = torch.sigmoid(output2[:, :-1])
        return colors, sigma


class Nerf(nn.Module):
    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(Nerf, self).__init__()
        
        self.input_layer = nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        
        self.hidden_layers = nn.ModuleList()
        for i in range(3):
            layer = nn.Linear(filter_size, filter_size)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.hidden_layers.append(layer)
        
        self.output_layer = nn.Linear(filter_size, 4)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
        self.activation = F.elu

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            
        x = self.output_layer(x)
        return x
