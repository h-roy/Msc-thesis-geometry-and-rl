import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


max = 1
min = -1


class Encoder(nn.Module):
    """
    Encoder to embed image observation (3, 64, 64) to vector (1024,)
    """

    def __init__(self, state_dim=1024):
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.ff = nn.Linear(256, state_dim)
        # self.dropout = nn.Dropout2d(0.1)
        # self.batchnorm1 = nn.BatchNorm2d(32)
        # self.batchnorm2 = nn.BatchNorm2d(64)
        # self.batchnorm3 = nn.BatchNorm2d(128)

    def forward(self, obs):
        # hidden = self.batchnorm1(F.relu(self.cv1(obs)))
        # hidden = self.dropout(self.batchnorm2(F.relu(self.cv2(hidden))))
        # hidden = self.batchnorm3(F.relu(self.cv3(hidden)))
        hidden = F.relu(self.cv1(obs))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        hidden = F.relu(self.cv4(hidden))
        embedded_obs = hidden.reshape(hidden.size(0), -1)
        # state = F.relu(self.ff(embedded_obs))
        state = torch.sigmoid(self.ff(embedded_obs)) * (max - min) + min
        return state


class Inverse(nn.Module):
    def __init__(self, state_dim=128, hidden_dim=1024, actions=4):
        super(Inverse, self).__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(state_dim + 1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, actions)
        )
        self.ff = torch.nn.Linear(state_dim, 1)

    def forward(self, z0, z1):
        context = F.relu(self.ff(z0))
        pred = self.body(torch.cat((context, z1 - z0), -1))
        # context = torch.cat((z0, z1 - z0), -1)
        # pred = self.body(z1 - z0)
        return pred


class Contrastive(nn.Module):
    def __init__(self, state_dim=128, hidden_dim=1024):
        super(Contrastive, self).__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(state_dim + 1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        self.ff = torch.nn.Linear(state_dim, 1)

    def forward(self, z0, z1):
        context = F.relu(self.ff(z0))
        context = torch.cat((context, z1 - z0), -1)
        pred = self.body(context)
        return pred

class ProbabilisticTransitionModel(nn.Module):

    def __init__(self, encoder_feature_dim, action_shape, layer_width=512, announce=True, max_sigma=1e-2, min_sigma=1e-4):
        super().__init__()
        print(encoder_feature_dim + action_shape)
        self.fc = nn. Linear(encoder_feature_dim + action_shape, layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)
        if announce:
            print("Probabilistic transition model chosen.")

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = torch.sigmoid(self.fc_mu(x)) * (max - min) + min
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        dist = td.normal.Normal(mu, sigma)
        return dist

    #def sample_prediction(self, x):
    #    mu, sigma = self(x)
    #    eps = torch.randn_like(sigma)
    #    return mu + sigma * eps


