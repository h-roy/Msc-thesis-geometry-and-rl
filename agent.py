import torch
import torch.nn.functional as F

from models import Encoder, Inverse, Contrastive, ProbabilisticTransitionModel


class Geometry_Agent(object):
    def __init__(
            self,
            action_shape,
            memory,
            device,
            state_dim=128,
            lr=1e-3,
            smoothness_max_dz=0.01,
            smootheness_coeff=20,  # = 10
            encoder_coeff=20,
            contrastive_coeff=0.1
    ):
        self.device = device
        self.loss_array = {'Inverse': [], 'Orthogonal': [],
                           'Smoothness': [], 'Transition': [], 'Contrastive': []}
        self.memory = memory
        # self.memory = deque(maxlen=MAX_MEMORY_LEN)

        self.action_shape = action_shape
        self.transition_model = ProbabilisticTransitionModel(state_dim, action_shape).to(device)
        self.smoothness_max_dz = smoothness_max_dz
        self.smootheness_coeff = smootheness_coeff
        self.encoder_coeff = encoder_coeff
        self.contrastive_coeff = contrastive_coeff
        self.encoder = Encoder(state_dim).to(device)
        self.inverse = Inverse(state_dim).to(device)
        self.contrastive = Contrastive(state_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=lr
        )
        self.decoder_optimizer = torch.optim.Adam(
            list(self.inverse.parameters()) +
            list(self.contrastive.parameters()), lr=lr
        )
        self.transition_optimizer = torch.optim.Adam(
            self.transition_model.parameters(), lr=lr
        )
        self.train()

    def train(self, training=True):
        # Sets module to training mode
        self.training = training
        self.encoder.train(training)

    def compute_transition_loss(self, z0, z1, z2, action_one_hot, next_action_one_hot):
        dist_1 = self.transition_model(torch.cat([z0, action_one_hot], dim=1))
        dist_2 = self.transition_model(torch.cat([z1, next_action_one_hot], dim=1))
        loss = torch.mean(-dist_1.log_prob(z1) - dist_2.log_prob(z2))
        # loss = torch.nn.MSELoss()(z1, dist_1.rsample()) + torch.nn.MSELoss()(z2, dist_2.rsample())
        self.loss_array['Transition'].append(loss.detach().cpu().numpy())
        return loss

    def compute_inverse_loss(self, z0, z1, z2, action, next_action):
        l_inv = torch.nn.CrossEntropyLoss()(self.inverse(z0, z1), target=action) + torch.nn.CrossEntropyLoss()(
            self.inverse(z1, z2), target=next_action)
        self.loss_array['Inverse'].append(l_inv.detach().cpu().numpy())
        return l_inv

    def compute_contrastive_loss(self, z0, z1, z2):
        with torch.no_grad():
            N = len(z1)
            idx = torch.randperm(N)  # shuffle indices of next states
        z1_neg = z1.view(N, -1)[idx].view(z1.size())
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_real_transition_1 = torch.cat([torch.ones(N), torch.zeros(N)], dim=0).to(z0.device)
        log_pr_real_1 = self.contrastive(z0_extended, z1_pos_neg)
        l_contr_1 = torch.nn.BCEWithLogitsLoss()(input=log_pr_real_1, target=is_real_transition_1.unsqueeze(-1).float())
        idx2 = torch.randperm(N)
        z2_neg = z2.view(N, -1)[idx2].view(z2.size())
        z1_extended = torch.cat([z1, z1], dim=0)
        z2_pos_neg = torch.cat([z2, z2_neg], dim=0)
        is_real_transition_2 = torch.cat([torch.ones(N), torch.zeros(N)], dim=0).to(z1.device)
        log_pr_real_2 = self.contrastive(z1_extended, z2_pos_neg)
        l_contr_2 = torch.nn.BCEWithLogitsLoss()(input=log_pr_real_2, target=is_real_transition_2.unsqueeze(-1).float())
        l_contr = l_contr_1 + l_contr_2
        self.loss_array['Contrastive'].append(l_contr.detach().cpu().numpy())
        return l_contr

    def compute_smoothness_loss(self, z0, z1, z2):
        with torch.no_grad():
            dimensionality_scale_factor = torch.sqrt(
                torch.as_tensor(z0.shape).float()[-1])  # distance scales as ~sqrt(dim)
        dz_1 = torch.norm(z1 - z0, dim=-1, p=2) / dimensionality_scale_factor
        dz_2 = torch.norm(z2 - z1, dim=-1, p=2) / dimensionality_scale_factor
        excess = torch.nn.functional.relu(dz_1 - self.smoothness_max_dz) + torch.nn.functional.relu(
            dz_2 - self.smoothness_max_dz)
        l_smoothness = torch.nn.MSELoss()(excess, torch.zeros_like(excess))
        self.loss_array['Smoothness'].append(l_smoothness.detach().cpu().numpy())
        return l_smoothness

    def compute_orthogonal_loss(self, z0, z1, z2, action, next_action):
        cosine_sim = torch.nn.CosineSimilarity()(z1 - z0, z2 - z1)
        label = (action == next_action).float()
        l_orthogonal = torch.nn.MSELoss()(cosine_sim, label)
        self.loss_array['Orthogonal'].append(l_orthogonal.detach().cpu().numpy())
        return l_orthogonal

    def update_encoder(self, batch_size):
        obs, action, reward, next_obs, not_done, _, _, next_action, next_obs_2 = self.memory.sample(
            batch_size=batch_size)
        action = action.to(torch.int64)
        next_action = next_action.to(torch.int64)
        obs = obs.transpose(1, 3).transpose(2, 3)
        next_obs = next_obs.transpose(1, 3).transpose(2, 3)
        next_obs_2 = next_obs_2.transpose(1, 3).transpose(2, 3)
        z0 = self.encoder(obs)
        z1 = self.encoder(next_obs)
        z2 = self.encoder(next_obs_2)
        l_inv = self.compute_inverse_loss(z0, z1, z2, action, next_action)
        l_contr = self.compute_contrastive_loss(z0, z1, z2)
        l_smoothness = self.compute_smoothness_loss(z0, z1, z2)
        l_orthogonal = self.compute_orthogonal_loss(z0, z1, z2, action, next_action)
        loss = l_inv + \
               self.smootheness_coeff * (self.smootheness_coeff * l_smoothness + l_orthogonal) + \
               self.contrastive_coeff * l_contr
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss, l_inv, l_smoothness, l_orthogonal, l_contr

    def update_transition(self, batch_size):
        obs, action, reward, next_obs, not_done, _, _ = self.memory.sample(batch_size=batch_size)
        action = action.to(torch.int64)
        action = F.one_hot(action, num_classes=self.action_shape)
        transition_loss = self.update_transition_model(obs, action, next_obs)
        self.transition_optimizer.zero_grad()
        transition_loss.backward()
        self.transition_optimizer.step()
        return transition_loss

    def update(self, batch_size):
        obs, action, reward, next_obs, not_done, _, _, next_action, next_obs_2 = self.memory.sample(
            batch_size=batch_size)
        action = action.to(torch.int64)
        next_action = next_action.to(torch.int64)
        obs = obs.transpose(1, 3).transpose(2, 3)
        next_obs = next_obs.transpose(1, 3).transpose(2, 3)
        next_obs_2 = next_obs_2.transpose(1, 3).transpose(2, 3)
        z0 = self.encoder(obs)
        z1 = self.encoder(next_obs)
        z2 = self.encoder(next_obs_2)
        l_inv = self.compute_inverse_loss(z0, z1, z2, action, next_action)
        l_contr = self.compute_contrastive_loss(z0, z1, z2)
        l_smoothness = self.compute_smoothness_loss(z0, z1, z2)
        l_orthogonal = self.compute_orthogonal_loss(z0, z1, z2, action, next_action)
        encoder_loss = l_inv + self.smootheness_coeff * (self.smootheness_coeff * l_smoothness + l_orthogonal) + \
                       self.contrastive_coeff * l_contr
        action_one_hot = F.one_hot(action, num_classes=self.action_shape)
        next_action_one_hot = F.one_hot(next_action, num_classes=self.action_shape)
        transition_loss = self.compute_transition_loss(z0, z1, z2, action_one_hot, next_action_one_hot)
        total_loss = self.encoder_coeff * encoder_loss + transition_loss
        self.transition_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.transition_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return total_loss, transition_loss, encoder_loss