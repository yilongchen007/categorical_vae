import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import tqdm
import json
from torch.utils.data import DataLoader, TensorDataset

from .model import CategoricalVAE, Encoder, Decoder
from .utils import split_tensor_efficient, categorical_kl_divergence, get_config


class CVAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        print("Using device:", self.device)

        self.data = np.load(config['data_path'])
        self._prepare_data()

        self.save_path = os.path.join(config['save_path'], config['version'])

        unique_counts = [len(self.data['actors']), len(self.data['actors']), len(self.data['actions']), len(self.data['dates'])]
        edim = config['embedding_dim']
        hdim = config['hidden_dim']
        N = config['N']
        K = config['K']

        self.model = CategoricalVAE(
            Encoder(unique_counts, [edim]*4, [N, K], hdim, config['encoder_type']),
            Decoder(unique_counts, [N, K], hdim, config['decoder_type'])
        ).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=config['initial_lr'], momentum=0.0)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config['lr_decay'])

        self.train_loader = DataLoader(
            TensorDataset(self.train_data),
            batch_size=config['batch_size'],
            shuffle=True
        )

        self.max_steps = config['max_steps']
        self.model_save_interval = config['model_save_interval']
        self.temp = config['initial_temp']
        self.min_temp = config['min_temp']
        self.temp_decay = config['temp_decay']

    def _prepare_data(self):
        train_data, _ = split_tensor_efficient(self.data['Y'])
        source, target, action, date = np.where(train_data > 0)
        counts = train_data[source, target, action, date]

        s_r = np.repeat(source, counts)
        t_r = np.repeat(target, counts)
        a_r = np.repeat(action, counts)
        d_r = np.repeat(date, counts)

        df = pd.DataFrame({
            'source_country': self.data['actors'][s_r],
            'target_country': self.data['actors'][t_r],
            'action': self.data['actions'][a_r],
            'date': self.data['dates'][d_r]
        })

        mappings = {
            'source_country': {v: i for i, v in enumerate(self.data['actors'])},
            'target_country': {v: i for i, v in enumerate(self.data['actors'])},
            'action': {v: i for i, v in enumerate(self.data['actions'])},
            'date': {v: i for i, v in enumerate(self.data['dates'])},
        }

        for col in df.columns:
            df[col] = df[col].map(mappings[col])

        tensor_data = torch.tensor(df.values, dtype=torch.long)
        tensor_data = tensor_data[torch.argsort(tensor_data[:, -1])]

        self.train_data = tensor_data.to(self.device)

    def train(self):
        recon_losses, kl_losses, total_losses = [], [], []
        step = 0
        progress_bar = tqdm.tqdm(total=self.max_steps, desc='Training')

        while step < self.max_steps:
            for (x_batch,) in self.train_loader:
                x_batch = x_batch.to(self.device)

                phi, x_prob = self.model(x_batch, self.temp)
                recon = sum([F.cross_entropy(x_prob[i], x_batch[:, i]) for i in range(x_batch.shape[1])]) / x_batch.shape[0]
                kl = torch.mean(torch.sum(categorical_kl_divergence(phi), dim=1))
                loss = recon + kl / 100

                recon_losses.append(recon.item())
                kl_losses.append(kl.item())
                total_losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (step + 1) % self.model_save_interval == 0:
                    os.makedirs(self.save_path, exist_ok=True)
                    torch.save(self.model.state_dict(), 
                               os.path.join(self.save_path, f"checkpoint_{step+1}.pth"))

                self.temp = max(self.temp * np.exp(-self.temp_decay * step), self.min_temp)
                self.lr_scheduler.step()

                progress_bar.set_description(f"Training | Recon: {recon:.4f} | KL: {kl:.4f}")
                progress_bar.update(1)

                step += 1
                if step >= self.max_steps:
                    break

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "final.pth"))
        with open(os.path.join(self.save_path, "loss_log.json"), 'w') as f:
            json.dump({
                'step': list(range(len(total_losses))),
                'reconstruction_loss': recon_losses,
                'kl_loss': kl_losses,
                'total_loss': total_losses
            }, f)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


def main():
    config = get_config()
    trainer = CVAETrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()