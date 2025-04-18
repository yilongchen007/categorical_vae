import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import re
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import numpy as np
import pandas as pd
import tqdm
import json
from torch.utils.data import DataLoader, TensorDataset

from .model import CategoricalVAE, Encoder, Decoder
from .utils import split_tensor_efficient, categorical_kl_divergence, get_config
from .plot import plot_decoder_prob_matrices


class CVAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        print(f"[INFO] Using device: {self.device}")

        self.data_path = config['data_path']
        self.dataset = config['dataset']
        self._prepare_data()
        self.train_loader = DataLoader(
            TensorDataset(self.train_data), batch_size=config['batch_size'], shuffle=True
        )

        # self.test_loader = DataLoader(
        #     TensorDataset(self.test_data), batch_size=config['batch_size'], shuffle=True
        # )

        self.save_path = os.path.join(config['save_path'], config['version'])
        self.plot_path = os.path.join(config['plot_path'], config['version'])

        
        edim = config['embedding_dim']
        hdim = config['hidden_dim']
        N = config['N']
        K = config['K']

        self.model = CategoricalVAE(
            Encoder(self.unique_counts, [edim]*4, [N, K], hdim, config['encoder_type']),
            Decoder(self.unique_counts, [N, K], hdim, config['decoder_type'])
        ).to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=config['initial_lr'])
        # self.optimizer = optim.SGD(self.model.parameters(), lr=config['initial_lr'], momentum=0.0)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config['lr_decay'])

        self.max_steps = config['max_steps']
        self.model_save_interval = config['model_save_interval']
        self.temp = self.init_temp = config['initial_temp']
        self.min_temp = config['min_temp']
        self.temp_decay = config['temp_decay']
        self.kl_weight = config['kl_weight']

        if config['cp_path']:
            if not os.path.exists(config['cp_path']):
                raise FileNotFoundError(f"CP initialization file not found at: {config['cp_path']}")
    
            with open(config['cp_path'], 'rb') as f:
                cp_tensor = pickle.load(f)
                self.initialize_decoder_logits_from_cp(cp_tensor)
        else:
            for layer in self.model.decoder.probability_matrices:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1)

        if config['pretrain_path']:
            if not os.path.exists(config['pretrain_path']):
                raise FileNotFoundError(f"Pretrained model file not found at: {config['pretrain_path']}")
            self.load_model(config['pretrain_path'])
    
        # self.analyze_decoder_logits_weights()

    def _prepare_data(self):

        if self.dataset == 'icews':

            self.data = np.load(self.data_path)
            train_data, test_data = split_tensor_efficient(self.data['Y'])

            self.train_data = self._build_tensor_from_array(train_data).to(self.device)
            self.test_data = self._build_tensor_from_array(test_data).to(self.device)

            self.unique_counts = [len(self.data['actors']), len(self.data['actors']), len(self.data['actions']), len(self.data['dates'])]

            print(f"[INFO] Prepared icews data: train={self.train_data.shape[0]}, test={self.test_data.shape[0]}")
        
        elif self.dataset == 'hamlet':

            self.unique_counts = [26, 26, 26, 26]

            # Example text file of a word list
            with open(self.data_path, 'r') as file:
                all_words = file.readlines()

            words = []
            for line in all_words:
                words.extend(re.findall(r'\b\w{4}\b', line))

            words = [word.lower() for word in words]

            chars = sorted(set("".join(words)))
            char_mapping = {chars[i]: i for i in range(len(chars))}

            self.train_data = torch.tensor([([char_mapping[char] for char in word]) for word in words]).to(self.device)

            print(f"[INFO] Prepared hamlet data: train={self.train_data.shape[0]}")

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")


    def _build_tensor_from_array(self, array):
        source, target, action, date = np.where(array > 0)
        counts = array[source, target, action, date]

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

        tensor = torch.tensor(df.values, dtype=torch.long)
        return tensor[torch.argsort(tensor[:, -1])]
    

    def initialize_decoder_logits_from_cp(self, cp_tensor, eps=1e-8):
        if not hasattr(self.model.decoder, "probability_matrices"):
            raise AttributeError("decoder does not have 'probability_matrices'")
        
        for i, layer in enumerate(self.model.decoder.probability_matrices):
            factor = cp_tensor.factors[i]  # shape: (num_categories, latent_dim)

            prob_matrix = factor / (factor.sum(axis=1, keepdims=True) + eps)
            logit_matrix = np.log(prob_matrix + eps)
            logit_tensor = torch.tensor(logit_matrix, dtype=torch.float32, device = self.device)

            assert logit_tensor.shape == layer.weight.shape, f"Shape mismatch: {logit_tensor.shape} vs {layer.weight.shape}"
            with torch.no_grad():
                layer.weight.copy_(logit_tensor)

        print(f"[INFO] Initialized decoder logits from CP factor")
    
    
    def train(self):

        self.model.train()
        recon_losses, kl_losses, total_losses = [], [], []
        step = 0
        progress_bar = tqdm.tqdm(total=self.max_steps, desc='Training')

        while step < self.max_steps:
            for (x_batch,) in self.train_loader:
                x_batch = x_batch.to(self.device)

                phi, x_prob = self.model(x_batch, temperature=self.temp)

                recon = sum([F.cross_entropy(x_prob[i], x_batch[:, i]) for i in range(x_batch.shape[1])]) / x_batch.shape[0]
                kl = torch.mean(torch.sum(categorical_kl_divergence(phi), dim=1))
                loss = recon + kl * self.kl_weight

                recon_losses.append(recon.item())
                kl_losses.append(kl.item())
                total_losses.append(recon.item()+kl.item())

                _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                loss.backward()

                # print(f"phi.grad std: {self.model._debug['phi'].grad.std().item():.6f}, z.grad std: {self.model._debug['z_given_x'].grad.std().item():.6f}")

                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} grad OK ✅, norm = {param.grad.norm().item():.4f}")
                #     else:
                #         pass
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (step + 1) % self.model_save_interval == 0:
                    os.makedirs(self.save_path, exist_ok=True)
                    torch.save(self.model.state_dict(), 
                               os.path.join(self.save_path, f"checkpoint_{step+1}.pth"))

                if step % 1000 == 1:
                    self.temp = np.maximum(self.init_temp*np.exp(-self.temp_decay*step), self.min_temp)
                    self.lr_scheduler.step() # should multiply learning rate by 0.9

                progress_bar.set_description(f"Training | Recon: {recon:.4f} | KL: {kl:.4f}")
                progress_bar.update(1)

                step += 1
                if step >= self.max_steps:
                    break

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{self.dataset}_final.pth"))
        with open(os.path.join(self.save_path, "loss_log.json"), 'w') as f:
            json.dump({
                'step': list(range(len(total_losses))),
                'reconstruction_loss': recon_losses,
                'kl_loss': kl_losses,
                'total_loss': total_losses
            }, f)

        # _ = self.evaluate_reconstruction()
        self.plot_prob_matrices()

    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{self.dataset}_saved.pth"))
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()

        print(f"[INFO] Initialized model from {path}")

    
    def evaluate_reconstruction(self):
        """
        Evaluate full reconstruction accuracy across all categorical variables
        and compute average cross entropy loss for each.

        Returns:
            dict: Overall accuracy and average cross-entropy loss per variable.
        """
        self.model.eval()
        correct = [0] * 4
        total = 0
        ce_loss = [0.0] * 4  # cross-entropy per variable
        loss_fn = CrossEntropyLoss(reduction='sum')  # sum over batch

        with torch.no_grad():
            for batch in self.train_loader:
                inputs = batch[0].to(self.device)  # shape: [B, 4]
                true_vals = [inputs[:, i] for i in range(4)]  # ground truth
                batch_size = inputs.size(0)

                _, x_hat_logits = self.model(inputs)  # list of [B, K_i] logits

                for i in range(4):  # loop over variables
                    preds = torch.argmax(x_hat_logits[i], dim=-1)
                    correct[i] += (preds == true_vals[i]).sum().item()

                    ce_loss[i] += loss_fn(x_hat_logits[i], true_vals[i]).item()

                total += batch_size

        accuracy = [c / total for c in correct]
        avg_ce = [l / total for l in ce_loss]

        for i, var in enumerate(['source', 'target', 'action', 'date']):
            print(f"[{var}] Acc: {accuracy[i]:.4f} | CE Loss: {avg_ce[i]:.4f}")

        overall_acc = sum(correct) / (total * 4)
        overall_ce = sum(ce_loss) / (total * 4)

        print(f"[Overall] Acc: {overall_acc:.4f} | Avg CE Loss: {overall_ce:.4f}")

        return {
            "accuracy": accuracy,
            "cross_entropy": avg_ce,
            "overall_accuracy": overall_acc,
            "overall_cross_entropy": overall_ce
        }
    

    def plot_prob_matrices(self):

        if self.dataset == 'icews':

            actors, actions, dates = (self.data[key] for key in [ 'actors', 'actions', 'dates'])

            plot_decoder_prob_matrices(self.model, actors, actions, dates, self.plot_path)



def main():
    config = get_config()
    trainer = CVAETrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()