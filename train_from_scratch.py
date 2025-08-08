import os.path
import pickle
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from vortezwohl.cache import LRUCache
from deeplotx.util import sha256
from deeplotx import LogisticRegression

from name2gender.model import ENCODER

cache = LRUCache(capacity=65536)
# base
model = LogisticRegression(input_dim=768, output_dim=1, num_heads=12, num_layers=4, head_layers=1, expansion_factor=2, model_name='name2gender-base', dtype=torch.float32)
# small
# model = LogisticRegression(input_dim=768, output_dim=1, num_heads=6, num_layers=2, head_layers=1, expansion_factor=1.5, model_name='name2gender-small', dtype=torch.float32)
print(model)


ds = load_dataset("erickrribeiro/gender-by-name")
train_dataset, test_dataset = ds['train'], ds['test']

# 0 for female 1 for male
train_names = train_dataset.to_dict()['Name']
train_genders = train_dataset.to_dict()['Gender']
test_names = test_dataset.to_dict()['Name']
test_genders = test_dataset.to_dict()['Gender']

male_names, female_names = [], []
for i, name in enumerate(train_names):
    if train_genders[i] == 0:
        female_names.append(name)
    else:
        male_names.append(name)

train_encoding_count = 0
test_encoding_count = 0


def encode(text: str, train: bool = True):
    global train_encoding_count, test_encoding_count
    if train:
        train_encoding_count += 1
        print(f'\rEncoding ({train_encoding_count}/{len(train_names)}):', text, end='' * 32, flush=True)
    else:
        test_encoding_count += 1
        print(f'\rEncoding ({test_encoding_count}/{len(test_names)}):', text, end='' * 32, flush=True)
    _hash = sha256(text)
    if _hash in cache:
        return cache[_hash]
    cache[_hash] = ENCODER.encode(text)
    return cache[_hash]


os.makedirs('./data', exist_ok=True)
if os.path.exists('./data/train_loader.pkl') and os.path.exists('./data/valid_loader.pkl'):
    with open('./data/train_loader.pkl', 'rb') as f:
        train_dataset_loader = pickle.load(f)
    with open('./data/valid_loader.pkl', 'rb') as f:
        valid_dataset_loader = pickle.load(f)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ratio = 0.9
    batch_size = 8
    all_texts = male_names + female_names
    text_embeddings = [encode(x) for x in all_texts]
    feature_dim = text_embeddings[0].shape[-1]
    dtype = text_embeddings[0].dtype
    labels = ([torch.tensor([1.], dtype=dtype, device=device) for _ in range(len(male_names))]
              + [torch.tensor([.0], dtype=dtype, device=device) for _ in range(len(female_names))])
    inputs = torch.stack(text_embeddings).to(device)
    labels = torch.stack(labels).to(device)
    dataset_size = len(labels)
    train_size = int(train_ratio * dataset_size)
    train_dataset = TensorDataset(inputs[:train_size], labels[:train_size])
    valid_dataset = TensorDataset(inputs[train_size:], labels[train_size:])
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    with open('./data/train_loader.pkl', 'wb') as f:
        pickle.dump(train_dataset_loader, f)
    with open('./data/valid_loader.pkl', 'wb') as f:
        pickle.dump(valid_dataset_loader, f)

train_step = 0
valid_step = 0
acc_train_loss = 0.
acc_valid_loss = 0.
eval_interval = 10000
log_interval = 200


writer = SummaryWriter()
alpha = 1e-3
rho = 0.25
num_epochs = 500
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-7)

try:
    for epoch in range(num_epochs):
        model.train()
        for batch_texts, batch_labels in train_dataset_loader:
            outputs = model.forward(batch_texts)
            loss = loss_function(outputs, batch_labels) + model.elastic_net(alpha=alpha, rho=rho)
            acc_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_step % log_interval == 0 and train_step > 0:
                writer.add_scalar('train/loss', acc_train_loss / log_interval, train_step)
                print(f'- Train Step {train_step} Loss {acc_train_loss / log_interval}', flush=True)
                acc_train_loss = 0.
            if train_step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    for batch_texts, batch_labels in valid_dataset_loader:
                        outputs = model.forward(batch_texts)
                        loss = loss_function(outputs, batch_labels)
                        acc_valid_loss += loss.item()
                        if valid_step % len(valid_dataset_loader) == 0 and valid_step > 0:
                            writer.add_scalar('valid/loss', acc_valid_loss / len(valid_dataset_loader), valid_step)
                            print(f'- Valid Step {valid_step} Loss {acc_valid_loss / len(valid_dataset_loader)}', flush=True)
                            acc_valid_loss = 0.
                            # eval and save
                            acc_count = .0
                            model.eval()
                            with torch.no_grad():
                                for i, name in enumerate(test_names):
                                    if round(model(encode(name, train=False)).item()) == test_genders[i]:
                                        acc_count += 1.
                            acc = 100 * (acc_count / len(test_names))
                            print(f'Accuracy: {acc:.4f}%', flush=True)
                            model.save(f'{int(time.time())}-ACC={acc:.2f}-{model_name}', model_dir='checkpoint')
                        valid_step += 1
                model.train()
            train_step += 1
except KeyboardInterrupt:
    acc_count = .0
    model.eval()
    with torch.no_grad():
        for i, name in enumerate(test_names):
            if round(model(encode(name, train=False)).item()) == test_genders[i]:
                acc_count += 1.
    acc = 100 * (acc_count / len(test_names))
    print(f'Accuracy: {acc:.4f}%', flush=True)
    model.save(f'{int(time.time())}-ACC={acc:.2f}-{model_name}', model_dir='checkpoint')
