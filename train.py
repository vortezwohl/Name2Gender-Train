import pickle
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from name2gender.model import ENCODER, load_model

model_name = 'name2gender-small'
model = load_model(model_name, dtype=torch.float32)
print(model)


def encode(text: str):
    print('\rEncoding:', text, end='')
    return ENCODER.encode(text)


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ratio = 0.85
batch_size = 32
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
eval_interval = 2000
log_interval = 200
valid_log_interval = 50


writer = SummaryWriter()
alpha = 1e-4
rho = 0.2
num_epochs = 500
train_loss_threshold = .0
valid_loss_threshold = .0
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-6)

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
            train_step += 1
        if train_step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                for batch_texts, batch_labels in valid_dataset_loader:
                    outputs = model.forward(batch_texts)
                    loss = loss_function(outputs, batch_labels)
                    acc_valid_loss += loss.item()
                    if valid_step % valid_log_interval == 0 and valid_step > 0:
                        writer.add_scalar('valid/loss', acc_valid_loss / valid_log_interval, valid_step)
                        print(f'- Valid Step {valid_step} Loss {acc_valid_loss / valid_log_interval}', flush=True)
                        acc_valid_loss = 0.
                    valid_step += 1
            model.train()
except KeyboardInterrupt:
    model.save(f'{int(time.time())}-{model_name}', model_dir='checkpoint')
    acc_count = .0
    model.eval()
    with torch.no_grad():
        for i, name in enumerate(test_names):
            if round(model(encode(name)).item()) == test_genders[i]:
                acc_count += 1.
    print(f'Accuracy: {acc_count / len(test_names):.4f}%', flush=True)
