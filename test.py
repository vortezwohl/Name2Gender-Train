import torch
from deeplotx import LogisticRegression
from datasets import load_dataset

from name2gender.model import ENCODER

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
    return ENCODER.encode(text)


model_name = input('Model name: ')
model = LogisticRegression(input_dim=768, output_dim=1, num_heads=6, num_layers=2, head_layers=1, expansion_factor=1.5).load(model_name, model_dir='checkpoint')
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

acc_count = .0
model.eval()
with torch.no_grad():
    for i, name in enumerate(test_names):
        if round(model(encode(name, train=False)).item()) == test_genders[i]:
            acc_count += 1.
acc = 100 * (acc_count / len(test_names))
print(f'Accuracy: {acc:.4f}%', flush=True)
