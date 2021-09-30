import torch
import torch.nn as nn
import torchtext.vocab
import numpy as np
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torch import optim
import torch.nn.functional as F


# Consists of class ids 1-4 where 1-World, 2-Sports, 3-Business, 4-Sci/Tech
train_data = AG_NEWS(root='.data', split='train')
test_data = AG_NEWS(root='.data', split='test')

train_data = to_map_style_dataset(train_data)
test_data = to_map_style_dataset(test_data)

tokenizer = get_tokenizer('basic_english')
vec = torchtext.vocab.GloVe(name='6B', dim=50)


def collate_batch(batch):
    pre_text = []
    labels = []
    pre_text = torch.tensor(pre_text)
    for label, text in batch:
        text = tokenizer(text)[:270]
        text += ["<pad>" for i in range(270 - len(text) if len(text) < 270 else 0)]
        pre_text = torch.cat([pre_text, vec.get_vecs_by_tokens(text).squeeze()], dim=0)
        labels.append(label-1)
    pre_text = pre_text.view(-1, 1, 270, 50)
    labels = torch.LongTensor(labels)

    return pre_text, labels


batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


# Model
class CNN(nn.Module):
    def __init__(self, batch, length, input_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.batch = batch
        self.conv3 = nn.Conv2d(1, 100, (3, input_size), bias=True)
        self.conv4 = nn.Conv2d(1, 100, (4, input_size), bias=True)
        self.conv5 = nn.Conv2d(1, 100, (5, input_size), bias=True)
        self.Max3_pool = nn.MaxPool2d((length - 3 + 1, 1))
        self.Max4_pool = nn.MaxPool2d((length - 4 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((length - 5 + 1, 1))
        self.linear1 = nn.Linear(300, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(self.batch, -1)
        x = self.dropout(x)
        x = self.linear1(x)

        return x


net = CNN(50, 270, 50)

# 학습
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-6)
acc = []

for epoch in range(300):
    losses = 0
    for (text, labels) in train_loader:
        predictions = net(text)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses = losses + loss.item()
    print(f'Epoch{epoch + 1},training loss: {losses/(len(train_data)/batch_size)}')


# Test
with torch.no_grad():
    num_correct = 0
    net.eval()
    for (text, labels) in test_loader:
        output = net(text)
        pred = torch.argmax(output, dim=1)
        correct_tensor = pred.eq(labels.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)
    test_acc = num_correct / (len(test_loader)*batch_size)
    print("Test accuracy: {:.3f}".format(test_acc))
