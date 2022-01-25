# %%
import time
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch
from torchtext.datasets import AG_NEWS

# %%
train_iter = AG_NEWS(root='../../dat/pyt', split='train')

# %%
test_iter = AG_NEWS(root='../../dat/pyt', split='test')

# %%
next(train_iter)

# %%
next(train_iter)

# %%
next(train_iter)

# %%

# %%
tokenizer = get_tokenizer('basic_english')

# %%


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# %%
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# %%
len(vocab)

# %%
vocab(['here', 'is', 'an', 'example'])

# %%


def text_pipeline(x): return vocab(tokenizer(x))
def label_pipeline(x): return int(x) - 1


# %%
xs = text_pipeline('here is an example')

# %%
label_pipeline('9')

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%


def collate_batch(batch):
    ys, xs, offsets = [], [], [0]
    for (y, x) in batch:
        ys.append(label_pipeline(y))
        ts = torch.tensor(text_pipeline(x), dtype=torch.int64)
        xs.append(ts)
        offsets.append(ts.size(0))
    ys = torch.tensor(ys, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    xs = torch.cat(xs)
    return ys.to(device), xs.to(device), offsets.to(device)


# %%
train_iter = AG_NEWS(root='../../dat/pyt', split='train')
training_data_loader = DataLoader(train_iter, batch_size=32, shuffle=False, collate_fn=collate_batch)

# %%
len(training_data_loader)

# %%
len(training_data_loader.dataset)

# %%


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.linear = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.linear.weight.data.uniform_(-0.5, 0.5)
        self.linear.bias.data.zero_()

    def forward(self, text, offsets):
        # compute mean vectors of all words in the text
        embeded = self.embedding(text, offsets)
        return self.linear(embeded)


# %%
num_class = len(set([label for (label, _) in train_iter]))

# %%
print(num_class)

# %%
train_iter = AG_NEWS(root='../../dat/pyt', split='train')
next(train_iter)

# %%
vocab_size = len(vocab)

# %%
model = NeuralNetwork(vocab_size, 64, num_class).to(device)

# %%
model

# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# %%
criterion = nn.CrossEntropyLoss()

# %%

# %%

def train(dataloader, model, criterion, optimizer):
    N = len(dataloader.dataset)
    model.train()
    start_time = time.time()
    for batch, (ys, xs, offsets) in enumerate(dataloader):
        zs = model(xs, offsets)
        loss = criterion(zs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 500 == 0:
            elapsed = time.time() - start_time
            loss, current = loss.item(), batch * len(xs)
            print(f"loss: {loss:>7f} [{current:>7d}/{N:>6d}], elapsed: {elapsed:>4f}")

# %%

def test(dataloader, model, criterion):
    N = len(dataloader.dataset)
    num_batch = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (ys, xs, offsets) in dataloader:
            zs = model(xs, offsets)
            test_loss += criterion(zs, ys).item()
            correct += (zs.argmax(1) == ys).type(torch.float).sum().item()
    test_loss /= num_batch
    correct /= N
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

#
epochs = 40
for t in range(epochs):
    train_iter = AG_NEWS(root='../../dat/pyt', split='train')
    training_data_loader = DataLoader(train_iter, batch_size=32, shuffle=False, collate_fn=collate_batch)
    print(f"Epoch {t+1}\n-------")
    train(training_data_loader, model, criterion, optimizer)

    train_iter = AG_NEWS(root='../../dat/pyt', split='train')
    training_data_loader = DataLoader(train_iter, batch_size=32, shuffle=False, collate_fn=collate_batch)
    test(training_data_loader, model, criterion)

    test_iter = AG_NEWS(root='../../dat/pyt', split='test')
    test_data_loader = DataLoader(test_iter, batch_size=32, shuffle=False, collate_fn=collate_batch)
    test(test_data_loader, model, criterion)
print("Done.")
