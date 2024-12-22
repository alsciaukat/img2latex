# %%

import torch
from torch import nn, zeros, cat
from torch import optim
from torch.utils.data import DataLoader
from construct_dataset import TeXExpressionDataset

from lexicons import ALL_TOKENS

BATCH_SIZE = 32

train_dataset = TeXExpressionDataset("./generated/record.tsv", "./generated/")
test_dataset = TeXExpressionDataset("./generated/record.tsv", "./generated/", test=True)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



# %%

class ConvolutionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(3,3), stride=(1,1),padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1),padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1),padding = 1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1),padding = 1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1),padding = 1)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=(1,1),padding = 1)
        self.conv8 = nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=(1,1),padding = 1)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.relu = nn.ReLU()
    
    def forward(self, image): # 600x400
        image = self.conv1(image) 
        image = self.pool(image) # 300x200
        image = self.relu(image)

        image = self.conv2(image) 
        image = self.pool(image) # 150x100
        image = self.relu(image)
        
        image = self.conv3(image)
        image = self.pool(image) # 75x50
        image = self.relu(image)

        image = self.conv4(image)
        image = self.pool(image) # 37x25
        image = self.relu(image)

        image = self.conv5(image)
        image = self.pool(image) # 19x12
        image = self.relu(image)

        image = self.conv6(image)
        image = self.pool(image) # 9x6
        image = self.relu(image)

        image = self.conv7(image)
        image = self.pool(image) # 4x3
        image = self.relu(image)
        
        image = self.conv8(image)
        image = self.pool(image) # 2x1
        image = self.relu(image)
        
        return image


class RNNCellLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_cell1 = nn.RNNCell(input_size=1024*2, hidden_size=256)
        self.rnn_cell2 = nn.RNNCell(input_size=128, hidden_size=128)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 108)

        self.relu = nn.ReLU()
    
    def forward(self, features, h0, h1, eot):
        if eot.sum() == len(eot):
            return torch.tensor([0.]*107 + [1.]).repeat(len(eot), 1), h0, h1
        h0 = self.rnn_cell1(features, h0)
        hidden1 = self.linear1(h0)
        hidden1 = self.relu(hidden1)
        h1 = self.rnn_cell2(hidden1)
        output = self.linear2(h1)
        output = self.relu(output)
        passive_filter = (torch.ones(len(eot)) - eot).unsqueeze(1).kron(torch.ones(108))
        active_filter = eot.unsqueeze(1).kron(torch.tensor([0.]*107 + [1.])) # (Nx) 108
        output = output * active_filter + output * passive_filter
        return output, h0, h1  # (Nx) 108, (Nx) 256, (Nx) 128
        

class RNNLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn_cell = RNNCellLayer()

    def forward(self, features):
        features = features.reshape(len(features), 1024*2)
        h0 = torch.zeros(len(features), 256)
        h1 = torch.zeros(len(features), 128)
        eot = torch.zeros(len(features))

        tokens = torch.Tensor(torch.Size((len(features), 108, 0)))

        for i in range(40):
            output, hidden0, hidden1 = self.rnn_cell(features, h0, h1, eot)

            h0 = hidden0
            h1 = hidden1
            eot = (output.argmax(dim=1) == 107).type(torch.float)

            output = output.reshape(len(features), 108, 1)
            tokens = torch.cat((tokens, output), dim=2)

        return tokens # (Nx) 108x40




class Image2LaTeX(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvolutionLayer()  #(Nx) 1024x2x1
        self.rnn = RNNLayer()         #(Nx) 40x128
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image):
        features = self.conv(image)
        tokens = self.rnn(features)
        tokens = self.softmax(tokens)
        return tokens

        
#%%

model = Image2LaTeX()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-2)

def debug(dataloader, model):
    iterator = iter(dataloader)
    image, tokens = next(iterator)

    pred = model(image)
    print(pred)
    print(tokens)


def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()            
    for batch_number, (image, tokens) in enumerate(dataloader):
        pred = model(image)
        loss = loss_function(pred, tokens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_number % 20 == 0:
            loss, current = loss.item(), (batch_number + 1)*len(image)
            print(f"loss: {loss:>5f}, {current:>5f}/{size:>5f}")


def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    number_batch = len(dataloader)
    model.eval()  # setup the model for evaluating. No tracking calculations
    test_loss, correct = 0, 0
    with torch.no_grad(): #  No tracking calculations
        for image, tokens in dataloader:
            pred = model(image)
            test_loss += loss_function(pred, tokens).item()
            correct += (pred.argmax(dim=1) == tokens).type(torch.float).sum().item()/40
    
    print(pred.argmax(dim=1))


    test_loss /= number_batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#%%

# debug(test_dataloader, model)
epochs = 5
# model.load_state_dict(torch.load("./model_state", weights_only=True))
# optimizer.load_state_dict(torch.load("./optimizer_state", weights_only=True))

for t in range(epochs):
    print(f"Epoch {t+1}\n")
    train(train_dataloader, model, loss_function, optimizer)
    test(test_dataloader, model, loss_function)
    print("Done!")
torch.save(model.state_dict(), "./model_state")
torch.save(optimizer.state_dict(), "./optimizer_state")
# %%
