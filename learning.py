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
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=(3,3), stride=(1,1),padding = 1)
        self.conv8 = nn.Conv2d(2048, 4096, kernel_size=(3,3), stride=(1,1),padding = 1)
        self.conv9 = nn.Conv2d(4096, 8192, kernel_size=(3,3), stride=(1,1),padding = 1)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.relu = nn.ReLU()
    
    def forward(self, image): # 1024x512
        image = self.conv1(image) 
        image = self.pool(image) # 512x256
        image = self.relu(image)

        image = self.conv2(image) 
        image = self.pool(image) # 256x128
        image = self.relu(image)
        
        image = self.conv3(image)
        image = self.pool(image) # 128x64
        image = self.relu(image)

        image = self.conv4(image)
        image = self.pool(image) # 64x32
        image = self.relu(image)

        image = self.conv5(image)
        image = self.pool(image) # 32x16
        image = self.relu(image)

        image = self.conv6(image)
        image = self.pool(image) # 16x8
        image = self.relu(image)

        image = self.conv7(image)
        image = self.pool(image) # 8x4
        image = self.relu(image)
        
        image = self.conv8(image)
        image = self.pool(image) # 4x2
        image = self.relu(image)
        
        image = self.conv9(image)
        image = self.pool(image) #2x1
        image = self.relu(image)
        
        return image

class LSTMLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(8192*2, 1024)

    def forward(self, features):
        features_sequence = features.reshape(len(features), 1, 8192*2).expand(len(features), 40, 8192*2).clone()
        pre_tokens, _ = self.lstm(features_sequence) # h0 and c0 defaults to zero
        return pre_tokens

class Image2LaTeX(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvolutionLayer()  #(Nx) 1024x16x8
        self.lstm = LSTMLayer()         #(Nx) 40x1024
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 108)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, image):
        features = self.conv(image)
        tokens = self.lstm(features)

        tokens = self.linear1(tokens)
        tokens = self.relu(tokens)
        tokens = self.linear2(tokens)
        tokens = self.relu(tokens)    #(Nx) 40x108
        tokens = self.softmax(tokens)
        tokens = tokens.permute(0,2,1)
        return tokens

model = Image2LaTeX()
        
#%%

loss_function = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()            
    for batch_number, (image, tokens) in enumerate(dataloader):
        image = image.type(torch.float)
        pred = model(image)
        loss = loss_function(pred, tokens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_number % 10 == 0:
            loss, current = loss.item(), (batch_number + 1)*len(image)
            print(f"loss: {loss:>5f}, {current:>5f}/{size:>5f}")


def test(dataloader, model, loss_function):
  size = len(dataloader.dataset)
  number_batch = len(dataloader)
  model.eval()  # setup the model for evaluating. No tracking calculations
  test_loss, correct = 0, 0
  with torch.no_grad(): #  No tracking calculations
    for image, tokens in dataloader:
        image = image.type(torch.float)
        pred = model(image)
        test_loss += loss_function(pred, tokens).item()
        correct += (pred.amax(dim=2) == tokens).type(torch.float).sum().item()/40

  test_loss /= number_batch
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#%%

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n")
    train(train_dataloader, model, loss_function, optimizer)
    test(test_dataloader, model, loss_function, optimizer)
    print("Done!")
# %%
