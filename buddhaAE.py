import os, shutil
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from numpy import genfromtxt
np.set_printoptions(threshold=np.nan)


training = True
test_relu = True
model_name = 'buddha_head_rgb'

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        #encoder
        self.layer1 = nn.Linear(3, 128*3*2)
        self.layer2 = nn.Linear(128*3*2, 64*3*2)
        self.layer3 = nn.Linear(64*3*2, 32*3*2)
        self.layer4 = nn.Linear(32*3*2, 16*3*2)
        self.layer5 = nn.Linear(16*3*2,16*3)
        self.layer6 = nn.Linear(16*3,2)
        #decoder
        self.layer7 = nn.Linear(2, 16*3)
        self.layer8 = nn.Linear(16*3, 16*3*2)
        self.layer9 = nn.Linear(16*3*2, 32*3*2)
        self.layer10 = nn.Linear(32*3*2, 64*3*2)
        self.layer11 = nn.Linear(64*3*2,128*3*2)
        self.layer12 = nn.Linear(128*3*2,3)

    def forward(self, x):
        #encode
        x = self.layer1(x)
        s = torch.sign(x)
        x = F.relu(x)

        x = self.layer2(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer3(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer4(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer5(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer6(x)        
        z=x
        #decode
        x = self.layer7(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer8(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer9(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer10(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer11(x)
        s = torch.cat((s,torch.sign(x)),1)
        x = F.relu(x)

        x = self.layer12(x)
        return x, z, s

if training == True:
    num_epochs = 400
    batch_size = 256
    learning_rate = 8e-5

    dataset = genfromtxt(model_name + '.csv', delimiter=',')
    dataset_tensor = torch.from_numpy(dataset).float()
    dataloader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    """training"""
    min_tot_loss = 1e99
    for epoch in range(num_epochs):
        for data in dataloader:
            img = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            y, z, _ = model(img)
            loss = criterion(y, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4e}'
              .format(epoch + 1, num_epochs, loss.data[0]))

        if loss.data[0] < min_tot_loss:
            min_tot_loss = loss.data[0]
            torch.save(model, './' + model_name + '_autoencoder.pth')

model = torch.load('./' + model_name + '_autoencoder.pth')

if test_relu:
    #clear folder
    folder = 'results'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    #record activation 
    testdata = genfromtxt(model_name + '.csv', delimiter=',')    
    numX = testdata.shape[0]
    testdata_tensor = torch.from_numpy(testdata).float()
    testloader = DataLoader(testdata_tensor, batch_size=10000, shuffle=False)
    count = 1
    for data in testloader:
        x = data
        x = x.view(x.size(0), -1)
        x = Variable(x).cuda()
        y, z, s = model(x)
        y = y.data.cpu().numpy()
        z = z.data.cpu().numpy()
        s = s.data.cpu().numpy()
        np.savetxt('results/output_z_' + model_name + '_' + str(count).zfill(2) + '.csv', z, delimiter=",")
        np.savetxt('results/output_y_' + model_name + '_' + str(count).zfill(2) +  '.csv', y, delimiter=",")
        np.savetxt('results/output_relu_' + model_name + '_' + str(count).zfill(2) + '.csv', s, delimiter=",")
        count = count+1
