import torch
import torch.optim as optim
from torch import nn
import torch.nn.init as init
from torchvision import transforms as T
from torchvision import datasets
import os
from prepare_data import Aircraft
from tqdm import tqdm
from torch.utils.data import DataLoader

lr = 0.0001
is_load = True
is_save = True
is_draw_conf = True
num_epochs = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {%s}: 1_0 or 1_1 expected" %version)

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)

        return torch.flatten(x, 1)


class Train(nn.Module):

    def __init__(self):
        super().__init__()
        # Data
        print('==> Preparing data..')
        IMG_SIZE = 380

        train_transform = T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=(-15, 15)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        test_transform = T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.trainloader = Aircraft('./data/aircraft', train=True, download=False, transform=train_transform)
        self.testloader = Aircraft('./data/aircraft', train=False, download=False, transform=test_transform)

        self.ls_train_acc = []
        self.ls_test_acc = []

        # weights download from https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth
        print('==> Building model..')
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)

        # finetuning for FGVCAIRCRAFT
        # self.model.classifier[1] = torch.nn.Conv2d(512, 100, kernel_size=(1, 1), stride=(1, 1))
        self.model.classifier._modules["1"] = nn.Conv2d(512, 100, kernel_size=(1, 1))
        self.model.num_classes = 100
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        self.model = self.model.to(device)

        # self.criterion = torch.nn.NLLLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    # Training
    def train(self):
        self.model.train()
        for epoch in  range(num_epochs):
            epoch = epoch + 1
            print('\nEpoch | %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0
            train_dl = DataLoader(self.trainloader, batch_size=20,shuffle=True)
            tqdm_stream_train = tqdm(train_dl)
            for i, data in enumerate(tqdm_stream_train):
                inputs = data[0]
                targets = data[1]
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100.0 * (correct / total)
            print('Train Loss: %.3f | Acc: %.3f%%' % (train_loss/len(train_dl), train_acc))
            if (epoch % 5 == 0 and epoch != 0):
                self.ls_train_acc.append(train_acc)

            self.test(epoch)

            if (epoch % 20 == 0 and epoch != 0):
                self.scheduler.step()


    def test(self, epoch):
        global best_acc
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        true_label = []
        pred_label = []
        test_dl = DataLoader(self.testloader, batch_size=20, shuffle=False)
        with torch.no_grad():
            for inputs, targets in test_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if is_draw_conf:
                    true_label += targets.cpu().detach().tolist()
                    pred_label += predicted.cpu().detach().tolist()

        test_acc = 100.0 * (correct / total)
        print('Test Loss: %.3f | Acc: %.3f%%' % (test_loss/len(test_dl), test_acc))

        if (epoch % 5 == 0 and epoch != 0):
            self.ls_test_acc.append(test_acc)

        # Save checkpoint.
        if is_save:
            acc = 100.*correct/total
            if acc > best_acc:
                print('Saving..')
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')

                torch.save(self.model.state_dict(), f'./checkpoint/squeezenet_ckpt_acc_{acc}.pth')
                best_acc = acc

        self.model.train()
        if is_draw_conf: return true_label, pred_label


if __name__ == "__main__":
    t = Train()
    t.train()
    w=1
