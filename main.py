import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.models as models
import torchvision.datasets as datasets

def set_parameters_requires_grad(model, extracting):
    if extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, data_loaders, criterion, optimizer, epochs=30):
    for epoch in range(epochs):
        print("Epoch %d / %d" % (epoch, epochs-1))
        print("-" * 15)

        for phase in ["Training", "Testing"]:
            if phase == "Training":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="Training"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "Training":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = correct.double() / len(data_loaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:4f}".format(phase,
                                                       epoch_loss, epoch_acc))

if __name__ == '__main__':
    data_dir = "final-project_data"

    image_transforms = {
        "Training": transforms.Compose([transforms.RandomRotation((-270, 270)),
                 transforms.Resize((224,224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]),
        "Testing": transforms.Compose([transforms.RandomRotation((-270, 270)),
                 transforms.Resize((224,224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])}

    data_generator = {k: datasets.ImageFolder(os.path.join(data_dir, k),
                      image_transforms[k]) for k in ["Training", "Testing"]}
    data_loader = {k: torch.utils.data.DataLoader(data_generator[k], batch_size=8,
                   shuffle=True, num_workers=4) for k in ["Training", "Testing"]}

    device = torch.device("cpu")
    model = models.resnet18(weights=True)

    set_parameters_requires_grad(model, True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    update_params = []
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            update_params.append(param)
            print("\t", name)

    train_model(model, data_loader, loss, optimizer)