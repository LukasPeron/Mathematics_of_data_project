import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})

print("Packages loaded successfully!")

# Enable cudnn auto-tuner for potential speedup
torch.backends.cudnn.benchmark = True

def create_cnn_model():
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
            self.fc1 = nn.Linear(8 * 5 * 5, 8)
            self.fc2 = nn.Linear(8, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 8 * 5 * 5)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return CNN()

def plot_training_history(histories):
    # Group histories by color to later produce a single legend entry per color
    grouped_by_color = {}
    for h in histories:
        # Extract BS and LR from the label (e.g. 'BS=32, LR=0.0032')
        # and group them under the same color
        c = h['color']
        bs_lr_str = h['label'].replace("BS=", "").replace("LR=", "").strip()
        if c not in grouped_by_color:
            grouped_by_color[c] = []
        grouped_by_color[c].append(bs_lr_str)

    plt.figure(figsize=(15,12))

    # Plot without legend labels for each line
    for i, history in enumerate(histories):
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        train_accuracies = history['train_accuracies']
        val_accuracies = history['val_accuracies']
        color = history['color']
        epochs = len(train_losses)
        x = range(1, epochs + 1)

        plt.subplot(2, 2, 1)
        plt.plot(x, train_losses, color=color)

        plt.subplot(2, 2, 2)
        plt.plot(x, val_losses, color=color)

        plt.subplot(2, 2, 3)
        plt.plot(x, train_accuracies, color=color)

        plt.subplot(2, 2, 4)
        plt.plot(x, val_accuracies, color=color)

    # Add titles and axes labels
    plt.subplot(2, 2, 1)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 4)
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('adam_history.png')

def train_and_evaluate(batch_size, learning_rate, epochs, train_loader, val_loader, test_loader, device):
    model = create_cnn_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler()  # mixed-precision
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f"BS: {batch_size}, LR: {learning_rate}, Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total
    print(f"BS: {batch_size}, LR: {learning_rate}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def main(epochs=20):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(dataset)) #= 48000, len(dataset) = 60000
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Original configs
    configurations = [
        {'batch_size': 16, 'learning_rate': 0.0016},
        {'batch_size': 32, 'learning_rate': 0.0032},
        {'batch_size': 64, 'learning_rate': 0.0064},
        {'batch_size': 128, 'learning_rate': 0.0128},
        {'batch_size': 200, 'learning_rate': 0.02},
        {'batch_size': 256, 'learning_rate': 0.0256},
        {'batch_size': 400, 'learning_rate': 0.04},
        {'batch_size': 512, 'learning_rate': 0.0512},
        {'batch_size': 720, 'learning_rate': 0.072},
        {'batch_size': 1024, 'learning_rate': 0.1024}
    ]

    configs_extended = []
    for c in configurations:
        configs_extended.append({
            'batch_size': c['batch_size'],
            'learning_rate': c['learning_rate'],
            'ratio_multiplier': 1.0
        })
        configs_extended.append({
            'batch_size': c['batch_size'],
            'learning_rate': c['learning_rate'] * 5.0,
            'ratio_multiplier': 5.0
        })

    b_sizes = [c['batch_size'] for c in configurations]
    min_bs, max_bs = min(b_sizes), max(b_sizes)
    histories = []

    for conf in configs_extended:
        bs = conf['batch_size']
        lr = conf['learning_rate']
        ratio_mult = conf['ratio_multiplier']
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, pin_memory=True)

        out = train_and_evaluate(bs, lr, epochs, train_loader, val_loader, test_loader, device)
        out['label'] = f"BS={bs}, LR={lr:.4g}"
        cval = (bs - min_bs) / (max_bs - min_bs + 1e-9)
        if ratio_mult == 1.0:
            out['color'] = plt.cm.Blues(0.3 + 0.7*cval)
        else:
            out['color'] = plt.cm.Reds(0.3 + 0.7*cval)
        histories.append(out)

    plot_training_history(histories)

if __name__ == "__main__":
    main(epochs=20)
