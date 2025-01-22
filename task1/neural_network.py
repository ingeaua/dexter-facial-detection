import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


class HOGDataset(Dataset):
    def __init__(self, negative_file, positive_file):
        self.negative_data = np.load(negative_file)
        self.positive_data = np.load(positive_file)

        self.negative_labels = np.zeros(self.negative_data.shape[0], dtype=np.float32)
        self.positive_labels = np.ones(self.positive_data.shape[0], dtype=np.float32)

        self.data = np.concatenate([self.negative_data, self.positive_data], axis=0)
        self.labels = np.concatenate([self.negative_labels, self.positive_labels], axis=0)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class HOGClassifier(nn.Module):

    def __init__(self, input_length):
        super().__init__()
        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(input_length, 256)
        self.second_layer = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, 2)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        x = self.flatten(x)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = self.output_layer(x)
        return x

    def train_model(self, data_loader, loss_function=None, optimizer=None, NUM_EPOCHS=10):

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)

        self.to(self.device)

        self.train(True)
        for epoch in range(NUM_EPOCHS):

            epoch_loss = 0.0
            num_batches = 0

            for batch_index, (hog_batch, labels_batch) in enumerate(data_loader):
                hog_batch = hog_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                pred = self(hog_batch)

                loss = loss_function(pred, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            print(f"Average loss for epoch {epoch + 1}: {avg_epoch_loss:.4f}")

            # if (epoch + 1) % 3 == 0:
            #     self.test_model_on_train(data_loader)
            #     compute_scores_and_plot(self, data_loader)

    def test_model_on_train(self, data_loader, loss_function=None):

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        correct = 0.
        test_loss = 0.
        size = len(data_loader.dataset)
        self.to(self.device)
        self.eval()

        with torch.no_grad():
            for hog_batch, labels_batch in data_loader:
                hog_batch = hog_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                pred = self(hog_batch)
                test_loss += loss_function(pred, labels_batch).item()
                correct += (pred.argmax(1) == labels_batch).type(torch.float).sum().item()

        correct /= size
        test_loss /= size
        print(f"Accuracy: {(100 * correct):>0.1f}%, Loss: {test_loss:>8f} \n")

    def get_score(self, x):
        x = x.to(self.device)
        logits = self(x)
        score = logits[:, 1] - logits[:, 0]
        return score[0].item()


def get_model(negative_desc_file, positive_desc_file, batch_size=32, num_epochs=10):
    dataset = HOGDataset(negative_desc_file, positive_desc_file)
    input_length = dataset.data.shape[1]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HOGClassifier(input_length=input_length)
    print('Start training NN')
    start_time = timeit.default_timer()
    model.train_model(data_loader, NUM_EPOCHS=num_epochs)
    end_time = timeit.default_timer()
    print(f'Done training NN - {end_time - start_time} sec')

    # compute_scores_and_plot(model, data_loader)

    return model


def compute_scores_and_plot(model, data_loader):
    model.eval()
    scores_positive = []
    scores_negative = []

    with torch.no_grad():
        for hog_batch, labels_batch in data_loader:
            hog_batch = hog_batch.to(model.device)
            labels_batch = labels_batch.to(model.device)

            logits = model(hog_batch)

            scores = logits[:, 1] - logits[:, 0]

            for score, label in zip(scores.cpu().numpy(), labels_batch.cpu().numpy()):
                if label == 1:
                    scores_positive.append(score)
                else:
                    scores_negative.append(score)

    scores_positive.sort()
    scores_negative.sort()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores_positive)), scores_positive, label="Scoruri exemple pozitive")
    plt.plot(range(len(scores_negative)), scores_negative, label="Scoruri exemple negative")
    plt.axhline(0, color='orange', label="0")
    plt.title("Distribuția scorurilor clasificatorului pe exemplele de antrenare")
    plt.xlabel("Nr exemplu antrenare")
    plt.ylabel("Scor clasificator")
    plt.legend()
    plt.show()


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to {filepath}")


def load_model(filepath, input_length):
    model = HOGClassifier(input_length=input_length)
    state_dict = torch.load(filepath, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {filepath}")
    return model
