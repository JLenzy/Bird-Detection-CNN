import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from freefield1010dataset import FreeField1010Dataset
from cnn import CNNNetwork

BATCH_SIZE = 400
EPOCHS = 50
LEARNING_RATE = 0.002

ANNOTATIONS_FILE = "/Users/jlenz/Desktop/Datasets/BirdAudioDetection/metadata.csv"
AUDIO_DIRECTORY = "/Users/jlenz/Desktop/Datasets/BirdAudioDetection/wav"
SAMPLE_RATE = 22050
NUM_SAMPLES = SAMPLE_RATE * 1

MODEL_NAME = "cnn.pth"

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # make a prediction
        prediction = model(input)
        prediction = torch.flatten(prediction)
        target = target.to(torch.float32)

        # calculate loss
        loss = loss_fn(prediction, target)

        #backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("--------------------")
    print("Finished training!")

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device.")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    ffbirds = FreeField1010Dataset(ANNOTATIONS_FILE,
                            AUDIO_DIRECTORY,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    train_dataloader = create_data_loader(ffbirds, BATCH_SIZE)

    # build the model
    cnn = CNNNetwork().to(device)
    print(cnn)

    # instantiate loss function + optimiser
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    #save
    torch.save(cnn.state_dict(), MODEL_NAME)
    print(f"Model trained and stored at {MODEL_NAME}")


