import torch
import torchaudio
from freefield1010dataset import FreeField1010Dataset
from cnn import CNNNetwork
from train import AUDIO_DIRECTORY, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES


def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        predicted = model(input)
        predicted = int(torch.round(predicted))
        expected = int(target)
    return predicted, expected



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device.")

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    # load bird audio dataset
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
    print(f"Dataset has {len(ffbirds)} samples.")
    # get a sample from the urban sound dataset for inference
    i = 0
    num_predictions = i + 5000
    num_of_errors = 0
    num_of_birds = 0
    num_of_bird_errors = 0


    print("Beginning test...")


    while i < num_predictions:
        input, target = ffbirds[i][0], ffbirds[i][1] # -[batch size, num_channels, fr, time]
        input.unsqueeze_(0)

        # make an inference
        predicted, expected = predict(cnn, input, target)
        #print(f"Predicted: '{predicted}', Expected: '{expected}'")
        if expected == 1:
            num_of_birds += 1

        if predicted != expected:
            num_of_errors += 1
            if expected == 1:
                num_of_bird_errors += 1
        i += 1

    accuracy = int((1.0 - (num_of_errors / i)) * 100)
    if num_of_birds != 0:
        bird_accuracy = 1.0 - (num_of_bird_errors / num_of_birds)
        bird_accuracy = int(bird_accuracy * 100)


    print(f"Test finished. {num_of_errors} errors out of {i} tests ({accuracy}% accuracy)")
    print(f"Total birds: {num_of_birds}. Errors with birds: {num_of_bird_errors} ({bird_accuracy}% accuracy)")