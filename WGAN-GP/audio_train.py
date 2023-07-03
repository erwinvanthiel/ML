"""
Training of WGAN-GP

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from audio_gan import Discriminator, Generator, initialize_weights
from PIL import Image
import matplotlib.pyplot as plt
from audio_dataset import AudioDataset, invert_spectrogram
import scipy.signal
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wavf

def unstack_complex(complex_matrix):
    real_matrix = complex_matrix[0]
    imaginary_matrix = complex_matrix[1]

    # Combine the real and imaginary parts to obtain a single complex matrix
    combined_matrix = real_matrix + 1j * imaginary_matrix
    return combined_matrix

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SPECTRUM_SIZE = (128,32)
Z_DIM = 100
NUM_EPOCHS = 100
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

dataset = AudioDataset("C:\\Users\\evanthiel\\Desktop\\ML\\WGAN-GP\\data\\")

# comment mnist above and uncomment below for training on CelebA
# dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(dataset, batch_size=32)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM).to(device)
critic = Discriminator().to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
fixed_noise = torch.randn(1, Z_DIM).to(device)
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, real in enumerate(loader):
        print(batch_idx)
        real = real.to(device).unsqueeze(1)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            step += 1
            with torch.no_grad():
                output = gen(fixed_noise).squeeze(0)
                print(real.shape)
                print(output.shape)
                audio_array = invert_spectrogram(output.squeeze(0))

                # Specify the desired output file path
                output_file_audio = "fake_audio\\fake_audio{0}.wav".format(step)
                output_file_spec = "fake_audio\\fake_spec{0}.jpg".format(step)

                # save spectrogram
                plt.figure()
                plt.imshow(output.squeeze())
                plt.savefig(output_file_spec)

                # Save the waveform as a WAV file
                sf.write(output_file_audio, audio_array, 16000, format='WAV')



# print(real.shape)
# audio_reconstructed = invert_spectrogram(real[0])

# # Play the reconstructed audio
# sd.play(audio_reconstructed, 16000)

