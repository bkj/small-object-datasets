from tqdm import tqdm

from torchvision import datasets
from torchvision import transforms as T

from matplotlib import pyplot as plt
from rcode import *

from sod_wrapper import SODWrapper

# obs_dataset = datasets.Omniglot(root='./data', background=True, download=True, transform=T.ToTensor())
obs_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
sod_dataset = SODWrapper(obs_dataset, obs_per_frame=5)
# _ = plt.imshow(sod_dataset[0][0].numpy().transpose(1, 2, 0), cmap='gray')
# show_plot()

# for _ in tqdm(obs_dataset):
#   pass

for _ in tqdm(sod_dataset):
  pass
