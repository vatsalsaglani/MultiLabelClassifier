# all import statements

import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import os
import zipfile
from tqdm import tqdm
from PIL import Image
from glob import glob
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader
