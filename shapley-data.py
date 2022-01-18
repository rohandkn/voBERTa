from voynich import VoynichManuscript
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torchtext.legacy.data import Field, TabularDataset, BucketIterator	
from sklearn.metrics import classification_report
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from sklearn.model_selection import train_test_split
from tqdm import trange
from sklearn.model_selection import KFold

def shap_get_sum(line_count, fold):
  shap_values_data = np.load("shap_values_data" + str(fold) + ".npy", allow_pickle=True)
  shap_values_values = np.load("shap_values_values" + str(fold) + ".npy", allow_pickle=True)
  shap_total_vals = np.array([{}, {}, {}, {}, {}, {}, {}])
  for i in range(line_count):
    curr_string = ""
    shap_weights = np.zeros(6)
    for j in range(len(shap_values_data[i])):
      if shap_values_data[i][j].strip(' ').isalpha():
        if shap_values_data[i][j][0] == ' ':
          if len(curr_string) > 0:
            for k in range(6):
              if curr_string in shap_total_vals[k]:
                shap_total_vals[k][curr_string] += np.abs(shap_weights[k])
                if k == 0:
                  shap_total_vals[6][curr_string] += 1
              else:
                shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
                if k == 0:
                  shap_total_vals[6][curr_string] = 1
          curr_string = shap_values_data[i][j][1:]
          shap_weights = np.abs(shap_values_values[i][j])
        else:
          curr_string += shap_values_data[i][j]
          shap_weights += np.abs(shap_values_values[i][j])
      else:
        if len(curr_string) > 0:
          for k in range(6):
            if curr_string in shap_total_vals[k]:
              shap_total_vals[k][curr_string] += np.abs(shap_weights[k])
              if k == 0:
                shap_total_vals[6][curr_string] += 1
            else:
              shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
              if k == 0:
                shap_total_vals[6][curr_string] = 1
        curr_string = ""
        shap_weights = np.zeros(6)
  np.save("shap_sum" + str(fold) + ".npy", shap_total_vals)
  print("Saved to " + "shap_sum" + str(fold) + ".npy")

def shap_get_max(line_count, fold):
  shap_values_data = np.load("shap_values_data" + str(fold) + ".npy", allow_pickle=True)
  shap_values_values = np.load("shap_values_values" + str(fold) + ".npy", allow_pickle=True)
  shap_total_vals = np.array([{}, {}, {}, {}, {}, {}])
  for i in range(line_count):
    curr_string = ""
    shap_weights = np.zeros(6)
    for j in range(len(shap_values_data[i])):
      if shap_values_data[i][j].strip(' ').isalpha():
        if shap_values_data[i][j][0] == ' ':
          if len(curr_string) > 0:
            for k in range(6):
              if curr_string in shap_total_vals[k]:
                if np.abs(shap_weights[k]) > shap_total_vals[k][curr_string]:
                  shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
              else:
                shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
          curr_string = shap_values_data[i][j][1:]
          shap_weights = np.abs(shap_values_values[i][j])
        else:
          curr_string += shap_values_data[i][j]
          shap_weights += np.abs(shap_values_values[i][j])
      else:
        if len(curr_string) > 0:
          for k in range(6):
            if curr_string in shap_total_vals[k]:
              if np.abs(shap_weights[k]) > shap_total_vals[k][curr_string]:
                shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
            else:
              shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
        curr_string = ""
        shap_weights = np.zeros(6)
  np.save("shap_max" + str(fold) + ".npy", shap_total_vals)
  print("Saved to " + "shap_max" + str(fold) + ".npy")
