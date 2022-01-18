from voynich import VoynichManuscript
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torchtext.legacy.data import Field, TabularDataset, BucketIterator	
from sklearn.metrics import classification_report
import numpy as np
from transformers import *
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import trange

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def encode_with_truncation(examples):


    """Mapping function to tokenize the sentences passed with truncation"""

    print(type(examples))
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


def voBERTa():
	lines = []
	labelNums = {}
	labelCount = 0
	labels = []
	vm = VoynichManuscript("voynich-text.txt", inline_comments=False)
	for page in vm.pages:
		if vm.pages[page].section in labelNums:
			section_label = labelNums[vm.pages[page].section]
		else:
			section_label = labelCount
			labelNums[vm.pages[page].section] = labelCount
			labelCount += 1
		for line in vm.pages[page]:
			lines.append("[CLS]" + line.text.replace(".", " ") + "[SEP]")
			labels.append(section_label)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = load_dataset("csv", delimiter='/', data_files=["voynich-text-processed.csv"], split="train")
	
	d = dataset.train_test_split(test_size=0.1)

	# tokenizing the train dataset
	train_dataset = d["train"].map(encode_with_truncation, batched=True)
	# tokenizing the testing dataset
	test_dataset = d["test"].map(encode_with_truncation, batched=True)

	model = BertForSequenceClassification.from_pretrained("pretrained-bert", num_labels=6)
	model.to(device)

	training_args = TrainingArguments(
	    output_dir="finetuned-model",          # output directory to where save model checkpoint
	    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
	    overwrite_output_dir=True,      
	    num_train_epochs=100,            # number of training epochs, feel free to tweak
	    per_device_train_batch_size=3, # the training batch size, put it as high as your GPU memory fits
	    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
	    per_device_eval_batch_size=3,  # evaluation batch size
	    logging_steps=100,             # evaluate, log and save model checkpoints every 1000 step
	    save_steps=1000,
	)

	# initialize the trainer and pass everything to it
	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=train_dataset,
	    eval_dataset=test_dataset,
	)

	trainer.train()

voBERTa()