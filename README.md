# voBERTa

voBERTa is BERT-based model to classifying pages in the Voynich manuscript by their sections.


## Usage

```command
# creates processed texts files
python3 process-text.py

# runs the pretraining on the unsupervised voynich data
python3 pretrain.py

# runs the finetuning of the supervised voynich data
python3 fine-tuning.py

# generates the shapley values
python3 shapley-values.py
```
### Process Text

This process generates *.txt and *.csv files to assist in the pretraining and finetuning process.

### Pretraining

Pretraining further trains the BERT model on unsupervised Voynich manuscript data to allow the model to famaliarize itself with Voynich script.

### Finetuning

Finetuning trains the voBERTa model on labelled Voynich manuscript page data.

### Shapley Values
The generation of Shapley values allows us to determine which words most indicate a given span's section.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)