# DNA-Inception
One of the hallmarks of cancer is somatic aberrations in the genomes and transcriptomes of malignant tumors, and this information offers a wealth of distinguishable tumor-specific events, that do not occur in normal tissue of the same individual. Here, we propose DNA-Inception, a convolutional neural network-based model, it follows a sequence-based approach to classify AS events specific to tumor tissues. DNA-Inception takes as an input DNA or RNA sequences of AS events and their corresponding labels based on whether each event occurs in tumor or normal tissue.


Using this software
-------------------

## Installation<a name="installation"></a>

```shell
git clone https://github.com/nec-research/DNA-Inception.git
```

You can train the model using:

```shell
python3 ./src/train_model.py --help
```


### Training and test datasets: ###
* Kim, Pora, et al. "ExonSkipDB: functional annotation of exon skipping event in human." Nucleic acids research 48.D1 (2020): D896-D907.
* Kahles, André, et al. "Comprehensive analysis of alternative splicing across tumors from 8,705 patients." Cancer cell 34.2 (2018): 211-224.
