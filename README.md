# DNA-Inception
One of the hallmarks of cancer is somatic aberrations in the genomes and transcriptomes of malignant tumors, this information offers a wealth of distinguishable tumor-specific events, that do not occur in normal tissue of the same individual. Here, we propose DNA-Inception, a convolutional neural network-based model, it follows a sequence-based approach to classify AS events specific to tumor tissues. DNA-Inception takes as an input DNA or RNA sequences of AS events and their corresponding labels, i.e., tumor-specific `1` or tissue-specific `0`. The following figure shows the architecture of our model:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<img src="https://github.com/nec-research/DNA-Inception/blob/main/figures/dnaincep_archi_font1_color2_round.png" width="648" height="312" alt="DNA-Inception" class="center"/>

Using this software
-------------------

## Installation<a name="installation"></a>

```shell
git clone https://github.com/nec-research/DNA-Inception.git
```

You can train the model using:

```shell
python ./src/train_model.py --help
```

An example input data format is provided in `./json`

### Training and test datasets are available from: ###
* Kim, Pora, et al. "ExonSkipDB: functional annotation of exon skipping event in human." Nucleic acids research 48.D1 (2020): D896-D907.
* Kahles, André, et al. "Comprehensive analysis of alternative splicing across tumors from 8,705 patients." Cancer cell 34.2 (2018): 211-224.
