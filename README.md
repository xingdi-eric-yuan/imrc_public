# Interactive Machine Reading Comprehension (iMRC)
---------------------------------------------------------------------------
Xingdi Yuan, Jie Fu, Marc-Alexandre Cote, Yi Tay, Christopher Pal, Adam Trischler

Existing machine reading comprehension (MRC) models do not scale effectively to real-world applications like web-level information retrieval and question answering (QA). We argue that this stems from the nature of MRC datasets: most of them are static environments wherein all the supporting documents and facts are fully observable. In this paper, we propose a simple method that reframes existing MRC datasets as interactive, partially observable environments. Specifically, we "occlude" the majority of a document's text and add context-sensitive commands that reveal ``glimpses'' of the hidden text to a model. We repurpose SQuAD and NewsQA as an initial case study, and then show how the interactive corpora can be used to train a model that seeks relevant information through sequential decision making. We believe that this kind of settings could pave the way to scaling models to web-scale QA scenarios.


## Dependencies

```
sudo apt update
conda create -p /tmp/imrc python=3.6 numpy scipy ipython matplotlib cython nltk pillow
source activate /tmp/imrc
pip install --upgrade pip
pip install numpy==1.16.4
pip install tqdm h5py pyyaml gym
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Pretrained Embeddings
download fasttext `crawl-300d-2M.vec.zip` from `https://fasttext.cc/docs/en/english-vectors.html`, unzip, and run `embedding2h5.py` for fast embedding loading

## Datasets
* We provide a split of the SQuAD dataset, because the official squad test data is hidden, so we split the training data to get an extra validation set, we use the official dev set as test set;
* Download NewsQA dataset and run their script to split and tokenize it.
