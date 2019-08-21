# Interactive Machine Reading Comprehension
---------------------------------------------------------------------------
Xingdi Yuan, Jie Fu, Marc-Alexandre Cote, Yi Tay, Christopher Pal, Adam Trischler


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
