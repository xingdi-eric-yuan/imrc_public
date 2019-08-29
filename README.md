# Interactive Machine Comprehension with Information Seeking Agents (iMRC)
---------------------------------------------------------------------------
Code for paper "Interactive Machine Comprehension with Information Seeking Agents".

## To Install Dependencies
```
sudo apt update
conda create -p ~/venvs/imrc python=3.6 numpy scipy ipython matplotlib cython nltk pillow
source activate ~/venvs/imrc
pip install --upgrade pip
pip install numpy==1.16.4
pip install tqdm h5py pyyaml gym visdom
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Pretrained Word Embeddings
Before first time running it, download fasttext crawl-300d-2M.vec.zip from [HERE](https://fasttext.cc/docs/en/english-vectors.html), unzip, and run [embedding2h5.py](./embedding2h5.py) for fast embedding loading in the future.

## Datasets
* We provide a split of the SQuAD dataset, because the official squad test data is hidden, so we split the training data to get an extra validation set, we use the official dev set as test set;
* Download NewsQA dataset and run their script to split and tokenize it.

## To Play
We provide a simple interactive demo, one can feel what an interactive MRC data looks like.

```
python play.py
```

## To Train
```
python main.py
```

## Citation

Please use the following bibtex entry:
```
@article{yuan2019imrc,
  title={Interactive Machine Comprehension with Information Seeking Agents},
  author={Yuan, Xingdi and Fu, Jie and C\^ot\'{e}, Marc-Alexandre and Tay, Yi and Pal, Christopher and Trischler, Adam},
  journal={CoRR},
  volume={abs/1908.10449},
  year= {2019},
  archivePrefix={arXiv},
  eprint={1908.10449}
}
```

## License

[MIT](./LICENSE)

