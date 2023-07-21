# Code for PCL

Code for "Learning to Learn from Corrupted Data for Few-Shot Learning" in IJCAI 2023. 

If you use the code in this repo for your work, please cite the following bib entries:

```
@inproceedings{An2023PCL,
  author    = {Yuexuan An and
               Xingyu Zhao and
               Hui Xue},
  title     = {Learning to Learn from Corrupted Data for Few-Shot Learning},
  booktitle = {Proceedings of the 32nd International Joint Conference on Artificial Intelligence, {IJCAI} 2023, Macao, China, 19-25 August 2023},
  pages     = {},
  year      = {2023},
}
```

## Requirements

- Python >= 3.6
- PyTorch (GPU version) >= 1.5
- NumPy >= 1.13.3
- Scikit-learn >= 0.20

## Getting started

- Change directory to `./filelists/cifar`
- Download [CIFAR-FS](https://drive.google.com/file/d/1i4atwczSI9NormW5SynaHa1iVN1IaOcs/view)
- run `python make.py` in the terminal

## Running the scripts

To train and test the PCL model in the terminal, use:

```bash
$ python run_PCL.py --dataset cifar --algorithm PCL_matchingnet --tao 2.0 --noise_type feature --noise_rate 0.2 --train_n_way 5 --test_n_way 5 --n_shot 5 --model_name Conv4 --device cuda:0
```

## Acknowledgment

Our project references the codes and datasets in the following repo and papers.

[CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)

Oriol Vinyals, Charles Blundell, Tim Lillicrap, Koray Kavukcuoglu, Daan Wierstra. Matching Networks for One Shot Learning. NIPS 2016: 3630-3638.
