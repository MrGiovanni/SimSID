<h1 align="center">SimSID: Unsupervised Anomaly Detection in Chest Radiography</h1>

SimSID is an unsupervised anomaly detection model for chest X-ray images. SimSID learns the
recurrent anatomical patterns that normal chest radiographs share, then flags a test image as
anomalous when its patterns do not fit — using **no anomaly labels during training**.

Anomaly detection in radiography is both easier and harder than in photographic images. It is
easier because radiography is spatially structured: consistent imaging protocols mean the same
anatomy lands in roughly the same place every time. It is harder because the anomalies are
subtle, and annotating them takes medical expertise. SimSID exploits the first fact to work
around the second.

This task is described in the literature as unsupervised anomaly detection, out-of-distribution
detection, one-class learning, novelty detection, and normality modeling. SimSID applies it to
two-dimensional chest radiography.

> [!NOTE]
> **SimSID is the journal extension of SQUID.** SQUID
> ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xiang_SQUID_Deep_Feature_In-Painting_for_Unsupervised_Anomaly_Detection_CVPR_2023_paper.html),
> code at [tiangexiang/SQUID](https://github.com/tiangexiang/SQUID)) introduced the method.
> SimSID (IEEE TPAMI 2024) is the significant technical improvement over it, and is the version
> to use. If you are looking for the CVPR paper's code, it is in the SQUID repository; this
> repository supersedes it.

<div align="center">

![SimSID overview](document/fig_introductory.png)

</div>

## Results

SimSID formulates anomaly detection as an image reconstruction task, using a **space-aware
memory matrix** and an **in-painting block in feature space**. During training it taxonomizes
the ingrained anatomical structures into recurrent visual patterns; at inference, patterns it
has not seen read as anomalies.

Against the previous state of the art in unsupervised anomaly detection, SimSID improves AUC by:

| benchmark | modality | SimSID AUC gain over prior state of the art |
|:---|:---|:---:|
| ZhangLab Chest X-ray | chest radiography | **+8.0%** |
| COVIDx | chest radiography | **+5.0%** |
| Stanford CheXpert | chest radiography | **+9.9%** |

The earlier SQUID model surpassed 13 state-of-the-art unsupervised anomaly detection methods by
at least 5 AUC points on two chest X-ray benchmarks. SimSID improves on SQUID.

<!-- TODO: add absolute AUC, accuracy, F1 and specificity per benchmark from Table 2 of the
     TPAMI paper, so the numbers can be quoted without reading the PDF. -->

## Paper

**Exploiting Structural Consistency of Chest Anatomy for Unsupervised Anomaly Detection in Radiography Images** <br/>
[Tiange Xiang](https://tiangexiang.github.io/)<sup>1</sup>, [Yixiao Zhang](https://scholar.google.com/citations?user=lU3wroMAAAAJ&hl=en)<sup>2</sup>, [Yongyi Lu](https://scholar.google.com/citations?user=rIJ99V4AAAAJ&hl=en&oi=ao)<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, [Chaoyi Zhang](https://chaoyivision.github.io/)<sup>1</sup>, [Weidong Cai](https://weidong-tom-cai.github.io/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com)<sup>2</sup> <br/>
<sup>1</sup>University of Sydney,  <sup>2</sup>Johns Hopkins University <br/>
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 46, no. 9, pp. 6070–6081, 2024 <br/>
[doi:10.1109/TPAMI.2024.3382009](https://doi.org/10.1109/TPAMI.2024.3382009) | [arXiv:2403.08689](https://arxiv.org/abs/2403.08689) | [paper](https://www.cs.jhu.edu/~alanlab/Pubs24/xiang2024exploiting.pdf)

**SQUID: Deep Feature In-Painting for Unsupervised Anomaly Detection** <br/>
[Tiange Xiang](https://tiangexiang.github.io/)<sup>1</sup>, [Yixiao Zhang](https://scholar.google.com/citations?user=lU3wroMAAAAJ&hl=en)<sup>2</sup>, [Yongyi Lu](https://scholar.google.com/citations?user=rIJ99V4AAAAJ&hl=en&oi=ao)<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, [Chaoyi Zhang](https://chaoyivision.github.io/)<sup>1</sup>, [Weidong Cai](https://weidong-tom-cai.github.io/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com)<sup>2</sup> <br/>
<sup>1</sup>University of Sydney,  <sup>2</sup>Johns Hopkins University <br/>
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 23890–23901 <br/>
[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Xiang_SQUID_Deep_Feature_In-Painting_for_Unsupervised_Anomaly_Detection_CVPR_2023_paper.html) | [arXiv:2111.13495](https://arxiv.org/abs/2111.13495) | [code](https://github.com/tiangexiang/SQUID)

## Installation

SimSID needs PyTorch, a CUDA-capable GPU, and a small set of standard scientific packages.

```bash
git clone https://github.com/MrGiovanni/SimSID.git
cd SimSID
conda create -n simsid python=3.10 -y
conda activate simsid
pip install -r requirements.txt
```

<details>
<summary>Reproducing the original environment (Python 3.6, CUDA 10.0)</summary>

`environment.yml` pins the exact 2020-era stack the paper was developed on: Python 3.6.10,
CUDA 10.0, torchvision 0.5.0. Python 3.6 reached end of life in December 2021 and CUDA 10.0
does not support GPUs newer than Turing, so this will not resolve on most current machines.
It is kept for the record.

```bash
conda env create -f environment.yml
conda activate simsid
```

</details>

## Data

SimSID is trained and evaluated on three public chest X-ray benchmarks. Download each, then set
`self.data_root` in `configs/base.py` to the directory holding them.

| dataset | what to download | link |
|:---|:---|:---|
| ZhangLab Chest X-ray | official train/test split plus our validation split | [Google Drive](https://drive.google.com/file/d/1kgYtvVvyfPnQnrPhhLt50ZK9SnxJpriC/view?usp=sharing) |
| Stanford CheXpert | official train/validation split plus our test split | [Google Drive](https://drive.google.com/file/d/14pEg9ch0fsice29O8HOjnyJ7Zg4GYNXM/view?usp=sharing) |
| COVIDx | see `dataloader/dataloader_covidx.py` | — |

`configs/base.py` ships a developer's local path as the default `data_root`. Change it before
running anything.

## Training SimSID

Experiments are driven by config files in `configs/`. Every config inherits from
`configs/base.py`, so read that one first.

| config | benchmark |
|:---|:---|
| `configs/zhang_dev.py` | ZhangLab Chest X-ray |
| `configs/chexpert_best.py` | Stanford CheXpert |
| `configs/covidx_dev.py` | COVIDx |

```bash
python main.py --config zhang_dev --exp experiment_name
```

Checkpoints, TensorBoard logs, and sample test images are written to `checkpoints/<exp>/`.

## Evaluating SimSID

```bash
python eval.py --exp experiment_name
```

`eval.py` reads the checkpoint written by `main.py` at `checkpoints/<exp>/`.

> [!IMPORTANT]
> **No pre-trained SimSID weights are released yet.** `checkpoints/` is created by `main.py`
> during training; there is nothing to download into it. Train a model first, or open an issue
> if you need released weights.

## Repository layout

| path | contents |
|:---|:---|
| `models/` | SimSID and its components: the space-aware memory matrix (`memory.py`), the feature-space in-painting block (`inpaint.py`), the autoencoder backbone (`squid.py`), and the discriminator |
| `dataloader/` | Loaders for ZhangLab (`dataloader_zhang.py`), CheXpert (`dataloader_chexpert.py`), and COVIDx (`dataloader_covidx.py`) |
| `configs/` | Experiment configs, all inheriting `configs/base.py` |
| `main.py` | Train SimSID |
| `eval.py` | Evaluate a trained SimSID checkpoint |

## Citation

If SimSID is useful in your research, please cite the TPAMI paper. If you use the CVPR version,
please also cite SQUID.

```bibtex
@article{xiang2024exploiting,
  title={Exploiting Structural Consistency of Chest Anatomy for Unsupervised Anomaly Detection in Radiography Images},
  author={Xiang, Tiange and Zhang, Yixiao and Lu, Yongyi and Yuille, Alan L. and Zhang, Chaoyi and Cai, Weidong and Zhou, Zongwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={46},
  number={9},
  pages={6070--6081},
  year={2024},
  doi={10.1109/TPAMI.2024.3382009},
  url={https://github.com/MrGiovanni/SimSID}
}

@inproceedings{xiang2023squid,
  title={SQUID: Deep Feature In-Painting for Unsupervised Anomaly Detection},
  author={Xiang, Tiange and Zhang, Yixiao and Lu, Yongyi and Yuille, Alan L. and Zhang, Chaoyi and Cai, Weidong and Zhou, Zongwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={23890--23901},
  year={2023},
  url={https://github.com/tiangexiang/SQUID}
}
```

## Contact

Yixiao Zhang — [yixiao.zhang.2023@gmail.com](mailto:yixiao.zhang.2023@gmail.com) <br/>
Zongwei Zhou — [zzhou82@jh.edu](mailto:zzhou82@jh.edu)

## Acknowledgements

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the
Patrick J. McGovern Foundation Award. We thank the authors of the ZhangLab, CheXpert, and
COVIDx datasets for making their data available.

## License

This work is licensed [CC BY-NC-ND 4.0](LICENSE) by The Johns Hopkins University.
