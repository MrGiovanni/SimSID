This repo provides official code for SimSID, a significant technical improvement over our previous CVPR version (SQUID) in exploring structural consistency for **unsupervised anomaly detection**.

<div align="center">
 
![logo](document/fig_introductory.png)  
</div>

Anomaly detection in radiography images can be both easier and harder than photographic images. It is easier because radiography images are spatially structured due to consistent imaging protocols. It is harder because anomalies are subtle and require medical expertise to annotate.

## Paper

<b>Exploiting Structural Consistency of Chest Anatomy for Unsupervised Anomaly Detection in Radiography Images</b> <br/>
[Tiange Xiang](https://tiangexiang.github.io/)<sup>1</sup>, [Yixiao Zhang](https://0-scholar-google-com.brum.beds.ac.uk/citations?user=lU3wroMAAAAJ&hl=fi)<sup>2</sup>, [Yongyi Lu](https://scholar.google.com/citations?user=rIJ99V4AAAAJ&hl=en&oi=ao)<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, [Chaoyi Zhang](https://chaoyivision.github.io/)<sup>1</sup>, [Weidong Cai](https://weidong-tom-cai.github.io/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com)<sup>2</sup> <br/>
<sup>1</sup>University of Sydney,  <sup>2</sup>Johns Hopkins University <br/>
TPAMI <br/>
[paper](https://www.cs.jhu.edu/~alanlab/Pubs24/xiang2024exploiting.pdf) | [code](https://github.com/MrGiovanni/SimSID)

**SQUID: Deep Feature In-Painting for Unsupervised Anomaly Detection** <br/>
[Tiange Xiang](https://tiangexiang.github.io/)<sup>1</sup>, [Yixiao Zhang](https://0-scholar-google-com.brum.beds.ac.uk/citations?user=lU3wroMAAAAJ&hl=fi)<sup>2</sup>, [Yongyi Lu](https://scholar.google.com/citations?user=rIJ99V4AAAAJ&hl=en&oi=ao)<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, [Chaoyi Zhang](https://chaoyivision.github.io/)<sup>1</sup>, [Weidong Cai](https://weidong-tom-cai.github.io/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com)<sup>2</sup> <br/>
<sup>1</sup>University of Sydney,  <sup>2</sup>Johns Hopkins University <br/>
CVPR, 2023 <br/>
[paper](https://arxiv.org/pdf/2111.13495.pdf) | [code](https://github.com/tiangexiang/SQUID)

## Training

We provide a training script main.py. This script can be used to train an unsupervised anomaly detection model on a dataset (ZhangLab, CheXpert or COVIDx) by specifying a model config:

```bash
python main.py --config zhang_dev.py --exp experiment_name
```
All the configs can be found in the `configs` subfolder.

## Test

You can test with our **pre-trained SimSID models** with eval.py. Weights for our pre-trained SimSID models can be found in the `checkpoints` subfolder.

```bash
python eval.py --exp experiment_name
```

## Contact Us
Yixiao Zhang: [yixiao.zhang.2023@gmail.com](mailto:yixiao.zhang.2023@gmail.com)
Zongwei Zhou: [giovanni.z.zhou@gmail.com](mailto:giovanni.z.zhou@gmail.com)

## Citation
If you find this work useful for your research, please consider citing it.
```bibtex
@article{xiang2024exploiting,
  title={Exploiting Structural Consistency of Chest Anatomy for Unsupervised Anomaly Detection in Radiography Images},
  author={Xiang, Tiange and Zhang, Yixiao and Lu, Yongyi and Yuille, Alan and Zhang, Chaoyi and Cai, Weidong and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2403.08689},
  year={2024}
}

@article{xiang2023painting,
  title={In-painting Radiography Images for Unsupervised Anomaly Detection},
  author={Xiang, Tiange and Liu, Yongyi and Yuille, Alan L and Zhang, Chaoyi and Cai, Weidong and Zhou, Zongwei},
  journal={IEEE/CVF Converence on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Acknowledgements
This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the Patrick J. McGovern Foundation Award. 
