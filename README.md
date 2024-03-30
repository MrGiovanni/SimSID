
This repo contains PyTorch training/test code for our paper exploring structural consistency in unsupervised anomaly detection.

## Paper

<b>Exploiting Structural Consistency of Chest Anatomy for Unsupervised Anomaly Detection in Radiography Images</b> <br/>
[Tiange Xiang](), [Yixiao Zhang](), [Yongyi Li](), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Chaoyi Zhang](), [Weidong Cai](), and [Zongwei Zhou](https://www.zongweiz.com/)<sup>*</sup> <br/>
Johns Hopkins University  <br/>
International Conference on Learning Representations (ICLR) 2024 (oral; top 1.2%) <br/>
[paper](https://www.cs.jhu.edu/~alanlab/Pubs23/li2023suprem.pdf) | [code](https://github.com/MrGiovanni/SuPreM) | [slides](document/promotion_slides.pdf) | [poster](document/dom_wse_poster.pdf) | [talk](https://vtizr.xetslk.com/s/1HUGNo
) | [news](https://www.cs.jhu.edu/news/ai-and-radiologists-unite-to-map-the-abdomen/)

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
```

## Acknowledgements
SimSID has been greatly inspired by the amazing work [SQUID](https://github.com/tiangexiang/SQUID)
