## SimSID: Exploiting Structural Consistency of Chest Anatomy for Unsupervised Anomaly Detection in Radiography Images</sub>

### [Paper](https://arxiv.org/pdf/2403.08689.pdf)

This repo contains PyTorch training/test code for our paper exploring structural consistency in unsupervised anomaly detection.

Radiography imaging protocols focus on particular body regions, therefore producing images of great similarity and yielding recurrent anatomical structures across patients. Exploiting this structured information could potentially ease the detection of anomalies from radiography images. To this end, we propose a Simple Space-Aware Memory Matrix for In-painting and Detecting anomalies from radiography images (abbreviated as SimSID). We formulate anomaly detection as an image reconstruction task, consisting of a space-aware memory matrix and an in-painting block in the feature space. During the training, SimSID can taxonomize the ingrained anatomical structures into recurrent visual patterns, and in the inference, it can identify anomalies (unseen/modified visual patterns) from the test image. Our SimSID surpasses the state of the arts in unsupervised anomaly detection by +8.0%, +5.0%, and +9.9% AUC scores on ZhangLab, COVIDx, and CheXpert benchmark datasets, respectively.

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