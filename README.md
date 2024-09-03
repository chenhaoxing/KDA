# Boosting Audio-visual Zero-shot Learning with Large Language Models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-audio-visual-zero-shot-learning-with/gzsl-video-classification-on-activitynet-gzsl)](https://paperswithcode.com/sota/gzsl-video-classification-on-activitynet-gzsl?p=boosting-audio-visual-zero-shot-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-audio-visual-zero-shot-learning-with/gzsl-video-classification-on-activitynet-gzsl-1)](https://paperswithcode.com/sota/gzsl-video-classification-on-activitynet-gzsl-1?p=boosting-audio-visual-zero-shot-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-audio-visual-zero-shot-learning-with/gzsl-video-classification-on-ucf-gzsl-cls)](https://paperswithcode.com/sota/gzsl-video-classification-on-ucf-gzsl-cls?p=boosting-audio-visual-zero-shot-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-audio-visual-zero-shot-learning-with/gzsl-video-classification-on-ucf-gzsl-main)](https://paperswithcode.com/sota/gzsl-video-classification-on-ucf-gzsl-main?p=boosting-audio-visual-zero-shot-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-audio-visual-zero-shot-learning-with/gzsl-video-classification-on-vggsound-gzsl)](https://paperswithcode.com/sota/gzsl-video-classification-on-vggsound-gzsl?p=boosting-audio-visual-zero-shot-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-audio-visual-zero-shot-learning-with/gzsl-video-classification-on-vggsound-gzsl-1)](https://paperswithcode.com/sota/gzsl-video-classification-on-vggsound-gzsl-1?p=boosting-audio-visual-zero-shot-learning-with)

This is the official code of the paper: Boosting Audio-visual Zero-shot Learning with Large Language Models. Data sets and environments can be prepared by referring to [AVCA](https://github.com/ExplainableML/AVCA-GZSL). 

![](Doc/KDA.jpg)
> [**Boosting Audio-visual Zero-shot Learning with Large Language Models**](https://arxiv.org/abs/2311.12268)               
> [Haoxing Chen](https://scholar.google.com/citations?hl=zh-CN&pli=1&user=BnS7HzAAAAAJ), [Yaohui Li](https://scholar.google.com/citations?user=pC2kmQoAAAAJ&hl=zh-CN), [Yan Hong](https://scholar.google.com/citations?user=ztq5-xcAAAAJ&hl=zh-CN), Zizheng Huang, [Zhuoer Xu](https://scholar.google.com/citations?user=na24qQoAAAAJ&hl=zh-CN&oi=ao), [Zhangxuan Gu](https://scholar.google.com/citations?user=Wkp3s68AAAAJ&hl=zh-CN&oi=ao), Jun Lan, Huijia Zhu, [Weiqiang Wang](https://scholar.google.com/citations?hl=zh-CN&user=yZ5iffAAAAAJ), [arXiv preprint arXiv: 2311.12268](https://arxiv.org/abs/2311.12268) 

## Training and Evaluating KDA
For example, conduct experiment on UCF-GZSL-main dataset:
```python
python main.py --root_dir avgzsl_benchmark_datasets/UCF/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512 --lr_scheduler --dataset_name UCF --zero_shot_split main_split --epochs 50 --lr 0.001 --n_batches 50 --bs 2048 --kda --retrain_all --exp_name KDA_UCF_all_main

```
Pay attention, you need to set the three parameters (drop_rate_enc/drop_rate_proj/momentum) in [model.py](https://github.com/chenhaoxing/KDA/blob/main/src/model.py) from lines 207 to 209 according to our paper. Additionally, when training the model on different datasets, modify lines 177-194 in [utils.py](https://github.com/chenhaoxing/KDA/blob/main/src/utils.py) to align the action IDs accordingly; and modify line 230 in [model.py](https://github.com/chenhaoxing/KDA/blob/main/src/model.py) to match the description file.


## Citing KDA
If you use KDA in your research, please use the following BibTeX entry.

```BibTeX
@article{KDA_2023,
      title={Boosting Audio-visual Zero-shot Learning with Large Language Models},
      author={Chen, Haoxing and Li, Yaohui and Hong, Yan and Xu, Zhuoer and Gu, Zhangxuan and Lan, Jun and Zhu, Huijia and Wang, Weiqiang},
      journal={arXiv preprint arXiv: 2311.12268},
      year={2023}
}
```

## Acknowledgement

This repo is built based on [AVCA](https://github.com/ExplainableML/AVCA-GZSL), thanks!


## Contacts
Please feel free to contact us if you have any problems.

Email: [haoxingchen@smail.nju.edu.cn](haoxingchen@smail.nju.edu.cn) or [hx.chen@hotmail.com](chen@hotmail.com)
