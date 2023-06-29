<div align="center">

# Exploiting Rich Textual User-Product Context for Improving Personalized Sentiment Analysis


**[Chenyang Lyu](https://lyuchenyang.github.io), [Linyi Yang](mailto:yanglinyi@westlake.edu.cn), [Yue Zhang](mailto:zhangyue@westlake.edu.cn), [Yvette Graham](mailto:ygraham@tcd.ie), [Jennifer Foster](mailto:jennifer.foster@dcu.ie)**

School of Computing, Dublin City University, Dublin, Ireland &#x1F3E0;

School of Engineering, Westlake University, China;

School of Computer Science and Statistics, Trinity College Dublin, Dublin, Ireland

</div>

## Introduction 📝
This repository contains the code and resources for our Findings of ACL 2023 paper titled "Exploiting Rich Textual User-Product Context for Improving Personalized Sentiment Analysis". In this paper, we propose a novel approach to improve personalized sentiment analysis by leveraging rich textual user-product context.

## Installation 📋
To get started, please install the required libraries by running the following command:

```angular2
pip install -r requirements.txt
```

## Download datasets 📥
Next, download the datasets from the following URL: [dataset](https://drive.google.com/file/d/1Bdt_jw-kiZCt7vJyfXe1hYmPKMinbtFu/view?usp=sharing).

Unzip the downloaded zip file and move all dataset files to the "data/personalized-sa/" directory.

## Training 🚀
To train the model, use the following code:

```
python run_cross_context_sa.py --task_name yelp-2013 \
    --model_type bert \
    --model_size base \
    --epochs 5 \
    --do_train \
    --weight_decay 0.0 \
    --learning_rate 5e-5 \
    --warmup_steps 0.2 \
    --max_seq_length 512 \                            
```

## Evaluation 📊
To evaluate a trained model with the specified parameters, use the following code:

```
python run_cross_context_sa.py --task_name yelp-2013 \
    --model_type bert \
    --model_size base \
    --epochs 5 \
    --do_eval \
    --weight_decay 0.0 \
    --learning_rate 5e-5 \
    --warmup_steps 0.2 \
    --max_seq_length 512 \                            
```

## License 📄
This work is licensed under a [Creative Commons Attribution 4.0 International Licence](http://creativecommons.org/licenses/by/4.0/).

## Citation 📄

Please cite our paper using the bibtex below if you found that our paper is useful to you:

```bibtex
@article{lyu2022exploiting,
  title={Exploiting Rich Textual User-Product Context for Improving Sentiment Analysis},
  author={Lyu, Chenyang and Yang, Linyi and Zhang, Yue and Graham, Yvette and Foster, Jennifer},
  journal={arXiv preprint arXiv:2212.08888},
  year={2022}
}
```
