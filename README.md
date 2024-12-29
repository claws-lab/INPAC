# INPAC

This repository contains the code and data for the KDD 2023 paper [Predicting Information Pathways Across Online Communities](https://arxiv.org/abs/2306.02259).

* Authors: [Yiqiao Jin](https://ahren09.github.io/), [Yeon-Chang Lee](https://sites.google.com/view/yclee/), [Kartik Sharma](https://ksartik.github.io/), [Meng Ye](https://www.linkedin.com/in/meng-ye-36711114b/), [Karan Sikka](https://www.linkedin.com/in/ksikka/), [Ajay Divakaran](https://www.linkedin.com/in/ajay-divakaran-3445361/), [Srijan Kumar](https://faculty.cc.gatech.edu/~srijan/)
* Organizations: [Georgia Institute of Technology](https://www.gatech.edu/), [SRI International](https://www.sri.com/)




If our code or data helps you in your research, please kindly cite us:
```
@inproceedings{jin2023predicting,
  title        = {Predicting Information Pathways Across Online Communities},
  author       = {Jin, Yiqiao and Lee, Yeon-Chang and Sharma, Kartik and Ye, Meng and Sikka, Karan and Divakaran, Ajay and Kumar, Srijan},
  year         = 2023,
  booktitle    = {KDD},
}

```

## Introduction

The problem of community-level information pathway prediction (CLIPP) aims at predicting the transmission trajectory of content across online communities. A successful solution to CLIPP holds significance as it facilitates the distribution of valuable information to a larger audience and prevents the proliferation of misinformation. Notably, solving CLIPP is non-trivial as inter-community relationships and influence are unknown, information spread is multi-modal, and new content and new communities appear over time. In this work, we address CLIPP by collecting large-scale, multi-modal datasets to examine the diffusion of online YouTube videos on Reddit. We analyze these datasets to construct community influence graphs (CIGs) and develop a novel dynamic graph framework, INPAC (Information Pathway Across Online Communities), which incorporates CIGs to capture the temporal variability and multi-modal nature of video propagation across communities. Experimental results in both warm-start and cold-start scenarios show that INPAC outperforms seven baselines in CLIPP. 

![INPAC](static/img/INPAC.png)

## 🤗 HuggingFace Dataset

We constructed real-world, large-scale datasets covering 60 months of Reddit posts sharing YouTube videos, from January 2018 to December 2022, available on 🤗 HuggingFace (<a href="https://huggingface.co/datasets/Ahren09/reddit">Ahren09/reddit</a>)

Install the `datasets` library:

```bash
pip install datasets
```

You can load the dataset using:

```python]
from datasets import load_dataset

dataset = load_dataset("Ahren09/reddit", "2018") 
```

where "2018" is the subset name. Replace it with "2019", ..., "2022" to load the other subsets

## Installation


```angular2html
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge tensorflow
```


## Run the code

NOTE: To avoid any import or path issues, it is recommended to use PyCharm.  

For the large dataset, run

```
python main.py --dataset_name large --do_static_modeling --session_split_method session --delta_t_thres 4.13625 --do_val

```

For the small dataset, run

```
python main.py --dataset_name small --do_static_modeling --session_split_method session --delta_t_thres 4.13625 --do_val

```

- `dataset_name`: `small` for the 3-month Small dataset, `large` for the 54-month Large dataset.
- `delta_t_thres`: The precomputed threshold in Section 3.2. You can also run without specifying `delta_t_thres` and let the code compute it for you.
- `c`, `mu`, `sigma`: Hyperparameters in the equation $\delta t^{thres} = \mu - c \sigma$.
- `resource`: `v` for video. We will include more types of resources in the future, such as `url`
- `eval_neg_sampling_ratio`: the number of negative items to sample for each positive interaction. This is for evaluation.
- `eval_every`: evaluate the model every `eval_every` epochs.

## Data

The data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1zgqbBHpQKt_RUJrIRISfZILUXLxPTx1I?usp=sharing). Please put the entire `data/` folder under `INPAC` 

The `urls_df.pkl` file contains the unfiltered data:

```angular2html
                                                 url           netloc post_id   timestamp       subreddit             author            v
0                       https://youtu.be/tmmpaOZ3nQg         youtu.be  eiazyl  1577836805  virtualreality          Zweetprot  tmmpaOZ3nQg
1        https://www.youtube.com/watch?v=LuAyGWqYza4  www.youtube.com  eib0a6  1577836845          FTMMen  00110100-00110010  LuAyGWqYza4
2        https://www.youtube.com/watch?v=d4hJA7IUaDs  www.youtube.com  eib0a6  1577836845          FTMMen  00110100-00110010  d4hJA7IUaDs
3  https://www.youtube.com/watch?v=5U_2V6yr-Nw&fe...  www.youtube.com  eib0a6  1577836845          FTMMen  00110100-00110010  5U_2V6yr-Nw
4                       https://youtu.be/tmmpaOZ3nQg         youtu.be  eib0em  1577836862         SteamVR          Zweetprot  tmmpaOZ3nQg
5                       https://youtu.be/mumHdNhclrM         youtu.be  eib0h6  1577836869  SmallYTChannel      thevinamazing  mumHdNhclrM
6                       https://youtu.be/tmmpaOZ3nQg         youtu.be  eib0nk  1577836892        VRGaming          Zweetprot  tmmpaOZ3nQg
7        https://www.youtube.com/watch?v=uxtqIvOP0rQ  www.youtube.com  eib0se  1577836909        ripplers            daNext1  uxtqIvOP0rQ
8                       https://youtu.be/tmmpaOZ3nQg         youtu.be  eib0ur  1577836917        HTC_Vive          Zweetprot  tmmpaOZ3nQg
9                       https://youtu.be/HE1Vy5lKuzw         youtu.be  eib0wn  1577836926      HelpMeFind            Sanojoj  HE1Vy5lKuzw

```

Each row represents a video $v_i$ being shared in a subreddit $s_j$ by some user $u_k$ at some time $t$. We retain videos that have been shared in >=3 online communities (subreddits). The filtered dataset is stored in `reddit_dataset.pkl` along with the mappings.


## Contact

If you have any questions, please contact the author Yiqiao Jin.
