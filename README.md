# Continual Learning in Visual Question Answering

<p align="center">
    <br>
    <img src="figure.svg" width="900"/>
    <br>
<p>

This repository contains information about the following three settings for studying continual learning in Visual Question Answering:

- Diverse Domains: Each task is defined based on the objects that appear in the images. Different objects are grouped randomly together in each of the five tasks.

- Taxonomy Domains: Each task is defined based on the objects that appear in the images. Objects from the same supercategory are grouped in the same task, leading to the five following tasks: Animals, Food, Interior, Sports, Transport.
The definitions of the tasks follow work by [Del Chiaro et al., 2020](https://arxiv.org/abs/2007.06271).

- Question Types: Each task is defined based on the question type:
Action Recognition (e.g. "What are the kids doing?"), Color Recognition (e.g. "What color hat is the man wearing?"), Counting ("How many of the people are wearing hats?"), Subcategory Recognition (e.g. "What type of hat is he wearing?"), Scene-level Recongition (e.g. "Was this photo taken indoors?").
The definitions of the tasks follow work by [Whitehead et al., 2021](https://arxiv.org/abs/2107.09106).

## Task Statistics per Setting

### Diverse Domains

| Task   | Train | Validation | Test | Number of Classes |
|--------|:-----:|:----------:|:--------:|:----------:|
|Group_1 |44254 |11148 |28315 |2205 |
|Group_2 |39867 |10202 |22713 |1874 |
|Group_3 |37477 |9386 |23095 |1849 |
|Group_4 |35264 |8871 |22157 |2119 |
|Group_5 |24454 |6028 |14490 |1777 |


### Taxonomy Domains

| Task   | Train | Validation | Test | Number of Classes |
|--------|:-----:|:----------:|:--------:|:----------:|
| Animals | 37270 | 9237 | 22588 | 1331 |
| Food | 26191 | 6612 | 15967 | 1365 |
| Interior | 43576 | 11038 | 26594 | 2096 |
| Sports | 32885 | 8468 | 19205 | 1471 |
| Transport | 41394 | 10280 | 25416 | 1954 |

### Question Types

| Task   | Train | Validation | Test | Number of Classes |
|--------|:-----:|:----------:|:--------:|:----------:|
| Action | 18730 | 4700 | 11008 | 233 |
| Color | 34588 | 8578 | 21559 | 92 |
| Count | 38857 | 9649 | 23261 | 42 |
| Scene | 25850 | 6417 | 14847 | 170 |
| Subcategory | 22324 | 8578 | 13564 | 659 |

## Data
1. Download the VQA data from the [visualqa.org](https://visualqa.org/download.html). Because the annotations from the test set are not publicly available, the VQA-v2 validation data are used as the test set in ContVQA, and the VQA-v2 training data are split into train and validation set.
2. Get the question ids for each task under the corresponding folder in `data/`. Each file contains the ids for the train/validation/test splits in the following format:

```
{
    task_name: 
    [
        question_id
    ]
}
```

For more details about the settings please refer to our [preprint](https://arxiv.org/abs/2210.00044). Note that the main results are averaged over five random task orders which can be found under `task_orders/`.

## Citation

```bibtex
@article{nikandrou2022task,
  title={Task formulation matters when learning continually: A case study in visual question answering},
  author={Nikandrou, Mavina and Yu, Lu and Suglia, Alessandro and Konstas, Ioannis and Rieser, Verena},
  journal={arXiv preprint arXiv:2210.00044},
  year={2022}
}


