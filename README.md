# Segue: Side-information Guided Generative Unlearnable Examples for Facial Privacy Protection in Real World 

Official PyTorch implementation of our IEEE ICASSP 2025 paper: ["Segue: Side-information Guided Generative Unlearnable Examples for Facial Privacy Protection in Real World"](https://ieeexplore.ieee.org/document/10889952).

Zhiling Zhang, Jie Zhang, Kui Zhang, Wenbo Zhou, Ting Xu, Daiheng Gao, Zixian Guo, Qinglang Guo, Weiming Zhang, Nenghai Yu

Contact: [zhilingzhang@mail.ustc.edu.en](mailto:zhilingzhang@mail.ustc.edu.en)


Abstract: 
*The widespread adoption of face recognition has raised privacy concerns regarding the collection and use of facial data. To address this, researchers have explored "unlearnable examples" by adding imperceptible perturbations during model training to prevent the model from learning target features. However, current methods are inefficient and cannot guarantee transferability and robustness at the same time, causing impracticality in the real world. To remedy it, we introduce Side-information Guided Generative Unlearnable Examples (Segue). Using a once-trained multiple-used model to generate perturbations, Segue avoids the time-consuming gradient-based approach. To improve transferability, we introduce side information such as true or pseudo labels, which are inherently consistent across different scenarios. For robustness enhancement, a distortion layer is integrated into the training pipeline. Experiments show Segue is 1000× faster than previous methods, transferable across datasets and models, and resistant to JPEG compression, adversarial training, and standard augmentations.*

## Experiments of Supervised Scenario.
kmeans_label=False
#### First:
1. train generator and surrogate model
2. save generator
```
python Segue_1.py False
```

#### Second:
1. load generator
2. generate perturbations for iamges and save perturbed images (to form unlearnable dataset)
```
python Segue_2.py False
```

#### Finally:
1. load clean dataset and unlearnable dataset
2. attack model
3. train attack model on unlearnable dataset and test on clean dataset
```
python Segue_3.py
```

## Experiments of Unsupervised Scenario.
kmeans_label=True

#### pre:
```
python kmeans.py
```

#### First:
1. train generator and surrogate model
2. save generator
```
python Segue_1.py True
```

#### Second:
1. load generator
2. generate perturbations for iamges and save perturbed images (to form unlearnable dataset)
```
python Segue_2.py True
```

#### Finally:
1. load clean dataset and unlearnable dataset
2. attack model
3. train attack model on unlearnable dataset and test on clean dataset
```
python Segue_3.py
```
