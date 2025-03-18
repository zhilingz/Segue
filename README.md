# Segue: Side-information Guided Generative Unlearnable Examples for Facial Privacy Protection in Real World 

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