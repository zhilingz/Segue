# GUE
第一步：训练噪声 
```
python noise_train.py
```

第二步：生成不可学习数据集 
```
python mknoisedata.py
```

第三步：代理模型在不可学习数据集上训练 
```
python train_model.py
```

噪声训练方法：
* GUE: Generative Unlearnable Examples
* UE: Unlearnable Examples
* RUE: Robust Unlearnable Examples
* TUE: Transferable Unlearnable Examples