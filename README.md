### 直接运行
```
./run.sh
```
`run.sh` 分三步运行

#### 第一步：训练噪声 
```
python noise_train.py $method $model $log $log_path
```

#### 第二步：生成不可学习数据集 
```
python mknoisedata.py $method $model $log $log_path
```

#### 第三步：代理模型在不可学习数据集上训练 
```
python train_model.py $method $model $log $log_path
```

<!-- 噪声训练方法：
* GUE: Generative Unlearnable Examples
* UE: Unlearnable Examples
* RUE: Robust Unlearnable Examples
* TUE: Transferable Unlearnable Examples -->