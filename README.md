# federate-distiliing
## 1.单个数据集维度的联邦学习

在单个数据集中进行联邦训练,直接运行run.py，运行参数如下：

```
python run.py --from_pretrained ./model --dataset svamp --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 16
```

其中利用--dataset选择不同的数据集

其他调整参数：

![image](https://github.com/panker123/federate-distiliing/assets/56550368/4bb90b4f-3274-4e78-84af-a25761109a38)


--subsample:选择训练数据的大小，原文依次选择了0.125，0.25，0.5，1三个数据大小进行训练

--epochs:训练轮数

--clients_number:初始客户端个数



实验结果：

![image](https://github.com/panker123/federate-distiliing/assets/56550368/6791f557-c886-4d42-958e-e63709b3cfa8)




## 2. 四个数据集维度的联邦学习

将四个数据集共同进行训练，初始4个客户端，每一个客户端代表一个数据集，每一个客户端在本地训练不同的数据集后进行聚合，使用聚合的参数在总的测试数据中进行测试。

运行run_1.py

```
python run_1.py
```

其他调整参数

![image](https://github.com/panker123/federate-distiliing/assets/56550368/6067aaca-cb9f-4192-8562-d7cf7e0a862a)



--subsample:调整训练数据集大小

--epochs:调整训练轮数



结果

![image](https://github.com/panker123/federate-distiliing/assets/56550368/f0727798-9bd6-4c89-b13b-b9bc939b1789)

