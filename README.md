# Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model

- Model framework
 
![png](https://github.com/user683/DRGO/blob/master/DRGO_Structure.png)

This paper designs a Distributionally Robust Graph model for OOD recommendation (DRGO). Specifically, our method first employs a simple and effective diffusion paradigm to alleivate the noisy effect in the latent space. Additionally, an entropy regularization term is introduced in the DRO objective function to avoid extreme sample weights in the worst-case distribution. At last, we provide a theoretical proof of the generalization error bound of DRGO as well as a theoretical analysis of how our approach mitigates noisy sample effects, which helps to better understand the proposed framework
from a theoretical perspective.

## Requirements

- torch==2.1.1+cu121  
- torch_geometric==2.5.3  
- torchaudio==2.1.1+cu121  
- torchvision==0.16.1+cu121  
- tornado==6.4.1  
- dgl==2.0.0+cu121

## Code Structures

```
.
├── dataset
├── logs
├── modules
│   ├── diffusion_model.py
│   ├── KMeans_fun.py
│   ├── LightGCN.py
│   ├── MLP_model.py
│   ├── SDE.py
│   └── Vgae.py
├── utils
│   ├── dataloader.py
│   ├── evaluation.py
│   ├── functions.py
│   ├── logger.py
│   └── loss_functions.py
├── main.py
└── parse.py

```
## Run Code

Run following python code (available dataset: "Food", "Kuairec") with default hyperparameters to reproduce our results.

- Food
```
python main.py --dataset Food --lr 0.001 --batch_size 4096 --epoch 20 --dims 64
```
- Kuairec
```
python main.py --dataset Food --lr 0.0001 --batch_size 4096 --epoch 20 --dims 64
```

## Dataset

|  Dataset   |  #Users  |  #Items  |  #Interactions  |   Density   |
|:----------:|:--------:|:--------:|:---------------:|:-----------:|
|    Food    |  7,809   |  6,309   |     216,407     | 4.4 × 10⁻³  |
|  KuaiRec   |  7,175   |  10,611  |    1,153,797    | 1.5 × 10⁻³  |
|  Yelp2018  |  8,090   |  13,878  |     398,216     | 3.5 × 10⁻³  |
|   Douban   |  8,735   |  13,143  |     354,933     | 3.1 × 10⁻³  |

