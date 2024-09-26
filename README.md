#  [NeurIPS 2024] MetaNCoV
Implementation of NeurIPS 2024 paper “[Meta-Learning Universal Priors Using Non-Injective Change of Variables](https://openreview.net/forum?id=E8b4yOLGZ5)” (to appear soon). 

## Preparation

Codes tested under the following environment:

------

- PyTorch 2.0.0
- CUDA 11.7
- CUDNN 8.5.0
- torchvision 1.10.1
- torch-utils 0.1.2
- [torchmeta](https://github.com/tristandeleu/pytorch-meta) (modified from 1.8.0)

---

One can setup the enviroment using `env_setup.sh`. 

It worth noting that the tests for CUB datasets requires downgrading the packages; see instructions in `env_setup.sh`

## Experiments

Default experimental setups can be found in `main.py`. To carry out the numerical test, use the commands

```
python main.py "--arguments" "values"
```

where `arguments` and `values` are the algorithm parameters that you want to alter.

To reproduce the results reported in our paper, please use the scripts provided in `bash_scripts.sh`. 

## Citation

> Y. Zhang, A. Sadeghi, and G. B. Giannakis, “Meta-Learning Universal Priors Using Non-Injective Change of Variables,” in *Proceedings of Advances in Neural Information Processing Systems*, 2024. 

```tex
@inproceedings{MetaNCoV, 
  author={Zhang, Yilang and Sadeghi, Alireza and Giannakis, Georgios B.}, 
  title={Meta-Learning Priors Using Unrolled Proximal Networks}, 
  booktitle={Advances in Neural Information Processing Systems}, 
  year={2024}, 
  url={https://openreview.net/forum?id=E8b4yOLGZ5},
}
```

