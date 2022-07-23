# GCARec
This is our Tensorflow implementation for our GCARec 2022 paper and a part of baselines:

>Bin Wu, Xiangnan He, Le Wu, Xue Zhang, Yangdong Ye. Graph Augmented Co-Attention Model for Socio-Sequential Recommendation, IEEE Transactions on Systems, Man and Cybernetics: Systems, Under Review

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.14.0
* numpy == 1.16.4
* scipy == 1.3.1
* pandas == 0.17

## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.
NOTE: The cpp implementation is much faster than python.**

## Examples to run GCARec:
run [main.py](./main.py) in IDE or with command line:
```
python main.py
```

NOTE :   
(1) the duration of training and testing depends on the running environment.  
(2) set model hyperparameters on .\conf\GCARec.properties  
(3) set NeuRec parameters on .\NeuRec.properties  
(4) the log file save at .\log\Ciao_u5_s2\  

## Dataset
We provide Ciao_u5_s2(Ciao) dataset.
  * .\dataset\Ciao_u5_s2.rating and Ciao_u5_s2.uu
  *  Each line is a user with her/his positive interactions with items: userID \ itemID \ ratings \time.
  *  Each user has more than 10 associated actions.

## Baselines
The list of available models in GCARec, along with their paper citations, are shown below:

| General Recommender | Paper                                                                                                         |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| BPRMF               | Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.                   |
| LightGCN            | Xiangnan He, et al., LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.|

| Sequential Recommender | Paper                                                                                                      |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| FPMC            |S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, Factorizing personalized markov chains for next-basket recommendation, WWW, 2010.|
| HGN            | C. Ma, P. Kang, and X. Liu, Hierarchical gating networks for sequential recommendation, KDD, 2019.|

| Social Recommender | Paper                                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------------|
| EAGCN              | B. Wu, L. Zhong, L. Yao, and Y. Ye, “EAGCN: An efficient adaptive graph convolutional network for item recommendation in social internet of things, IOT, 2022.|
