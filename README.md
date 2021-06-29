# Generation of Synthetic Health Data with Sequential Treatments: a Case Study on Colorectal Cancer Patients Data in the Netherlands Cancer Registry

This repository contains the (public) code used in my master thesis. 

## Abstract
Sharing medical and patient data with a larger (research) community is beneficial to further stimulate and accelerate advances in medical knowledge and patient care. However, medical patient data is often highly sensitive and private, making it impossible to distribute the data to a wider audience. The recent literature proposes the generation of synthetic medical data that mimics the statistical properties of the original data to overcome these privacy concerns. This thesis focuses on the generation and evaluation of a type of data largely unexplored in the existing literature: health event data. More specifically, we focus on the specific application of cancer patient data, including both static patient covariates and treatment sequences, in the Netherlands Cancer Registry maintained by the Netherlands Comprehensive Cancer Organization. Based on a literature review, we adapted or directly applied two existing generative methods to our use case. We find that both generative methods are able to generate high-quality synthetic data by considerably outperforming the naive baseline on a set of seven quality metrics. We defined these seven (existing) metrics to evaluate the quality of the specific type of health event data from several aspects and confirm their validity. With these metrics, we aim to motivate and enable healthcare organizations to participate in the synthetic health data movement while ensuring and evaluating the quality of the generated data. In a follow-up experiment, we implement the differentially private version of one of the selected generative methods. The results show that - using additional techniques including generalization and post-processing - we can obtain a differentially private synthetic data set with reasonable quality for a privacy budget generally considered just acceptable. For stricter privacy guarantees, characteristics and statistical properties of the original data are lost in the synthetic data. We encourage future work to further investigate the privacy aspect of synthetic health event data. 

## Repository structure
Original data was stored outside the repository to prevent accidental commits including confidential data. Furthermore, data frame outputs containing confidential data, extensive evaluation figures disclosing data set information, and code (notebooks) for exploratory data analysis were omitted from this public repository. 

* */evaluation metrics*: contains quality evaluation script with QualityEvaluator class, including code for all seven quality evaluation metrics presented in the thesis (Section 4.3)

* */experiments*: contains code for conducting the experiments in the thesis. Directory is further divided into sub-directories (see tree structure below). */experiments/Experiment_NP/DoppelGANger* uses code from https://github.com/fjxmlzn/DoppelGANger. Each experiment folder */experiments/Experiment_NP* and */experiments/Experiment_DP* contains a quality evaluation notebook, with the quality evaluation results for that experiment. */experiments/Experiment_NP/QualityEvaluator_NP.ipynb* is the quality evaluation for the No Privacy (NP) experiments (Section 5.1), and */experiments/Experiment_NP/QualityEvaluator_PBS-DP-Exp2.ipynb* and QualityEvaluator_PBS-DP-Naive.ipynb are the quality evaluations for the Differentially Private (DP) experiments for PrivBayes for Exp. 2 and Naive respectively (Section 5.2). 

``` bash 
├───Experiment1_NP # NoPrivacy experiments (Section 4.4.1)
│       ├───DoppelGANger # Method elaborated upon in Section 4.2.3 (method reference: Lin et al., 2020)
│       ├───MarginalSynthesizer # Method elaboarated upon in Section 4.2.4
│       ├───PrivBayes # Method elaborated upon in Section 4.2.2 (method adapted from Zhang et al., 2017)
│       └───synthetic_data # Where the synthetic data is stored, content is omitted due to potential privacy issues
├───Experiment2_DP # Differential Privacy experiments (Section 4.4.2)
│       ├───PrivBayes # Contains both Exp 2. and Naive DP experiments
│       └───synthetic_data # Where the synthetic data is stored, content is omitted due to confidentiality
``` 

* */notebooks*: directory where exploratory data analysis notebooks were stored. Content of this folder including its sub-directories is omitted due to confidentiality of the outputs. 
* */preprocessing*: directory with script used for pre-processing the raw data from the Netherlands Cancer Registry (NCR). 
* */synthesis*: source code for MarginalSynthesizer and PrivBayes. Adapted from source: https://github.com/daanknoors/synthetic_data_generation. 


## References
Lin,  Z.,  Jain,  A.,  Wang,  C.,  Fanti,  G.,  &  Sekar,  V.  (2020).  Using  GANs  for  Sharing Networked Time Series Data, 464–483. https://doi.org/10.1145/3419394.3423643

Zhang, J., Cormode, G., Procopiuc, C. M., Srivastava, D., & Xiao, X. (2017). Privbayes:Private data release via bayesian networks. *ACM Transactions on Database Systems (TODS), 42*(4), 1–41.
