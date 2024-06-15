# Active learning methods for protein-ligand binding

## To-do:
- [X] Get AL cycle running for one target, we can use GP model to start with.
2. Things that could be explored-
- [X]  initial sample selection strategies
- [X]  acquisition functions and protocols
- []  studying the patterns on more datasets
- []  finetuning Chemberta2 model in place of GP, we can use PEFT techniques like having adapters on Chemebrta2 transformers (we can discuss this)


### Relevant Papers
#### Traversing Chemical Space with Active Deep Learning in Low-data Scenarios

- The paper "Traversing Chemical Space with Active Deep Learning in Low-data Scenarios" explores the application of active deep learning in drug discovery, focusing on three targets of therapeutic interest: Pyruvate kinase M2 (PKM2), Aldehyde dehydrogenase 1 (ALDH1), and Vitamin D receptor (VDR)
The study constructs a screening library from which training sets and molecules for screening are drawn. It examines the impact of structural diversity in the starting set by creating subsets with varying degrees of diversity. _The adaptive nature of active learning allows for the discovery of relevant structure-activity relationships during screening._

- Two deep learning strategies, neural networks using molecular fingerprints and graph neural networks, are employed for bioactivity predictions
**Neural networks (multi-layer perceptron)** that learn 
from molecular fingerprints in the form of Extended 
Connectivity Fingerprints43 (ECFPs). These molecular 
fingerprints encode the presence of radial, atom centred substructures. 
**Graph neural networks**, which learn directly from the 
molecular graph. Molecular graphs are a direct 
numeric representation of molecular topology, with 
nodes and edges representing atoms and chemical 
bonds respectively

- The study evaluates six acquisition functions to select molecules for screening, highlighting the crucial role of the acquisition function in hit retrieval and active learning performance
The research finds that active deep learning significantly improves hit enrichment compared to traditional methods, **with exploitation and acquisition based on mutual information outperforming other acquisition functions**Bayesian Active Learning by Disagreement (BALD), mutual information is used to quantify the uncertainty of a model's predictions. The mutual information between the input (molecule) and the output (bioactivity) is low when there are many possible ways of predicting the data with high certainty, given the same model

- **The acquisition size was determined through preliminary experiments where we compared acquiring 16, 32, or 64 molecules per cycle and found no significant performance differences**



#### Benchmarking active learning protocols for ligand binding affinity prediction
- Gaussian processes (GP) and conditional processes (CP) models in predicting the binding affinity of compounds to target proteins. The models are trained on four datasets: TYK2, USP7, D2R, and Mpro, using a 5-fold split for training and testing. The models' performance is assessed using the coefficient of determination R2, Spearman ρ, and the Recall of top 2% and 5% samples across the test sets.It also discusses the impact of noise on the models' performance.

- It evaluates the influence of active learning parameters, dataset features, and sample selection protocols on the performance of machine learning models, Gaussian Process models and Chemprop, in predicting the binding affinity of compounds to target proteins

- The smaller the batch size, the better, The confinement of
chemical space is essential for a small number of samples to be able to represent a sizable
pool. In practice, however, a combinatorial exploration of substituents is not always desirable
when a strict filtering of compounds is necessary due to constraints on the physicochemical
properties. In such cases, increasing the batch size is a more defensive approach to achieve
success with AL

- In exploitation phase, findings consistently suggest that training in small batches
results in the highest Recall. The performance gains are incremental when reducing
the batch size below 30 samples. But very small batches are undesirable from a practical
perspective, due to the increase in the number of AL cycles and the overall turnaround time

- GP works well even with heavy noise The confinement of
chemical space is essential for a small number of samples to be able to represent a sizable
pool

- **Understanding the relationships between data, models, and selection strategies in AL
pipelines paves the way to establishing protocols for choosing these parameters in an automated fashion. Using the findings from this work to parameterize AL pipelines based on
distances between data points and the retrospective evaluation of the models performance
is a promising avenue for future work.**




#### Large-Scale Pretraining Improves Sample Efficiency of Active Learning-Based Virtual Screening
- The study demonstrates that transformer-based language models (MoLFormer) and graph neural networks (MolCLR) can serve as more sample-efficient surrogate models in the Bayesian optimization active learning framework for virtual screening.

- Utilizing a Bayesian optimization active learning framework, the study implements a batched Bayesian optimization system called MolPAL, comprising a surrogate model, an acquisition function, and an objective function.

- The pretrained models, MoLFormer and MolCLR, are adapted within the MolPAL framework for virtual screening. MoLFormer is a molecular language transformer, while MolCLR is a molecular contrastive learning-pretrained graph isomorphism networ

- **MolPAL
consists of three components, including a surrogate model, an
acquisition function, and an objective function. The objective
function can be a docking protocol in structure-based drug
discovery. At the beginning of the virtual screening of a large
compound pool for potential hits to a protein target, a small
batch of molecules are randomly selected and docked. The
docking scores are used to train the surrogate model in a
supervised manner such that the surrogate model can be used
to predict the docking scores of all of the other molecules in
the pool. The acquisition function evaluates the molecule pool
based on the prediction and further selects another batch of
molecules to augment the training data set.**

- The study also shows that smaller acquisition batch sizes can improve the top-k molecule retrieval rate. The greedy acquisition strategy is effective in most cases, while the effect of the UCB strategy varies on different surrogate models and data sets. The UCB strategy can increase the diversity of acquired molecules at the cost of reduced top-k retrieval rate.

- The methodology is extended to ligand-based virtual screening using Rapid Overlay of Chemical Structures (ROCS) to assess the performance of pretrained models in this domain

- For the Enamine 50k library, MoLFormer retrieves 78.36% of the top-500 molecules after five iterations of acquisition using the greedy strategy, outperforming other models. The UCB strategy slightly improves the top-500 retrieval rate of deep learning models but negatively impacts LightGBM and RF. The EF of MoLFormer is 13.2, exceeding the MolCLR (11.33), D-MPNN (11.15), LightGBM (11.57), and RF (9.09).

- For the larger Enamine HTS set, MoLFormer retrieves 92.24% of the top-1000 molecules by exploring only 1.2% of the whole data set, exceeding all other models. The D-MPNN and MolCLR consistently outperform LightGBM and RF.

- Smaller acquisition batch sizes improve the top-k retrieval rate when the total number of explored compounds remains the same. For example, after exploring 0.6% of the data set (12,855 molecules) using the greedy strategy, MoLFormer retrieves 81.58% of the top-1000 compounds with the 0.1% batch size, which is higher than 75.9% with the 0.2% batch size

- The study shows that adding a term of uncertainty with a small weight during acquisition can benefit the performance of the active learning, and the top-1000 retrieval rate peaks when the uncertainty weight (β) is 2.
