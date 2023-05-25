# Identifying Ambiguous Similarity Conditions via Semantic Matching

The code repository for "Identifying Ambiguous Similarity Conditions via Semantic Matching" (Accepted by CVPR 2022) in PyTorch. 

## Main idea of DISCOVERNET
Rich semantics inside an image result in its ambiguous relationship with others, i.e., two images could be similar in one condition but dissimilar in another. By organizing instances into triplets, Weakly Supervised Conditional Similarity Learning (WS-CSL) learns multiple embeddings to match those semantic conditions without explicit condition labels. However, similarity relationships in a triplet are ambiguous except providing a condition. For example, it's difficult to say whether ‘aircraft’ is similar to ‘bird’ or ‘train’ without a ‘can fly’ or a ‘vehicle’ condition. To this end, by predicting the comparison's correctness after assigning the learned embeddings to their optimal conditions, we introduce a novel criterion to evaluate how much WS-CSL could cover latent semantics as the supervised model. Furthermore, we propose the Distance Induced Semantic COndition VERification Network (DISCOVERNET), which characterizes the instance-instance and triplets-condition relations in a ‘decompose-and-fuse’ manner.  To make the learned embeddings cover all semantics, DISCOVERNET utilizes a set module or an additional regularizer over the correspondence between a triplet and a condition. DISCOVERNET achieves state-of-the-art performance on benchmarks like UT-Zappos-50k and Celeb-A w.r.t. different criteria.

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.6 and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)

- Dataset: please download the dataset and put images into a folder data, and modify the data path in **triplet_image_loader.py**. 


## Model Training and Evaluation
Please use **train_balseline_new.py** and follow the instructions below. 


## Training scripts for DISCOVERNET

For example, to train Discovernet with the set module on the zapppos dataset: 

    $ python train_balseline_new.py --dataset zapppos --pi 


to train Discovernet with the semantic regularization on the celeba dataset: 

    $ python train_balseline_new.py --dataset celeba --HIK
