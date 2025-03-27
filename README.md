# **<u>Graph Convolutional Network</u>**
## project Overview
Graph Convolutional Networks (GCNs) have emerged as a powerful framework for analyzing graph-structured data. 
In this project, we implemented several GCN variants to classify nodes in the Zachary's Karate Club dataset into their respective communities. 
This dataset, a widely studied example in network science, represents interactions within a karate club, where nodes correspond to members and edges represent their social interactions.
To evaluate the effectiveness of our models, we performed various analyses, including hyperparameter optimization using grid search, comparing the performance of custom GCN variants, and different numbers of layers.
These explorations allowed us to gain a deeper understanding of the capabilities and design choices in GCNs.
When you run the program, it will sequentially execute various models we explored, using different data splits and configurations with either 2 or 4 layers.

## Dataset Overview
The Zachary's Karate Club dataset is loaded using the: \texttt{torch\_geometric.datasets.KarateClub} module. 
Below are some key properties of the dataset:
    Number of Nodes: 34
    Number of Edges: 156
    Average Node Degree: 4.59
    Node Features: 34 
    Classes: 4 (community labels)
    Isolated Nodes: None
    Self-Loops: None
    Undirected Graph: Yes

## Model Implementation
We implemented a GCN using the PyTorch Geometric library and explored three different forward pass variations of the GCN convolution layer.
In order to do so we modified the message passing framework which is the core mechanism in GCN where nodes exchange information with their neighbors to update their representations.
This process involves aggregating features from neighboring nodes and combining them with the node's own features, enabling the model to capture both local and global graph structure. 
For each model, we experimented with 2 to 5 layers and tested various aggregation functions.
