import torch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import KarateClub
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import torch.nn.functional as F
from gcn_conv_sum import GCNConv_sum
from gcn_conv_no_propagate import GCNConv_without_prop


'''SETUP'''
# Set random seed for reproducibility
torch.manual_seed(12345)
np.random.seed(12345)

'''LOADING DATA'''
dataset = KarateClub()
data = dataset[0]
def split_data(data,split):
    
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Randomly split the nodes
    indices = np.random.permutation(num_nodes)
    train_indices = indices[:int(split* num_nodes)]
    test_indices = indices[int(split * num_nodes):]
    
    # Assign masks
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    # Assign masks to the dataset
    data.train_mask = train_mask
    data.test_mask = test_mask


'''MODELS'''
'''regular GCN model'''
class GCN(torch.nn.Module): #for grid search
    def __init__(self, hidden_channels=8, dropout=0.5):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)
        self.classifier = Linear(2, dataset.num_classes)
        self.dropout = dropout  # Store dropout probability

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)  # Apply dropout
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)  # Apply dropout again
        h = h.tanh()
        out = self.classifier(h)
        return out, h


class GCN_2LAYERS(torch.nn.Module):
    def __init__(self, hidden_channels=8, aggr='sum'):
        super(GCN_2LAYERS, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels, aggr=aggr, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, 2, aggr=aggr, add_self_loops=True)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h
    
class GCN_5LAYERS(torch.nn.Module):
    def __init__(self, hidden_channels=8, aggr='max'):
        super(GCN_5LAYERS, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 24, aggr=aggr, add_self_loops=True)
        self.conv2 = GCNConv(24, 12, aggr=aggr, add_self_loops=True)
        self.conv3 = GCNConv(12, 6, aggr=aggr, add_self_loops=True)
        self.conv4 = GCNConv(6, 4, aggr=aggr, add_self_loops=True)
        self.conv5 = GCNConv(4, 2, aggr=aggr, add_self_loops=True)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        h = self.conv5(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h
    

'''GCN model without propagation'''
class GCN_2LAYERS_NO_PROPAGATE(torch.nn.Module):
    def __init__(self, hidden_channels=8, aggr='max'):
        super(GCN_2LAYERS_NO_PROPAGATE, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv_without_prop(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv_without_prop(hidden_channels, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h 

class GCN_4LAYERS_NO_PROPAGATE(torch.nn.Module):
    def __init__(self, hidden_channels=8, aggr='max'):
        super(GCN_4LAYERS_NO_PROPAGATE, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv_without_prop(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv_without_prop(hidden_channels, 24)
        self.conv3 = GCNConv_without_prop(24, 12)
        self.conv4 = GCNConv_without_prop(12, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h 


'''GCN model with sum'''    
class GCN_2LAYERS_SUM(torch.nn.Module):
    def __init__(self, hidden_channels=8, aggr='max'):
        super(GCN_2LAYERS_SUM, self).__init__()
        torch.manual_seed(12345)
        self.first_fc = Linear(dataset.num_features, 34)
        self.conv1 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.conv2 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.class_prepare = Linear(34, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.first_fc(x)
        h = self.conv1(h, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.class_prepare(h)
        h = h.tanh()
        out = self.classifier(h)
        return out, h   
        
class GCN_4LAYERS_SUM(torch.nn.Module):
    def __init__(self, hidden_channels=8, aggr='max'):
        super(GCN_4LAYERS_SUM, self).__init__()
        torch.manual_seed(12345)
        self.first_fc = Linear(dataset.num_features, 34)
        self.conv1 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.conv2 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.conv3 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.conv4 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.class_prepare = Linear(34, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.first_fc(x)
        h = self.conv1(h, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index) 
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        h = self.class_prepare(h)
        h = h.tanh()
        out = self.classifier(h)
        return out, h


class GCN_2LAYERS_SUM_WITH_DROPOUT(torch.nn.Module):
    def __init__(self, hidden_channels=8, aggr='max', dropout=0.5):
        super(GCN_2LAYERS_SUM_WITH_DROPOUT, self).__init__()
        torch.manual_seed(12345)
        self.first_fc = Linear(dataset.num_features, 34)
        self.conv1 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.conv2 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.class_prepare = Linear(34, 2)
        self.classifier = Linear(2, dataset.num_classes)
        self.dropout = dropout  # Store dropout probability

    def forward(self, x, edge_index):
        h = self.first_fc(x)
        h = F.dropout(h, p=self.dropout, training=self.training)  # Apply dropout
        h = self.conv1(h, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.class_prepare(h)
        h = h.tanh()
        out = self.classifier(h)
        return out, h   
       
class GCN_4LAYERS_SUM_WITH_DROPOUT(torch.nn.Module):
    def __init__(self, hidden_channels=8, aggr='max', dropout=0.5):
        super(GCN_4LAYERS_SUM_WITH_DROPOUT, self).__init__()
        torch.manual_seed(12345)
        self.first_fc = Linear(dataset.num_features, 34)
        self.conv1 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.conv2 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.conv3 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.conv4 = GCNConv_sum(34, 34, aggr=aggr, add_self_loops=True)
        self.class_prepare = Linear(34, 2)
        self.classifier = Linear(2, dataset.num_classes)
        self.dropout = dropout  # Store dropout probability

    def forward(self, x, edge_index):
        h = self.first_fc(x)
        h = F.dropout(h, p=self.dropout, training=self.training)  # Apply dropout
        h = self.conv1(h, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index) 
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        h = self.class_prepare(h)
        h = h.tanh()
        out = self.classifier(h)
        return out, h

    
'''TRAINING AND EVALUATION'''
def train_and_eval(data, model, split= 0.8, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    losses = []
    accuracies = []
    epochs = 10000
    for epoch in range(epochs):
        model.train()
        # split for Monte Carlo Cross-Validation
        split_data(data, split)
        optimizer.zero_grad()
        out, h = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            losses.append(loss.item())
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
            
        # Evaluate 
        model.eval()
        with torch.no_grad():
            out, embeddings = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / data.test_mask.sum().item()
            if epoch % 1000 == 0:
                accuracies.append(acc)
                print(f'Epoch {epoch:03d}, Accuracy: {acc:.2%}')

    mean_accuracy = torch.tensor(accuracies).mean().item()

    return embeddings, pred, losses, mean_accuracy

def perform_grid_search(data):
    param_grid = {
        'hidden_channels': [8, 16],
        'learning_rate': [0.01, 0.001],
        'weight_decay': [0, 0.0001, 0.001],
        'dropout': [0, 0.3, 0.5]
    }
    
    # Perform grid search
    best_acc = 0
    best_params = None
    best_predictions = None
    best_model_state = None
    best_losses = None
    best_embedding = None
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Evaluating parameters: {param_dict}")
        
        # Create model with current hyperparameters
        model = GCN(hidden_channels=param_dict['hidden_channels'], dropout=param_dict['dropout'])
        
        # Train and evaluate the model
        embeddings, predictions, losses, val_acc = train_and_eval_model(data, model, split=0.8)
        
        print(f"Validation Accuracy: {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = param_dict
            best_predictions = predictions
            best_model_state = model.state_dict()
            best_losses = losses
            best_embedding = embeddings
            
    # Print the best result
    print("\nBest Hyperparameters:")
    print(best_params)
    print(f"Best Validation Accuracy: {best_acc:.4%}")

    return best_embedding, best_predictions, best_losses, best_params, best_acc


'''VISUALIZATION'''
def visualize_losses(losses, ax):
    ax.plot(losses, label='Loss')
    ax.set_title('Training Loss vs epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

def visualize_embeddings(embeddings, labels, ax):
    tsne = TSNE(n_components=2, random_state=12345)
    reduced_embeddings = tsne.fit_transform(embeddings.detach().numpy())
    
    scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                         c=labels, cmap='Set1', s=100)
    # Create a color bar with only 4 distinct colors
    cbar = plt.colorbar(scatter, ax=ax, boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Class 0', 'Class 1', 'Class 2', 'Class 3'])  # Set custom tick labels
    ax.set_title("t-SNE visualization of node embeddings")

def visualize_classification_results(data, pred, acc, ax_gt, ax_pred):
    # Convert to NetworkX graph for visualization
    G = to_networkx(data, to_undirected=True)
    
    # Ground truth
    ax_gt.set_title("Ground Truth")
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=42),
        with_labels=True,
        node_color=data.y.numpy(),
        cmap="Set2",
        ax=ax_gt,
    )
    
    # Model predictions
    ax_pred.set_title(f"Model Predictions (Accuracy: {acc:.2%})")
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=42),
        with_labels=True,
        node_color=pred,
        cmap="Set2",
        ax=ax_pred,
    )

def visualize_results(losses, data, pred, acc, embeddings, labels, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle(f"Model: {model_name}", fontsize=16)
    visualize_losses(losses, axes[0, 0])
    visualize_embeddings(embeddings, labels, axes[0, 1])
    visualize_classification_results(data, pred, acc, axes[1, 0], axes[1, 1])
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()


'''MAIN FUNCTION'''
def main(data):
    # Print dataset information
    print("Dataset Information:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Average node degree: {data.edge_index.size(1) / data.num_nodes:.2f}")
    print(f"Number of node features: {data.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
    print(f"Contains self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}\n")
    
    models = {
        'Regular GCN - 2 layers ': GCN_2LAYERS,
        'Regular GCN - 5 layers': GCN_5LAYERS,
        'GCN Without Propagation- 2 layers': GCN_2LAYERS_NO_PROPAGATE,
        'GCN Without Propagation- 4 layers': GCN_4LAYERS_NO_PROPAGATE,
        'GCN With Sum - 2 layers': GCN_2LAYERS_SUM,
        'GCN With Sum - 4 layers': GCN_4LAYERS_SUM,
        'GCN With Sum - 2 layers and dropout': GCN_2LAYERS_SUM_WITH_DROPOUT,
        'GCN With Sum - 4 layers and dropout': GCN_4LAYERS_SUM_WITH_DROPOUT,
        # 'grid search': GCN
    }
    splits = [0.8, 0.5]
    learning_rate = [0.001, 0.01, 0.1]
    aggregations = ['max', 'sum', 'mean']
    if 'grid search' in models:
        print("Performing Grid Search:")
        embeddings, predictions, losses, best_params, best_acc = perform_grid_search(data)
        visualize_results(losses, data, predictions, best_acc, embeddings, data.y, 'Grid Search', 0.8)
    else:    
        for model_name, model_class in models.items():
            for split in splits:
                 for lr in learning_rate:
                     for aggr in aggregations:
                        print(f"Training and Evaluating {model_name}")
                        model = model_class()
                        embeddings, predictions, losses, acc= train_and_eval(data, model)

                        # Visualize results
                        visualize_results(losses, data, predictions, acc, embeddings, data.y, model_name)

if __name__ == "__main__":
    dataset = KarateClub()
    data = dataset[0]
    main(data)
 