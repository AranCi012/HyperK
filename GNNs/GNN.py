import tabular_to_graph_with_constraints as tg
import os 
import pandas as pd
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve,auc
import seaborn as sns
from collections import Counter



def print_class_distribution(data, set_name):
    count_dict = {0: 0, 1: 0}
    for graph in data:
        count_dict[graph.y.item()] += 1
    print(f"Distribuzione delle classi in {set_name}: {count_dict}")




class SpectralGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpectralGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_weight)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)

        x = self.fc(x)
        return torch.sigmoid(x)
    

def graph_to_pyg_data(graph):
    num_nodes = graph.number_of_nodes()
    node_features = []
    # Calcola il grado dei nodi
    degrees = dict(graph.degree(weight=None))  # grado non pesato
    weighted_degrees = dict(graph.degree(weight='weight'))  # grado pesato (somma dei pesi degli archi)
    # Calcola la centralità di grado, betweenness e closeness (proprietà topologiche)
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph, normalized=True, weight='weight')
    closeness_centrality = nx.closeness_centrality(graph)
    # Creiamo le feature per ogni nodo
    for node in graph.nodes:
        features = [
            degrees[node],                # Grado (connettività locale)
            weighted_degrees[node],       # Grado pesato (somma pesi archi)
            degree_centrality[node],      # Centralità di grado
            betweenness_centrality[node], # Centralità di betweenness
            closeness_centrality[node]    # Centralità di closeness
        ]
        node_features.append(features)
    x = torch.tensor(node_features, dtype=torch.float)
    # Ottieni gli edge_index (archi) dal grafo NetworkX
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    # Se il grafo ha pesi sugli archi, crea anche edge_attr
    if nx.get_edge_attributes(graph, 'weight'):
        edge_attr = torch.tensor([graph[u][v].get('weight', 1) for u, v in graph.edges], dtype=torch.float).unsqueeze(1)
    else:
        edge_attr = None
    # Crea il grafo PyTorch Geometric Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def process_graph_dict(graph_dict, label):
    data_list = []
    i=1
    for name, graph in graph_dict.items():
        print(f'processo il {i} grafo')
        data = graph_to_pyg_data(graph)
        data.y = torch.tensor([label], dtype=torch.long)
        data_list.append(data)
    return data_list

def get_graph_laplacian(A):

    # From https://towardsdatascience.com/spectral-graph-convolution-explained-and-implemented-step-by-step-2e495b57f801
    # Computing the graph Laplacian A is an adjacency matrix of some graph G

    N = A.shape[0] # number of nodes in a graph
    D = np.sum(A, 0) # node degrees
    D_hat = np.diag((D + 1e-5)**(-0.5)) # normalized node degrees
    L = np.identity(N) - np.dot(D_hat, A).dot(D_hat) # Laplacian
    return torch.from_numpy(L).float()


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        
        # Reshape target to match output dimensions
        data.y = data.y.view(-1, 1).float()  # Assicurati che sia float per BCEWithLogitsLoss
        
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():  # Assicurati di non calcolare i gradienti durante la valutazione
        for data in loader:
            output = model(data)
            data.y = data.y.view(-1, 1).float() 
            loss = criterion(output, data.y)
            total_loss += loss.item()
            
            # Applica la sigmoide per convertire l'output in probabilità
            pred = torch.sigmoid(output) >= 0.5  # Se output >= 0.5, pred è 1, altrimenti 0
            
            # Confronta con il target
            correct += pred.eq(data.y).sum().item()

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    
    return accuracy, avg_loss

def evaluate_model_on_test(model, test_loader):
    model.eval()  # Setta il modello in modalità valutazione
    y_true = []
    y_pred = []
    y_scores = []  # Per la ROC

    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            data.y = data.y.view(-1, 1).float()
            
            # Aggiungi i veri valori di etichetta a y_true
            y_true.extend(data.y.cpu().numpy())

            # Predizioni
            y_scores.extend(torch.sigmoid(out).cpu().numpy())  # Probabilità per la classe 1
            predicted = (torch.sigmoid(out) >= 0.5).long()  # Predici 1 se la probabilità è >= 0.5, altrimenti 0
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred, y_scores

def plot_confusion_matrix(y_true, y_pred, label_mapping):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_mapping.values(), 
                yticklabels=label_mapping.values())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def main():

    label_mapping = {0: "AD", 1: "NC"}
    
    AD_path="AD"
    NC_path="NC"

    graph_dict_AD=tg.create_graph_no_binary(AD_path) # Dizionario di 92 grafi
    graph_dict_NC=tg.create_graph_no_binary(NC_path) # Dizionario di 125 grafi
    pyg_data_list_AD = process_graph_dict(graph_dict_AD, label=0)  
    pyg_data_list_NC = process_graph_dict(graph_dict_NC, label=1) 

  
    full_data_list = pyg_data_list_AD + pyg_data_list_NC

    torch.save({'data': full_data_list, 'labels': [data.y.item() for data in full_data_list]}, 'graphs_and_labels_no_binary.pt')
    # Dividi in training, validation e test (esempio 70-15-15)
    

    '''
    loaded_data = torch.load('graphs_and_labels_no_binary.pt')

    # I grafi e le etichette sono salvati in due chiavi: 'data' e 'labels'
    full_data_list = loaded_data['data']  # Lista dei grafi
    labels = loaded_data['labels']  # Lista delle etichette
    print(f"Numero di grafi caricati: {len(full_data_list)}")
    print(f"Etichette: {labels}")

    # Controllo delle dimensioni
    if len(full_data_list) != len(labels):
        raise ValueError("Il numero di grafi e etichette non corrisponde!")

    # Associa le etichette ai grafi
    for i, graph in enumerate(full_data_list):
        graph.y = torch.tensor([labels[i]], dtype=torch.long)

    # Divisione dei dati con stratificazione
    train_data, test_data = train_test_split(full_data_list, test_size=0.1, random_state=42, stratify=labels)

    # Controlla se test_data ha almeno un campione prima di procedere
    if len(test_data) > 0:
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42, stratify=labels[:len(test_data)])
    else:
        raise ValueError("test_data è vuoto dopo la prima divisione!")

    # Crea i DataLoader
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    print_class_distribution(train_data, "train")
    print_class_distribution(val_data, "validation")
    print_class_distribution(test_data, "test")
    

    input_dim = train_data[0].num_node_features
    hidden_dim = 64
    output_dim = 1
    model = SpectralGNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    class_weights = torch.tensor([125/92])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    #pos_weight=class_weights)    


    # Addestramento del modello
    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_acc, val_loss = test(model, val_loader, criterion)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Inferenza sul set di test
    y_true, y_pred, y_scores = evaluate_model_on_test(model, test_loader)

    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, label_mapping)

    # ROC Curve
    plot_roc_curve(y_true, y_scores)
'''

if __name__ == "__main__":
    main()
