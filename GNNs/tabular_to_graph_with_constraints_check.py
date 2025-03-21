import os 
import pandas as pd
import numpy as np
#import networkx as nx
#import torch
import matplotlib.pyplot as plt
#from torch_geometric.data import Data
#from torch_geometric.utils import from_networkx

def normalize_with_constraints(path):
    modified_dir = os.path.join(path, "modified_files")
    os.makedirs(modified_dir, exist_ok=True)  

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
                df = df.applymap(lambda x: 1 if x > 0.3 else 0)
                np.fill_diagonal(df.values, 0)
                new_file_path = os.path.join(modified_dir, file)
                df.to_csv(new_file_path, index=False, header=False)
            except UnicodeDecodeError as e:
                print(f"Errore nella lettura del file {file_path}: {e}")


def remove_bullshit(path):
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith("modified.csv"):
                file_path = os.path.join(root, file)
                os.remove(file_path)  

def create_graph_no_binary(path):
    graphs = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.DS' not in file:  # Ignora file di sistema di macOS
                print(f'Processando {file}')
                file_path = os.path.join(root, file)
                
                # Carica il CSV
                df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1') 
                
                # Applica la trasformazione ai valori, rendendoli positivi
                df = df.applymap(lambda x: abs(x) if isinstance(x, (int, float)) else x)

                print(f'Matrice di adiacenza per {file}:')
                
                # Crea un grafo vuoto
                G = nx.Graph()
                
                # Aggiungi tutti i nodi (anche senza archi)
                num_nodes = df.shape[0]  # Supponiamo che la matrice sia quadrata
                G.add_nodes_from(range(num_nodes))
                
                # Aggiungi gli archi per le connessioni
                for i in range(df.shape[0]):
                    for j in range(i + 1, df.shape[1]):  # Consideriamo solo la parte triangolare superiore per evitare doppioni
                        if df.iloc[i, j] != 0:  # Aggiungi un arco se il valore non è zero
                            G.add_edge(i, j, weight=df.iloc[i, j])  # Aggiungi il peso se necessario
                
                print(f'Nodi nel grafo per {file}: {G.nodes()}')
                print(f'Numero di nodi: {len(G.nodes())}, Numero di archi: {len(G.edges())}')
                graphs[file] = G  # Salva il grafo
    return graphs





def create_single_graph_from_binary_matrix(path):
    df = pd.read_csv(path, header=None, encoding='ISO-8859-1')
    G = nx.Graph() 
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if df.iloc[i, j] == 1:  
                G.add_edge(i, j)
    plt.figure(figsize=(8, 8))  
    pos = nx.spring_layout(G, seed=42)  
    nx.draw(
        G, 
        pos, 
        with_labels=False,  
        node_size=10,  
        node_color='black',  
        edge_color='red',
        width=0.5 
    )  
    plt.title("Grafo binario", fontsize=16)
    plt.show()
    


def create_graph_from_binary_matrix(path):
    graphs = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.DS' not in file:
                print(f'Processando {file}')
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1') 
                print(f'Matrice di adiacenza per {file}:')
                # Crea un grafo vuoto
                G = nx.Graph()
                
                # Aggiungi tutti i nodi (anche senza archi)
                num_nodes = df.shape[0]  # Supponiamo che la matrice sia quadrata
                G.add_nodes_from(range(num_nodes))
                
                # Aggiungi gli archi per le connessioni
                for i in range(df.shape[0]):
                    for j in range(i + 1, df.shape[1]):  # Consideriamo solo la parte triangolare superiore per evitare doppioni
                        if df.iloc[i, j] == 1:  
                            G.add_edge(i, j)
                
                print(f'Nodi nel grafo per {file}: {G.nodes()}')
                print(f'Numero di nodi: {len(G.nodes())}, Numero di archi: {len(G.edges())}')
                graphs[file] = G  # Salva il grafo
    return graphs

def plot_graphs(graphs):

    '''
       input: dictionary of graphs
       output -> plot of graphs
    ''' 

    for file_name, G in graphs.items():
        plt.figure(figsize=(8, 8))  
        pos = nx.spring_layout(G, seed=42)  
        nx.draw(
            G, 
            pos, 
            with_labels=False,  
            node_size=10,  
            node_color='black',  
            edge_color='red',
            width=0.5 
        )  
        plt.title("Grafo binario {file_name}", fontsize=16)
        plt.show()

def save_graphs_as_imgs(path,graphs):
    '''
       input: graphs-> dictionary of graphs
       output -> plot of graphs
    ''' 
    img_dir = os.path.join(path, "images")
    os.makedirs(img_dir, exist_ok=True) 
    for file_name, G in graphs.items():
        plt.figure(figsize=(8, 8))  
        pos = nx.spring_layout(G)  
        nx.draw(
            G, 
            pos, 
            with_labels=False,  
            node_size=10,  
            node_color='blue',  
            edge_color='red',
            width=0.5 
        )  
        img_path = os.path.join(img_dir, f"{file_name}.png")
        plt.savefig(img_path, format="PNG")
        plt.close()
    print(f"Tutti i grafi sono stati salvati come immagini in {img_dir}.")
    
def save_events_to_csv(df, output_dir):
    """
    Salva ogni riga di un dataframe come un file CSV separato.

    Args:
        df (pd.DataFrame): Il dataframe contenente gli eventi simulati.
        output_dir (str): La directory in cui salvare i file CSV.

    Returns:
        None
    """
    # Crea la directory se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Itera su ogni riga del dataframe
    for idx, row in df.iterrows():
        event_data = row.to_frame().transpose()  # Converte la riga in un dataframe
        file_name = f"event_{idx + 1}.csv"  # Nome del file CSV
        file_path = os.path.join(output_dir, file_name)
        
        # Salva il dataframe come CSV
        event_data.to_csv(file_path, index=False)
    
    print(f"Salvati {len(df)} eventi nella directory '{output_dir}'.")

def main():
    # Percorso del file di input
    path = "HyperK/Hyperk/shared/100k_ranvtx_ranmom_0_1000_pandas/out.ext.pandas.0"
    
    # Carica il file in un DataFrame
    try:
       df = pd.read_pickle(path)
    except Exception as e:
        print(f"Errore nel leggere il file: {e}")
        return
    
    # Directory di output per i file CSV
    output_dir = "Hyperk/hk/pandas_to_csv"
    
    # Salva gli eventi come file CSV
    save_events_to_csv(df, output_dir)

if __name__ == "__main__":
    main()