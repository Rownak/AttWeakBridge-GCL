from sklearn.manifold import TSNE
import seaborn as sns, pandas as pd;
import csv
from collections import defaultdict
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import config

def convert_keys_to_int(d):
    new_dict = {}
    for k, v in d.items():
        new_key = int(k) if k.isdigit() else k
        if isinstance(v, dict):
            new_dict[new_key] = convert_keys_to_int(v)
        else:
            new_dict[new_key] = v
    return new_dict


result_dir = config.OUTPUT_DIR
embeddings_dir = config.EMBEDDING_DIR
data_dir = config.DATA_DIR
print(result_dir)
with open(data_dir+'attack_weak_range.json') as fp:
    attack_weak_range = json.load(fp)

attack_size=attack_weak_range['attack'][1]
weakness_size=attack_weak_range['cwe'][1]-attack_size


from collections import defaultdict
sample = 10
dual = "dual3"
loss_func = "triplet"
plot_dir = result_dir+"plots/sample_{}/{}/".format(sample,loss_func)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(plot_dir+"src/"):
    os.makedirs(plot_dir+"src/")
models = ["pt_gpt2-xl", "gpt2-xl"]
gnn_models = ["GCN","GAT"]
features = ["text", "text_hop", "node2vec", "deepwalk"]
# gnn_models = ["GAN_embeddings","GCN_embeddings"]
graph_embeddings = {}
graph_embeddings["node2vec"]=np.load(embeddings_dir+"node2vec.npy")
graph_embeddings["deepwalk"]=np.load(embeddings_dir+"deepwalk.npy")
gnn_embeddings = defaultdict(list)
text_hop_embeddings = defaultdict(list)
text_hop_embeddings["bert"].append(np.load(embeddings_dir+"pt_bert/text_embeddings.npy"))
count=0
for model in models:
    text_emb_dir = embeddings_dir+model+"/"
    gcl_data_dir = result_dir+"gcl_data/"+model+"/"
    text_hop_embeddings["text"].append(np.load(text_emb_dir+"text_embeddings.npy"))
    text_hop_embeddings["hop_text"].append(np.load(text_emb_dir+"text_hop_embeddings.npy"))
    

    gnn_hop = []
    gnn_node2vec = []
    gnn_deepwalk = []
    gnn_text = []
    for j in range(len(gnn_models)):

        if(count==0):
            dual = ""
            # gnn_embeddings2=np.load(gcl_data_dir+"sample_{}/{}/{}/text_hop_gm_1.0.npy".format(sample,gnn_models[j],loss_func))
            # gnn_hop.append(gnn_embeddings2)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/node2vec_gm_1.0.npy".format(sample,gnn_models[j],loss_func))
            gnn_node2vec.append(gnn_embeddings3)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/deepwalk_gm_1.0.npy".format(sample,gnn_models[j],loss_func))
            gnn_deepwalk.append(gnn_embeddings3)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/text_gm_1.0.npy".format(sample,gnn_models[j],loss_func))
            gnn_text.append(gnn_embeddings3)
        else:
            dual = "dual3"
            # gnn_embeddings2=np.load(gcl_data_dir+"sample_{}/{}/{}/text_hop_{}_gm_1.0.npy".format(sample,gnn_models[j],loss_func,dual))
            # gnn_hop.append(gnn_embeddings2)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/text_node2vec_{}_gm_1.0.npy".format(sample,gnn_models[j],loss_func,dual))
            gnn_node2vec.append(gnn_embeddings3)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/text_deepwalk_{}_gm_1.0.npy".format(sample,gnn_models[j],loss_func,dual))
            gnn_deepwalk.append(gnn_embeddings3)

        
    
        # gnn_embeddings2.append(np.load(dir_name+"{}/sample_{}/hop_gm_1.0.npy".format(gnn_models[j],sample)))
        # gnn_emb_matrix2.append(cosine_similarity(gnn_embeddings2[-1]))
        # gnn_embeddings2.append(np.load(dir_name+"{}/sample_{}/node2vec_gm_1.0.npy".format(gnn_models[j],sample)))
        # gnn_emb_matrix2.append(cosine_similarity(gnn_embeddings2[-1]))
    # gnn_embeddings["hop_text"].append(gnn_hop)
    gnn_embeddings["node2vec"].append(gnn_node2vec)
    gnn_embeddings["deepwalk"].append(gnn_deepwalk)
    if(count==0):
        gnn_embeddings["text"].append(gnn_text)
    count+=1

top_attack = [64, 123, 144, 123, 64, 19, 123, 156, 150, 185, 41, 83, 57, 69, 177, 83, 41, 19, 177, 137, 177, 47, 123, 150, 123, 62, 170, 172, 123, 177, 41, 170, 111, 91, 50, 85, 62, 177, 123, 123, 123, 150, 150, 150, 137, 150, 177, 19, 167, 111, 19, 29, 62, 41, 108, 123, 96, 126, 108, 123, 146, 150, 158, 177, 46, 126, 150, 25, 150, 126, 167, 25, 123, 123, 167, 123, 46, 41, 167, 16, 177, 41, 106, 146, 45, 46, 177, 177, 134, 24, 83, 16, 170, 46, 16, 76, 126, 16, 150, 177, 46, 170, 41, 64, 144, 16, 46, 126, 170, 123, 41, 126, 170, 177, 62, 150, 143, 108, 29, 177, 126, 167, 156, 76, 134, 94, 177, 160, 146, 19, 50, 126, 30, 123, 19, 104, 170, 146, 143, 106, 105, 167, 146, 45, 156, 177, 9, 83, 137, 156, 123, 105, 41, 135, 113, 21, 146, 177, 41, 113, 156, 123, 113, 156, 86, 189, 19, 150, 123, 64, 150, 111, 76, 134, 146, 177, 43, 123, 41, 5, 57, 146, 170, 134, 150, 19, 69, 123, 167, 150, 177, 156, 156, 91, 24, 177, 185, 123, 177, 167]
top_cwe = [722, 651, 722, 1098, 689, 554, 1068, 536, 554, 722, 554, 509, 891, 498, 554, 487, 908, 616, 537, 722, 1098, 498, 1022, 444, 341, 700, 891, 407, 944, 536, 536, 1098, 616, 616, 358, 387, 722, 661, 554, 891, 1024, 437, 373, 1098, 689, 555, 944, 944, 444, 554, 341, 490, 725, 1098, 373, 555, 689, 1004, 515, 833, 342, 435, 407, 440, 446, 341, 833, 498, 509, 651, 838, 490, 1004, 446, 554, 908, 1098, 341, 833, 616, 373, 381, 341, 1002, 342, 944, 945, 425, 498, 536, 373, 446, 616, 722, 430, 616, 616, 1098, 908, 1023, 616, 437, 555, 616, 689, 437, 554, 617, 373, 616, 944, 944, 651, 703, 1125, 944, 492, 509, 498, 714, 446, 537, 381, 554, 616, 430, 662, 342, 1095, 722, 678, 554, 488, 667, 446, 358, 446, 1071, 498, 554, 891, 538, 446, 891, 981, 981, 554, 435, 797, 725, 517, 1076, 402, 1098, 944, 387, 1004, 437, 205, 908, 1125, 1071, 722, 703, 891, 944, 205, 945, 214, 1121, 440, 617, 722, 554, 317, 555, 425, 715, 491, 407, 1020, 370, 341, 205, 425, 1098, 554, 671, 539, 341, 908, 491, 402, 554, 425, 446, 944, 342, 341, 373]

def tsne_plot1(embeddings, filename):

    # First 203 items are attack tag, last 933 items are cwe tag
    tags = np.array(['attack'] * attack_size + ['cwe'] * weakness_size)
    
    # Perform t-SNE to reduce embeddings to 2D
    # tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(perplexity=25, n_components=2, init='pca', max_iter=5000, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot the t-SNE results
    plt.figure(figsize=(8,5))
    
    # Scatter plot for attack tag
    plt.scatter(embeddings_2d[tags == 'attack', 0], embeddings_2d[tags == 'attack', 1],
                c='orange', label='ATT&CK', alpha=0.6)
    
    # Scatter plot for cwe tag
    plt.scatter(embeddings_2d[tags == 'cwe', 0], embeddings_2d[tags == 'cwe', 1],
                c='blue', label='CWE', alpha=0.6)
    
    # Adding title and labels
    # plt.title("{}".format(filename))
    # plt.xlabel('t-SNE dimension 1', fontsize=14)
    # plt.ylabel('t-SNE dimension 2', fontsize=14)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Adding legend
    plt.legend(fontsize=18)
    fig = plt.gcf()
    # fig.set_size_inches(10, 6)
    plt.savefig(plot_dir+filename+"_tsne.png", dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()


def tsne_plot2(embeddings, filename):

    # First 203 items are attack tag, last 933 items are cwe tag
    tags = np.array(['attack'] * attack_size + ['cwe'] * weakness_size)
    
    # Perform t-SNE to reduce embeddings to 2D
    # tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(perplexity=25, n_components=2, init='pca', max_iter=5000, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot the t-SNE results
    plt.figure(figsize=(8,5))
    
    # Scatter plot for attack tag
    plt.scatter(embeddings_2d[tags == 'attack', 0], embeddings_2d[tags == 'attack', 1],
                c='orange', label='ATT&CK', alpha=0.6)
    
    # Scatter plot for cwe tag
    plt.scatter(embeddings_2d[tags == 'cwe', 0], embeddings_2d[tags == 'cwe', 1],
                c='blue', label='CWE', alpha=0.6)

    # Scatter plot for list_column1, purple color
    plt.scatter(embeddings_2d[top_attack, 0], embeddings_2d[top_attack, 1],
                c='red', label='top similar ATT&CK', alpha=0.6)
    
    # Scatter plot for list_column2, red color
    plt.scatter(embeddings_2d[top_cwe, 0], embeddings_2d[top_cwe, 1],
                c='purple', label='top similar CWE', alpha=0.6)
    
    # Adding title and labels
    # plt.title("{}".format(filename))
    # plt.xlabel('t-SNE dimension 1', fontsize=14)
    # plt.ylabel('t-SNE dimension 2', fontsize=14)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Adding legend
    plt.legend(fontsize=14)
    fig = plt.gcf()
    # fig.set_size_inches(10, 6)
    plt.savefig(plot_dir+filename+"_topsim_tsne.png", dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
def tsne_plot3(embeddings, filename):

    # First 203 items are attack tag, last 933 items are cwe tag
    tags = np.array(['attack'] * attack_size + ['cwe'] * weakness_size)
    # embeddings = embeddings[203:]
    # Perform t-SNE to reduce embeddings to 2D
    # tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(perplexity=25, n_components=2, init='pca', max_iter=5000, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Standardize features by removing the mean and scaling to unit variance
    embeddings_2d_scaled = StandardScaler().fit_transform(embeddings_2d)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.08, min_samples=3)  # adjust these parameters based on your data
    clusters = dbscan.fit_predict(embeddings_2d_scaled)

    # Plot the t-SNE results
    plt.figure(figsize=(8,5))
    

    unique_clusters = np.unique(clusters)
    print("number of clusters:",len(unique_clusters))
    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap('nipy_spectral', n_clusters)  # Get a colormap with sufficient unique colors
    # cmap = plt.cm.get_cmap('viridis', 256)  # Get the 'viridis' colormap with 256 colors
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap=cmap, alpha=0.6)
    
    # Adding title and labels
    # plt.title("{}".format(filename))
    # plt.xlabel('t-SNE dimension 1', fontsize=14)
    # plt.ylabel('t-SNE dimension 2', fontsize=14)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Adding legend
    # plt.legend(fontsize=18)
    fig = plt.gcf()
    # fig.set_size_inches(10, 6)
    plt.savefig(plot_dir+filename+"_tsne.png", dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()
    return clusters


def tsne_plot4(embeddings, filename, clusters):

    # First 203 items are attack tag, last 933 items are cwe tag
    tags = np.array(['attack'] * attack_size + ['cwe'] * weakness_size)
    
    # Perform t-SNE to reduce embeddings to 2D
    # tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(perplexity=25, n_components=2, init='pca', max_iter=5000, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8,5))
    unique_clusters = np.unique(clusters)
    print("number of clusters:",len(unique_clusters))
    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap('nipy_spectral', n_clusters)  # Get a colormap with sufficient unique colors
    # cmap = plt.cm.get_cmap('viridis', 256)  # Get the 'viridis' colormap with 256 colors
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap=cmap, alpha=0.6)
    
    # Adding title and labels
    # plt.title("{}".format(filename))
    # plt.xlabel('t-SNE dimension 1', fontsize=14)
    # plt.ylabel('t-SNE dimension 2', fontsize=14)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Adding legend
    plt.legend(fontsize=18)
    fig = plt.gcf()
    # fig.set_size_inches(10, 6)
    plt.savefig(plot_dir+filename+"_tsne.png", dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()


tsne_plot1(graph_embeddings["deepwalk"], "deepwalk")
tsne_plot2(graph_embeddings["deepwalk"], "deepwalk")
clusters = tsne_plot3(graph_embeddings["deepwalk"], "deepwalk_dbscan")
tsne_plot4(gnn_embeddings['deepwalk'][0][0], "pt_gpt2xl_gcn_deepwalk_dbscan", clusters)
