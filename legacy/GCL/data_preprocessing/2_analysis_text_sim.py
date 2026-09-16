import pickle
import numpy as np
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import config

# models = ["pt_SecRoBERTa","SecRoBERTa","pt_SecureBERT","SecureBERT","pt_gpt2-xl","gpt2-xl"]
models = ["pt_SecRoBERTa"]
result_dir = config.OUTPUT_DIR

attack_sim_list = []
weak_sim_list = []
for i in range(len(models)):
    analysis_model = models[i]
    output_dir = result_dir+"gcl_data/"+analysis_model
    file1 = open(output_dir+'/attack_text.pkl','rb')
    file2 = open(output_dir+'/weak_text.pkl','rb')
    attack_sim_list.append(pickle.load(file1))
    weak_sim_list.append(pickle.load(file2))


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def histogram(cosine_sim_pairs, analysis_model, graph_name):
    # Extract the cosine similarity values from the filtered results
    cosine_sim_values = [pair[1] for row in cosine_sim_pairs for pair in row]
    # Define the bins for the histogram
    bins = np.arange(0, 1.1, 0.05)  # Bins from 0 to 1 with step size 0.1
    plt.figure(figsize=(5, 3))
    # Create the histogram
    plt.hist(cosine_sim_values, bins=bins, edgecolor='black')
    
    # Set the x-axis and y-axis labels
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    xt = np.arange(0,1.1,0.1)
    plt.xticks(xt)
    
    # Set the title of the histogram
    plt.title(graph_name+' Graph Sim. in '+analysis_model)
    analysis_model = models[i]
    output_dir = result_dir+"gcl_data/"+analysis_model
    plt.savefig(output_dir+f"/{graph_name}_graph_deepwalk_sim_histogram.png",dpi=200)
    # Show the plot
    # plt.show()

for i in range(len(models)):
    histogram(weak_sim_list[i], models[i], "CWE")

for i in range(len(models)):
    histogram(attack_sim_list[i], models[i], "ATTACK")

print("Saved in ", output_dir)