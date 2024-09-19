import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Function to read tensors from a file
def read_tensors(file_path):
    # tensors = []
    # with open(file_path, 'r') as file:
    #     for line in file:
    #         # Remove unwanted characters and split by comma
    #         clean_line = line.replace('[', '').replace(']', '').strip()
    #         tensor_values = list(map(float, clean_line.split(',')))
    #         tensor = torch.tensor(tensor_values)
    #         tensors.append(tensor.squeeze())
    # return torch.stack(tensors)
    try:
        # Load the tensor from the file
        tensor_data = torch.load(file_path)
        return tensor_data
    except Exception as e:
        print(f"An error occurred while loading the tensor: {e}")
        return None

for layer_id in range(32):
    save_path = 'plots/weights/layer_'+str(layer_id)+'.png'
    fig, axs = plt.subplots(2, 4, subplot_kw={'projection' : '3d'}, figsize=(20,10))
    axs = axs.flatten()
    paths = ['_attention_wq.pt', '_attention_wk.pt', '_attention_wv.pt', '_attention_wo.pt', '_ffn_w1.pt', '_ffn_w2.pt', '_ffn_w3.pt']
    titles = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'FFN W1', 'FFN W2', 'FFN W3']
    for i in range(7):
        file_path = 'weights/layer_'+str(layer_id)+paths[i]
        tensor = torch.abs(read_tensors(file_path)).cpu().numpy()
        print(tensor.shape)
        X = np.arange(0, tensor.shape[1], 1)
        Y = np.arange(0, tensor.shape[0], 1) 
        X, Y = np.meshgrid(X, Y)
        axs[i].plot_surface(X, Y, tensor, cmap='viridis')
        axs[i].set_title(titles[i])
    fig.delaxes(axs[7])
    fig.suptitle(f"Layer {layer_id} Weights", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Weight plots for Layer {layer_id} saved to {save_path}")
