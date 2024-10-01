import torch
import torch.nn.functional as F
import numpy
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


og_file_path = 'activations/new/layer_31_ffn_output.pt'
og_tensor = read_tensors(og_file_path).cpu()
og_tensor = og_tensor[:5, :]

fnn_error = numpy.empty(0)
paths = ['bfp4/size_64/shape_1x64', 'bfp4/size_64/shape_8x8','bfp4/size_16/shape_1x16','bfp4/size_16/shape_4x4']
titles = ['1x64','8x8','1x16','4x4']
for i,path in enumerate(paths):
    quant_file_path = 'quantize/'+path+'/layer_31_ffn_output.pt'
    quant_tensor = read_tensors(quant_file_path).cpu()
    quant_tensor = quant_tensor.view(5, 512 * 4096)
    # print(quant_tensor.shape)

    error=F.mse_loss(og_tensor, quant_tensor)
    fnn_error = numpy.append(fnn_error,error)
    # print(fnn_error)


print(fnn_error.shape)
save_path = 'plots/next/quant_bfp4.png'

# paths = ['_attention_input.pt', '_attention_output.pt', '_ffn_input.pt', '_ffn_output.pt']
# titles = ['Attention Input', 'Attention Output', 'FFN Input', 'FFN Output']

X = numpy.array([64,64,16,16])
print(X.shape)
fig, ax = plt.subplots()

plt.scatter(X, fnn_error)

# Add labels to each point
for i, label in enumerate(titles):
    plt.text(X[i], fnn_error[i], label, fontsize=12, ha='right')
plt.xscale('log')

# ax.plot(X, fnn_error, marker='o', linestyle='-', color='r', label='FFN layers')

# Add labels and title
ax.set_xlabel('Size')
ax.set_ylabel('MSE')
ax.set_title('BFP4 Quantization errors for different Group sizes and shapes')

# Add a legend
ax.legend()

os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close(fig)
# print(f"Weight plots for Layer {layer_id} saved to {save_path}")
