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


attention_error = numpy.empty(0)
fnn_error = numpy.empty(0)
for layer_id in range(32):
    paths = ['_attention_input.pt', '_attention_output.pt', '_ffn_input.pt', '_ffn_output.pt']
    titles = ['Attention Input', 'Attention Output', 'FFN Input', 'FFN Output']

    og_file_path = 'activations/new/layer_'+str(layer_id)+'_attention_output.pt'
    og_tensor = read_tensors(og_file_path).cpu()
    print(og_tensor.shape)
    og_tensor = og_tensor[:5, :]
    print(og_tensor.shape)

    quant_file_path = 'quantize/bfp16/size_4096/shape_1x4096/layer_'+str(layer_id)+'_attention_output.pt'
    quant_tensor = read_tensors(quant_file_path).cpu()
    print(quant_tensor.shape)
    quant_tensor = quant_tensor.view(5, 512 * 4096)
    print(quant_tensor.shape)
    
    error=F.mse_loss(og_tensor, quant_tensor)
    print(error.shape)
    numpy.append(attention_error, error)

    og_file_path = 'activations/new/layer_'+str(layer_id)+'_ffn_output.pt'
    og_tensor = read_tensors(og_file_path).cpu()
    print(og_tensor.shape)
    og_tensor = og_tensor[:5, :]
    print(og_tensor.shape)

    quant_file_path = 'quantize/bfp16/size_4096/shape_1x4096/layer_'+str(layer_id)+'_ffn_output.pt'
    quant_tensor = read_tensors(quant_file_path).cpu()
    print(quant_tensor.shape)
    quant_tensor = quant_tensor.view(5, 512 * 4096)
    print(quant_tensor.shape)

    error=F.mse_loss(og_tensor, quant_tensor)
    print(error.shape)
    numpy.append(fnn_error,error)

print(attention_error.shape)
print(fnn_error.shape)
save_path = 'plots/next/attention_outputs.png'

# paths = ['_attention_input.pt', '_attention_output.pt', '_ffn_input.pt', '_ffn_output.pt']
# titles = ['Attention Input', 'Attention Output', 'FFN Input', 'FFN Output']

X = numpy.arange(0, 32, 1)
print(X.shape)
fig, ax = plt.subplots()
ax.plot(X, attention_error, marker='o', linestyle='-', color='b', label='Attention layers')
ax.plot(X, fnn_error, marker='o', linestyle='-', color='b', label='FFN layers')

# Add labels and title
ax.set_xlabel('Layers')
ax.set_ylabel('MSE')
ax.set_title('Quantization errors at different layers')

# Add a legend
ax.legend()

os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close(fig)
# print(f"Weight plots for Layer {layer_id} saved to {save_path}")
