import torch
from typing import Optional
import fire
from datasets import load_dataset
from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    # mantissa_length: int = 8,
    # exponent_length: int = 8,
    # group_shape_1: int = 1,
    # group_shape_2: int = 4096,
    input_length: int = 5,
    max_gen_len: Optional[int] = None,
):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    tokenizer = generator.tokenizer
    dataset = torch.tensor([tokenizer.encode("\n\n".join(test["text"]), bos=False, eos=False)])

    # def generate_text(input_text, max_length=50):
    #     input_ids = tokenizer.encode(input_text, bos=True, eos=False)
    #     input_ids = torch.tensor(input_ids).unsqueeze(0)
    #     output_ids = generator.generate(input_ids, max_gen_len=max_length)
    #     return tokenizer.decode(output_ids[0])

    # Open a file to write the generated texts
    bfp_sizes = [[8,8], [4,4], [2,2]]
    bfp_shapes_1 = [[4096,1],[64,64],[1,1024],[1024,1],[32,32],[1,256],[256,1],[16,16],[1,64],[64,1],[8,8],[1,16],[16,1],[4,4]]
    bfp_shapes_2 = [[1,64],[64,1],[8,8],[1,16],[16,1],[4,4]]
    bfp_shapes = [bfp_shapes_1, bfp_shapes_2, bfp_shapes_2]
    for j, emsize in enumerate(bfp_sizes):
        mantissa_length = emsize[0]
        exponent_length = emsize[1]
        for myshape in bfp_shapes[j]:
            group_shape_1 = myshape[0]
            group_shape_2 = myshape[1]
            for i in range(input_length):
                # if i >= 1:
                #     break
                prompt_tokens = dataset[:, (i * 512) : ((i + 1) * 512)].tolist()
                inputs = [mantissa_length, exponent_length, group_shape_1, group_shape_2]
                print(inputs)
                generated_ids = generator.generate(prompt_tokens, 512, inputs)
                generated_text = [generator.tokenizer.decode(ids) for ids in generated_ids]

if __name__ == "__main__":
    fire.Fire(main)
