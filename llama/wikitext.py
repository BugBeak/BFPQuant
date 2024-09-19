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
    with open('generated_texts.txt', 'w') as file:
        for i in range(40):
            # if i >= 1:
            #     break
            prompt_tokens = dataset[:, (i * 512) : ((i + 1) * 512)].tolist()
            generated_ids = generator.generate(prompt_tokens, 512)
            generated_text = [generator.tokenizer.decode(ids) for ids in generated_ids]
            file.write(f"Generated for entry {i+1}: {generated_text}\n")

if __name__ == "__main__":
    fire.Fire(main)
