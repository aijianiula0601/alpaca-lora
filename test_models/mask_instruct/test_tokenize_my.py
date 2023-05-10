import os
import sys
import json

from datasets import load_dataset

from transformers import LlamaForCausalLM, LlamaTokenizer

pdj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"pdj:{pdj}")
sys.path.append(pdj)

from utils.prompter import Prompter

base_model = 'decapoda-research/llama-7b-hf'

tokenizer = LlamaTokenizer.from_pretrained(base_model)

cutoff_len: int = 256
add_eos_token: bool = False

train_on_inputs: bool = True  # if False, masks out inputs in loss
prompt_template_name = "conversation"
prompter = Prompter(prompt_template_name)


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_conversation_prompt(data_point["background"], data_point["qas"])
    tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)

    # mask background
    background_prompt = prompter.generate_conversation_prompt(data_point["background"])
    tokenized_user_prompt = tokenize(background_prompt, add_eos_token=add_eos_token)
    background_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        background_prompt_len -= 1

    tokenized_full_prompt["labels"] = [-100] * background_prompt_len + tokenized_full_prompt["labels"][
                                                                       background_prompt_len:
                                                                       ]  # could be sped up, probably
    return tokenized_full_prompt


data_path = "/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/conversation_mask_instruct/debug.json"
data = load_dataset("json", data_files=data_path)
train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

print('-' * 100)
for d in train_data:
    print(json.dumps(d))
    break
