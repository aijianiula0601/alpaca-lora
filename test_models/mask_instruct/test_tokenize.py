import json

from datasets import load_dataset

from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

base_model = 'decapoda-research/llama-7b-hf'

tokenizer = LlamaTokenizer.from_pretrained(base_model)

cutoff_len: int = 256
add_eos_token: bool = False

train_on_inputs: bool = True  # if False, masks out inputs in loss
prompt_template_name = "alpaca"
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

    print(f"-----result:{result}")
    print("="*100)

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
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )


    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
                                              -100
                                          ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                user_prompt_len:
                                                                ]  # could be sped up, probably

    print('---------tokenized_full_prompt:', tokenized_full_prompt.keys())
    return tokenized_full_prompt


data_path = "/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/debug.json"
data = load_dataset("json", data_files=data_path)
train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

print('-' * 100)
for d in train_data:
    print(json.dumps(d))
    break
