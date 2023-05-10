import os
import sys
import json
import torch
import copy

from datasets import load_dataset

from transformers import LlamaForCausalLM, LlamaTokenizer

pdj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"pdj:{pdj}")
sys.path.append(pdj)

from utils.prompter import Prompter
from test_models.mask_instruct.data_utils import get_dataset_prompt

base_model = 'decapoda-research/llama-7b-hf'

tokenizer = LlamaTokenizer.from_pretrained(base_model)

cutoff_len: int = 2048
add_eos_token: bool = False

train_on_inputs: bool = True  # if False, masks out inputs in loss
prompt_template_name = "alpaca"
prompter = Prompter(prompt_template_name)
DEFAULT_EOS_TOKEN = "</s>"
IGNORE_INDEX = -100
DEFAULT_SEGMENT_TOKEN = "###"

token_max_len = cutoff_len

base_model = 'decapoda-research/llama-7b-hf'
tokenizer = LlamaTokenizer.from_pretrained(base_model)


def _tokenize_string(text):
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    print("tokenized:", tokenized)
    tokenized_ids = tokenized.input_ids[0]
    tokenized_ids_len = tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
    return tokenized_ids, tokenized_ids_len


def _preprocess_example(conversation_dic):
    """
    :param conversation_dic: example:{
        "background":"---",
        "human_name":"a",
        "bot_name":"b",
        "qas":{
            "turn_0":{"question":"---","answer":"---"},
            ...
            "turn_n":{"question":"---","answer":"---"}
        }
    }
    :param tokenizer: tokenizer model
    :return: dic
    """
    dataset_name = conversation_dic['dataset_name']
    default_segment_token_ids, default_segment_token_ids_len = _tokenize_string(DEFAULT_SEGMENT_TOKEN)

    turn_n = len(conversation_dic["qas"])
    human_name = conversation_dic['human_name']
    bot_name = conversation_dic['bot_name']
    header = get_dataset_prompt(dataset_name, human_name, bot_name, background=conversation_dic['background'])
    head_ids, header_ids_len = _tokenize_string(header)

    input_ids_tensor_list = [head_ids]

    for i in range(turn_n):
        cur_turn_qa = conversation_dic['qas'][f'turn_{i}']
        cur_question_string = human_name + ": " + cur_turn_qa["question"] + DEFAULT_EOS_TOKEN
        cur_question_string_token_ids, cur_question_string_token_ids_len = _tokenize_string(cur_question_string)
        cur_answer_string = bot_name + ": " + cur_turn_qa["answer"] + DEFAULT_EOS_TOKEN
        cur_answer_string_token_ids, cur_answer_string_token_ids_len = _tokenize_string(cur_answer_string)

        if header_ids_len + default_segment_token_ids_len + cur_question_string_token_ids_len + default_segment_token_ids_len + cur_answer_string_token_ids_len > token_max_len:
            break

        # question
        input_ids_tensor_list.append(default_segment_token_ids)
        input_ids_tensor_list.append(cur_question_string_token_ids)
        header_ids_len += default_segment_token_ids_len + cur_question_string_token_ids_len

        # answer
        input_ids_tensor_list.append(default_segment_token_ids)
        input_ids_tensor_list.append(cur_answer_string_token_ids)
        header_ids_len += default_segment_token_ids_len + cur_answer_string_token_ids_len

    input_ids = torch.cat(input_ids_tensor_list, dim=0)
    label_ids = copy.deepcopy(input_ids)
    label_ids[:len(head_ids)] = IGNORE_INDEX

    return input_ids, label_ids


data_path = "/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/debug_multi_dataset_qas.json"
data = load_dataset("json", data_files=data_path)
train_data = data["train"].shuffle().map(_preprocess_example)

print('-' * 100)
for d in train_data:
    print(json.dumps(d))
    break
