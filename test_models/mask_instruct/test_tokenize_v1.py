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
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    tokenized_ids = tokenized['input_ids']
    return tokenized_ids, len(tokenized_ids)


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

    input_ids = copy.deepcopy(head_ids)

    for i in range(turn_n):
        cur_turn_qa = conversation_dic['qas'][f'turn_{i}']
        if cur_turn_qa is None:
            break
        cur_question_string = human_name + ": " + cur_turn_qa["question"] + DEFAULT_EOS_TOKEN
        cur_question_string_token_ids, cur_question_string_token_ids_len = _tokenize_string(cur_question_string)
        cur_answer_string = bot_name + ": " + cur_turn_qa["answer"] + DEFAULT_EOS_TOKEN
        cur_answer_string_token_ids, cur_answer_string_token_ids_len = _tokenize_string(cur_answer_string)

        if header_ids_len + default_segment_token_ids_len + cur_question_string_token_ids_len + default_segment_token_ids_len + cur_answer_string_token_ids_len > token_max_len:
            break

        # question
        input_ids.extend(default_segment_token_ids)
        input_ids.extend(cur_question_string_token_ids)
        header_ids_len += default_segment_token_ids_len + cur_question_string_token_ids_len

        # answer
        input_ids.extend(default_segment_token_ids)
        input_ids.extend(cur_answer_string_token_ids)
        header_ids_len += default_segment_token_ids_len + cur_answer_string_token_ids_len

    label_ids = copy.deepcopy(input_ids)
    for i in range(len(head_ids)):
        label_ids[i] = IGNORE_INDEX
    attention_mask = [1 for _ in range(len(label_ids))]

    return {'input_ids': input_ids, 'labels': label_ids, 'attention_mask': attention_mask}


data_path = "/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/debug_multi_dataset_qas.json"
data = load_dataset("json", data_files=data_path)
train_data = data["train"].shuffle().map(_preprocess_example)

print('-' * 100)
for d in train_data:
    print(json.dumps(d))
    break
