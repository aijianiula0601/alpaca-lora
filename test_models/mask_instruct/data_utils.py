import os
import sys


def sota_prompt(human_name, bot_name, background):
    return f"Here is a conversation between {human_name} and {bot_name} related to the description below. {background}\n\n"


def gpt4_prompt(human_name, bot_name, background):
    return f"Below is an chat between {human_name} and {bot_name}.\n\n"


def persona_chat_prompt(human_name, bot_name, background):
    return f"{background}\n\n The above describes the state information of {bot_name} from the perspective of first person. The following is a conversation between {human_name} and {bot_name}.\n\n"


def empathetic_dialogues_prompt(human_name, bot_name, background):
    return f"Here is a conversation between {human_name} and {bot_name}.\n"


def stanford_52k_prompt(human_name, bot_name, background=""):
    if background == "":
        return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    else:
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"


def get_dataset_prompt(dataset_name, human_name, bot_name, background):
    if dataset_name == "sota":
        return sota_prompt(human_name, bot_name, background)

    elif dataset_name == "gpt4":
        return gpt4_prompt(human_name, bot_name, background)

    elif dataset_name == "persona_chat":
        return persona_chat_prompt(human_name, bot_name, background)

    elif dataset_name == "empathetic_dialogues":
        return empathetic_dialogues_prompt(human_name, bot_name, background)

    elif dataset_name == "stanford_52k":
        return stanford_52k_prompt(human_name, bot_name, background)
    else:
        raise Exception(f"Error dataset name:{dataset_name}")
