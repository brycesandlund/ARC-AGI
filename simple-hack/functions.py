import re

from enums import ModelType


def parse_completion(completion: str, model_type: ModelType) -> tuple[str, str]:
    """
    Parses a completion string that may contain <think>...</think> or <reasoning>...</reasoning> blocks.
    Extracts thinking/reasoning content and the final answer.
    
    Args:
        completion (str): The model's completion string.
        model_type (ModelType): The type of model, which determines the tags to use.
        
    Returns:
        tuple[str, str]: A tuple of (thinking_content, final_content).
    """
    if model_type == ModelType.INSTRUCT:
        if "####" in completion:
            parts = completion.split("####", 1)
            thinking_content = parts[0].strip()
            content = parts[1].strip()
            return thinking_content, content

        answer_start_tag = "<answer>"
        answer_end_tag = "</answer>"

        start_pos = completion.find(answer_start_tag)
        if start_pos != -1:
            end_pos = completion.find(answer_end_tag, start_pos)
            if end_pos != -1:
                thinking_content = completion[:start_pos].strip()
                content = completion[start_pos + len(answer_start_tag) : end_pos].strip()
                return thinking_content, content
        return "", ""

    if model_type == ModelType.THINKING:
        start_tag = "<think>"
        end_tag = "</think>"
    else:
        start_tag = "<reasoning>"
        end_tag = "</reasoning>"

    end_tag_pos = completion.rfind(end_tag)

    if end_tag_pos != -1:
        content = completion[end_tag_pos + len(end_tag) :].strip()

        # Extract thinking content
        think_part = completion[:end_tag_pos]
        start_tag_pos = think_part.rfind(start_tag)

        if start_tag_pos != -1:
            thinking_content = think_part[start_tag_pos + len(start_tag) :].strip()
        else:
            # If no start tag is found, assume everything before the end tag is thinking.
            thinking_content = think_part.strip()

        return thinking_content, content

    # No end tag found
    return "", ""