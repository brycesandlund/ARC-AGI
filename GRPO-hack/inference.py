from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

# load the tokenizer and the model
base_model_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

def generate_text_base(messages, max_new_tokens=500):
    text = base_model_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = base_model_tokenizer([text], return_tensors="pt").to(base_model.device)

    # conduct text completion
    generated_ids = base_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    output_ids = generated_ids[0].tolist()

    content = base_model_tokenizer.decode(output_ids, skip_special_tokens=False).strip("\n")

    return {
        "output_ids": output_ids,
        "content": content
    }

message = [
    {
        "content": "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.\n\nRespond in the format of the training output examples\n\n--Training Examples--\n--Example 0-- \n\n INPUT: \n\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 5, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 5, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 5, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n\n\nOUTPUT: \n\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 5, 1, 5, 0, 0, 0, 0]\n[0, 0, 1, 0, 1, 0, 0, 0, 0]\n[0, 0, 5, 1, 5, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 5, 1, 5, 0]\n[0, 0, 0, 0, 0, 1, 0, 1, 0]\n[0, 5, 1, 5, 0, 5, 1, 5, 0]\n[0, 1, 0, 1, 0, 0, 0, 0, 0]\n[0, 5, 1, 5, 0, 0, 0, 0, 0]\n\n\n--Example 1-- \n\n INPUT: \n\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 5, 0, 0, 0, 0, 5, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 5, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 5, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n\n\nOUTPUT: \n\n[0, 5, 1, 5, 0, 0, 5, 1, 5]\n[0, 1, 0, 1, 0, 0, 1, 0, 1]\n[0, 5, 1, 5, 0, 0, 5, 1, 5]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 5, 1, 5, 0, 0, 0, 0, 0]\n[0, 1, 0, 1, 0, 0, 0, 0, 0]\n[0, 5, 1, 5, 0, 5, 1, 5, 0]\n[0, 0, 0, 0, 0, 1, 0, 1, 0]\n[0, 0, 0, 0, 0, 5, 1, 5, 0]\n\n\n\n--End of Training Examples--\n\n--Test Input--\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 5, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 5, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 5, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 5, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0, 0]\n\n--End of Test Input--\n\nYour response:",
        "role": "user"
    },
]

response = generate_text_base(messages=message)

print(response['content'])
