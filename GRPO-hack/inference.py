from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
# BASE_MODEL_NAME = "/root/arc/ARC-AGI/checkpoint-2800"

# load the tokenizer and the model

# load the tokenizer and the model
base_model_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
base_model_tokenizer.pad_token = '<|endoftext|>'
base_model_tokenizer.eos_token_id = 151643

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

def generate_text_base(messages, max_new_tokens=500):
    # text = base_model_tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    # )
    text = messages

    model_inputs = base_model_tokenizer([text], return_tensors="pt").to(base_model.device)

    # conduct text completion
    generated_ids = base_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
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
# print(message[0]['content'])


challenge =  {
    "train": [
        {
            "input": [
                [
                    6,
                    6,
                    0
                ],
                [
                    6,
                    0,
                    0
                ],
                [
                    0,
                    6,
                    6
                ]
            ],
            "output": [
                [
                    6,
                    6,
                    0,
                    6,
                    6,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    6,
                    0,
                    0,
                    6,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    6,
                    6,
                    0,
                    6,
                    6,
                    0,
                    0,
                    0
                ],
                [
                    6,
                    6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    6,
                    6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    6,
                    6,
                    0,
                    6,
                    6,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    6,
                    0,
                    0,
                    6,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    6,
                    6,
                    0,
                    6,
                    6
                ]
            ]
        },
        {
            "input": [
                [
                    4,
                    0,
                    4
                ],
                [
                    0,
                    0,
                    0
                ],
                [
                    0,
                    4,
                    0
                ]
            ],
            "output": [
                [
                    4,
                    0,
                    4,
                    0,
                    0,
                    0,
                    4,
                    0,
                    4
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    4,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    4,
                    0,
                    4,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                    0
                ]
            ]
        },
        {
            "input": [
                [
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    2
                ],
                [
                    2,
                    0,
                    2
                ]
            ],
            "output": [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    0,
                    2
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2
                ],
                [
                    2,
                    0,
                    2,
                    0,
                    0,
                    0,
                    2,
                    0,
                    2
                ]
            ]
        },
        {
            "input": [
                [
                    2,
                    2,
                    2
                ],
                [
                    0,
                    0,
                    0
                ],
                [
                    0,
                    2,
                    2
                ]
            ],
            "output": [
                [
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    2,
                    2,
                    0,
                    2,
                    2,
                    0,
                    2,
                    2
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    2,
                    2,
                    0,
                    2,
                    2
                ]
            ]
        },
        {
            "input": [
                [
                    0,
                    7,
                    7
                ],
                [
                    7,
                    7,
                    7
                ],
                [
                    0,
                    7,
                    7
                ]
            ],
            "output": [
                [
                    0,
                    0,
                    0,
                    0,
                    7,
                    7,
                    0,
                    7,
                    7
                ],
                [
                    0,
                    0,
                    0,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    7,
                    7,
                    0,
                    7,
                    7
                ],
                [
                    0,
                    7,
                    7,
                    0,
                    7,
                    7,
                    0,
                    7,
                    7
                ],
                [
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7
                ],
                [
                    0,
                    7,
                    7,
                    0,
                    7,
                    7,
                    0,
                    7,
                    7
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    7,
                    7,
                    0,
                    7,
                    7
                ],
                [
                    0,
                    0,
                    0,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    7,
                    7,
                    0,
                    7,
                    7
                ]
            ]
        }
    ],
    "test": [
        {
            "input": [
                [
                    7,
                    0,
                    7
                ],
                [
                    7,
                    0,
                    7
                ],
                [
                    7,
                    7,
                    0
                ]
            ]
        }
    ]
}

challenge2 = {
        "train": [
            {
                "input": [
                    [
                        7,
                        9
                    ],
                    [
                        4,
                        3
                    ]
                ],
                "output": [
                    [
                        7,
                        9,
                        7,
                        9,
                        7,
                        9
                    ],
                    [
                        4,
                        3,
                        4,
                        3,
                        4,
                        3
                    ],
                    [
                        9,
                        7,
                        9,
                        7,
                        9,
                        7
                    ],
                    [
                        3,
                        4,
                        3,
                        4,
                        3,
                        4
                    ],
                    [
                        7,
                        9,
                        7,
                        9,
                        7,
                        9
                    ],
                    [
                        4,
                        3,
                        4,
                        3,
                        4,
                        3
                    ]
                ]
            },
            {
                "input": [
                    [
                        8,
                        6
                    ],
                    [
                        6,
                        4
                    ]
                ],
                "output": [
                    [
                        8,
                        6,
                        8,
                        6,
                        8,
                        6
                    ],
                    [
                        6,
                        4,
                        6,
                        4,
                        6,
                        4
                    ],
                    [
                        6,
                        8,
                        6,
                        8,
                        6,
                        8
                    ],
                    [
                        4,
                        6,
                        4,
                        6,
                        4,
                        6
                    ],
                    [
                        8,
                        6,
                        8,
                        6,
                        8,
                        6
                    ],
                    [
                        6,
                        4,
                        6,
                        4,
                        6,
                        4
                    ]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [
                        3,
                        2
                    ],
                    [
                        7,
                        8
                    ]
                ]
            }
        ]
    }
def grid_to_string(grid):
    # Convert each row to a string representation with brackets
    rows_str = [str(row) for row in grid]
    # Join rows with newline characters
    grid_str = "\n".join(rows_str)
    return grid_str

prompt = "<|im_start|>user\nYou are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.\n\nRespond in the format of the training output examples\n\n--Training Examples--"

for i, input_example in enumerate(challenge["train"]):
    prompt += f"\n--Example {i}-- \n\n INPUT: \n\n"
    prompt += grid_to_string(input_example['input'])
    prompt += "\n\n\nOUTPUT: \n\n"
    prompt += grid_to_string(input_example['output'])
    prompt += "\n\n"

prompt += "\n\n--End of Training Examples--\n\n--Test Input--\n"
prompt += grid_to_string(challenge["test"][0]['input'])
prompt += "\n\n--End of Test Input--\n\nProvide answer in <answer> tag precisely so that I can parse and validate.<|im_end|>\n<|im_start|>assistant"

# response = generate_text_base(messages=[{ "content": prompt, "role": "user" }], max_new_tokens=4000)
response = generate_text_base(messages=prompt, max_new_tokens=4000)
print(response['content'])
# print(prompt)
import pdb
pdb.set_trace()
