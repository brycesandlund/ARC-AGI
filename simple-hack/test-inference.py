from transformers import AutoModelForCausalLM, AutoTokenizer

def is_correct(content):
    """
    Evaluates the mathematical expression in content and returns True if it equals 16.
    
    Args:
        content (str): The mathematical expression to evaluate
        
    Returns:
        bool: True if the expression evaluates to 16, False otherwise
    """
    try:
        # Clean the content by stripping whitespace and newlines
        expression = content.strip()
        
        # Evaluate the mathematical expression
        result = eval(expression)
        
        # Check if the result equals 16 (with some floating point tolerance)
        return abs(result - 16) < 1e-10
        
    except (SyntaxError, NameError, ZeroDivisionError, TypeError, ValueError) as e:
        # Return False if the expression is invalid or causes an error
        print(f"Error evaluating expression '{content}': {e}")
        return False

model_name = "Qwen/Qwen3-1.7B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Using the numbers 8, 4, 2 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 16. Answer exactly in plain mathematical notation (no LaTeX), with no additional text. For example, if the provided numbers are 2, 7, and 2, a valid answer would be: 2 * 7 + 2."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.1
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

raw = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip("\n")
# print("raw:", raw)

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

# Check if the generated expression is correct
result_is_correct = is_correct(content)
print(f"Is the answer correct? {result_is_correct}")
