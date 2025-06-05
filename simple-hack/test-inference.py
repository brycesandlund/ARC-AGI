from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs, max_length=300)
print(tokenizer.batch_decode(generated_ids)[0])
