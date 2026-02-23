import torch
from unsloth import FastLanguageModel


def run_schemapilot(user_query):
    # Load the fine-tuned adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="your-username/SchemaPilot-Phi3-LoRA",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompt = f"Instruction: Output ONLY JSON.\nInput: {user_query}\nOutput:"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64)

    return tokenizer.batch_decode(outputs)


# Test call
# print(run_schemapilot("Reschedule my 3pm to 5pm"))
