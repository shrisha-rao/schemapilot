import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_agent():
    model_id = "microsoft/Phi-3-mini-4k-instruct" # Or your base model
    adapter_id = os.getenv("MODEL_PATH", "./model")
    
    print("Loading model on CPU (this may take a minute)...")
    
    # Load base model in FP32 (CPU doesn't support 4-bit/16-bit as well as GPU)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32, 
        device_map="cpu"
    )
    
    # Load your fine-tuned LoRA "Agent" layers
    model = PeftModel.from_pretrained(base_model, adapter_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_agent()
    
    query = "Book the conference room for 3pm."
    prompt = f"Instruction: Output ONLY JSON.\nInput: {query}\nOutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    
    print("\n--- Agent Result ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
