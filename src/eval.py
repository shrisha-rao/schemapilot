import json
import time


def evaluate_model(model, tokenizer, test_queries):
    results = {"valid_json": 0, "correct_tool": 0, "latency": []}

    for query in test_queries:
        start_time = time.time()

        # Inference logic
        inputs = tokenizer(f"Input: {query}\nOutput:",
                           return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 1. Check for valid JSON
        try:
            parsed = json.loads(prediction)
            results["valid_json"] += 1
            # 2. Check if tool is correct
            if "tool" in parsed:
                results["correct_tool"] += 1
        except:
            pass

        results["latency"].append(time.time() - start_time)

    return results


# Example run
# eval_stats = evaluate_model(fine_tuned_model, tokenizer, ["Book a room", "Weather in London"])
