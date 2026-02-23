import json


def validate_and_route(model_output):
    """
    Ensures the model didn't hallucinate extra text and 
    that the JSON is actually parsable.
    """
    try:
        data = json.loads(model_output)
        # In a real app, you'd trigger the tool here
        print(f"✅ Success: Routing to {data['tool']}")
        return data
    except json.JSONDecodeError:
        print("❌ Error: Model produced invalid JSON.")
        return None
