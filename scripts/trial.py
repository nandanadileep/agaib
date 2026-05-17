import json
from main import ScreenOutput

api_key = "AIzaSyCg_fHYi6R-jwUOcp7neXuynuG__LUC9b0"

# Sample LLM output (simulate what LLM would return)
sample_output = {
    "elements": [
        {
            "identity": {"type": "button", "role": "primary", "semantic_type": "submit"},
            "position": {"x": 100, "y": 200, "width": 80, "height": 40},
            "content": {"label": "Submit"},
            "metadata": {"id": "btn-1"}
        },
        {
            "identity": {"type": "input"},
            "position": {"x": 50, "y": 100, "width": 200, "height": 30},
            "content": {"placeholder": "Enter username"},
            "metadata": {"id": "input-1"}
        }
    ]
}

# Validate with Pydantic
result = ScreenOutput.model_validate(sample_output)
print(f"Validated! Found {len(result.elements)} elements")
for el in result.elements:
    print(f"  - {el.metadata.id}: {el.identity.type} at ({el.position.x}, {el.position.y})")

