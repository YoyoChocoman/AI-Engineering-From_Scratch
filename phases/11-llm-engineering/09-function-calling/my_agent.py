import json

print("=== AI Agent Core: Function Calling ===")

# Step 1: A JSON Schema for AI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch the specific city's weather and temperature.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City's name"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send Email to the specific contacter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient's Email address"},
                    "subject": {"type": "string", "description": "Subject of the letter"},
                    "body": {"type": "string", "description": "Body of the letter"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

# Step 2: Python
def execute_get_weather(city):
    print(f"\n[Executing...] Connecting to weather forecast querying {city}'s weather...")
    return {"temp_c": 25, "condition": "Sunny"}

def execute_send_email(to, subject, body):
    print(f"\n[Executing...] Sending Email to {to}...")
    print(f"Subject: {subject}\nBody: {body}")
    return {"status": "success", "message": "Email sent"}

# Simulate AI Execution and Reply
print("\nUser query: How the weather today? Send an Email to boss@company.com say that I would like to have a day leave.")

mock_ai_command = [
    {"name": "get_weather", "arguments": '{"city": "Taipei"}'},
    {"name": "send_email", "arguments": '{"to": "boss@company.com", "subject": "Leave Notice", "body": "I would like to havea day leave to relax."}'}
]

print("\nTool Calls:")
print(json.dumps(mock_ai_command, indent=2, ensure_ascii=False))

for call in mock_ai_command:
    func_name = call["name"]
    args = json.loads(call["arguments"])

    if func_name == "get_weather":
        result = execute_get_weather(args["city"])
        print(f"-> Result: {result}")

    elif func_name == "send_email":
        result = execute_send_email(args["to"], args["subject"], args["body"])
        print(f"-> Result: {result}")