from mcp.server.fastmcp import FastMCP
import json

print("=== Activate Backend Data MCP Server ===")
mcp = FastMCP("Company-Backend-Data-Server")

FAKE_DATABASE = {
    "user_101": {"name": "Alice", "plan": "Enterprise", "monthly_spend": 5000},
    "user_102": {"name": "Bob", "plan": "Pro", "monthly_spend": 200},
}

# Step 1: Define Resource (Read-Only)
@mcp.resource("db://users")
def get_all_user() -> str:
    """
    Fetch all basic info of all users in the company
    """
    return json.dumps(list(FAKE_DATABASE.keys()))

# Step 2: Define Tool (AI's Execution)
@mcp.tool()
def get_user_details(user_id: str) -> str:
    """
    Fetch the detailed plan and spent of the specific user's when user_id is entered
    """
    if user_id in FAKE_DATABASE:
        return json.dumps(FAKE_DATABASE[user_id])
    return json.dumps({"error": "User Not Found"})

def upgrade_user_plan(user_id: str, new_plan: str) -> str:
    """
    [Hazardous] Upgrade user's plan to new_plan (Example: Enterprise)。
    """
    if user_id in FAKE_DATABASE:
        FAKE_DATABASE[user_id]["plan"] = new_plan
        return json.dumps({"status": "success", "new_plan": new_plan})
    return json.dumps({"error": "User not found"})

# Activate with srdio
if __name__ == "__main__":
    print("\nMcp Server Executing..")
    mcp.run(transport="stdio")