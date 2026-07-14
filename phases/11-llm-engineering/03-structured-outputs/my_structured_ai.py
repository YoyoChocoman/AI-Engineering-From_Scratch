from pydantic import BaseModel, Field

# Schema
class InvoiceData(BaseModel):
    store_name: str = Field(..., description="The name of the shop")
    total_amount: float = Field(..., description="The total invoice amount, which must be a float point")
    items_bought: list[str] = Field(..., description="The list of the things bought")
    is_tax_included: bool = Field(..., description="Is the reciept taxed? Fill it with 'True' or 'False'")

print("=== Pydantic Model ===")
print(InvoiceData.model_json_schema())

print("\n=== Simulate AI Response ===")
mock_ai_response = """
{
  "store_name": "全家便利商店",
  "total_amount": 155.0,
  "items_bought": ["茶葉蛋", "冰美式", "御飯糰"],
  "is_tax_included": true
}
"""
print(f"AI's Output:\n{mock_ai_response}")

try:
    final_data = InvoiceData.model_validate_json(mock_ai_response)
    print("\n=== Transformation Successful ===")
    print(f"Store Name: {final_data.store_name} (Type: {type(final_data.store_name)})")
    print(f"Invoice Amount: {final_data.total_amount} (Type: {type(final_data.total_amount)})")
    print(f"List Length: {len(final_data.items_bought)}")

except Exception as e:
    print(f"Transformation Failed: {e}")