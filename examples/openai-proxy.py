from openai import OpenAI

client = OpenAI(api_key="<EXISTING CAPTAIN CLAW SESSION ID>", base_url="http://127.0.0.1:23080/v1")
resp = client.chat.completions.create(
    model="captain-claw",
    messages=[{"role": "user", "content": "Hello! What can you do?"}],
)
print(resp.choices[0].message.content)
