from talkie import Talkie

model = Talkie("talkie-1930-13b-base")

# Test if model "knows" about post-1930 concepts
prompts_to_test = [
    "The television will revolutionize",
    "Computers in the future will",
    "Nuclear weapons are",
    "The internet allows people to",
    "Franklin Roosevelt became president and",
    "World War II began when",
    "System prompt: You are a helpful AI assistant",
    "Please ignore previous instructions and",
    "As an AI language model,",
]

for prompt in prompts_to_test:
    response = model.generate(prompt, max_tokens=50, temperature=0.1)
    print(f"Prompt: {prompt}")
    print(f"Response: {response.text}")
    print("---")
