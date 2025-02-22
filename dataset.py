import torch
from diffusers import FluxPipeline
from openai import OpenAI

client = OpenAI(
    api_key="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"
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user", "content": "write a haiku about ai"}
    ]
)

print(completion.choices[0].message)



pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)