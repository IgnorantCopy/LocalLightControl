import torch
from diffusers import FluxPipeline
from openai import OpenAI
import base64


def encode_image(image_path: str):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_objects(image_path: str) -> str:
    api_key="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"
    prompt = '''
    List all the objects in this image, including the background. I have these requirements as follows:
    1. Do not describe the objects in detail, just list them.
    2. Use '/' to separate different names of the same object if exists.
    '''
    image_url = {
        "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
    }
    client = OpenAI(
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ]}
        ],
        max_tokens=300,
    )
    return completion.choices[0].message.content


def get_instructions(image_path: str) -> list:
    objects = get_image_objects(image_path)
    if not objects:
        return None
    objects = objects.split(" / ")
    instructions = []
    for obj in objects:
        instructions.append(f"make the {obj} deep dark")
    return instructions


def get_paired_images(image_path: str):
    instructions = get_instructions(image_path)
    if not instructions:
        return None
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)


if __name__ == "__main__":
    for i in get_instructions("./data/101839123902129423.jpg"):
        print(i)