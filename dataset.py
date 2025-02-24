import torch
from diffusers import FluxPipeline
from openai import OpenAI
import base64
import utils


def encode_image(image_path: str):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_objects(image_path: str) -> str:
    api_key = utils.get_api_key('openai')
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
        return []
    objects = objects.split(" / ")
    instructions = []
    for obj in objects:
        instructions.append(f"make the {obj.lower()} deep dark")
    return instructions


def get_paired_images(image_path: str):
    instructions = get_instructions(image_path)
    if not instructions:
        return None
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)


if __name__ == "__main__":
    for i in get_instructions("./data/stairs.jpg"):
        print(i)