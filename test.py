import torch
from diffusers import FluxPipeline
from huggingface_hub import login
import utils


login(utils.get_api_key('huggingface'))
torch.cuda.empty_cache()
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, transformers=None)
pipe = pipe.to("cuda")
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]
image.save("output.png")