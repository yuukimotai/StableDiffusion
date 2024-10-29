from diffusers import DiffusionPipeline
import datetime

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

pipe.enable_attention_slicing()

prompt = "スプラトゥーンのイカちゃんがスプラシューターでバルーンを撃ってる写真"

image = pipe(prompt).images[0]
today = datetime.datetime.now()

image.save(f'{today}_output.png')