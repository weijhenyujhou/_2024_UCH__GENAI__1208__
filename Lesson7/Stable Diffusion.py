from diffusers import StableDiffusionPipeline
import torch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 加載 Stable Diffusion 模型
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#pipe = pipe.to("cuda")  # 如果有 GPU，則移動到 GPU

# 生成圖片
prompt = "A futuristic cityscape at sunset, with flying cars and neon lights"
image = pipe(prompt).images[0]

# 保存或顯示圖片
image.save("output.png")
image.show()
