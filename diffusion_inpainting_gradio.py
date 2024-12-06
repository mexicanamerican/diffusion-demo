import gradio as gr
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from PIL import ImageOps, Image

device = "cuda"
model_path = "/root/huggingface/models/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
pipe = AutoPipelineForInpainting.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16").to(device)
print(f"[Diffusion Inpainting] load from {model_path}")

ip_adapter_path = "/root/huggingface/models/h94/IP-Adapter"
pipe.load_ip_adapter(ip_adapter_path, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

print(f"[Diffusion Inpainting] load ip-adapter from {ip_adapter_path}")


def make_mask(mask_image: Image):
    mask = mask_image.convert("RGBA").resize((1024, 1024))
    output_mask = Image.new("RGBA", mask.size, (255, 255, 255, 255))
    output_mask.paste(mask, (0, 0), mask=mask)
    output_mask = output_mask.convert('RGB')
    output_mask = ImageOps.invert(output_mask)
    return output_mask


def predict(crop_data: dict, mask_data: dict, prompt, ip_adapter_scale: float,
     guidance_scale=8.5, strength=1.0, steps=50):

    for k, v in crop_data.items():
        if k == "composite":
            v.save(f"tmp/crop.png")
            ip_image = v

    for k, v in mask_data.items():
        if k == "background":
            v.save(f"tmp/{k}.png")
            image = v
        elif k == "layers":
            v[0].save(f"tmp/layer_0.png")
            mask = v[0]
        elif k == "composite":
            pass

    original_size = image.size
    init_image = image.convert("RGB").resize((1024, 1024))
    mask_image = make_mask(mask)
    mask_image.save(f"tmp/mask.png")

    print(f"[Diffusion Inpainting] ip_adapter_scale: {ip_adapter_scale} guidance_scale: {guidance_scale} strength: {strength} prompt: {prompt}")

    pipe.set_ip_adapter_scale(ip_adapter_scale)
    generator = torch.Generator(device="cpu").manual_seed(4)
    output = pipe(prompt=prompt, image=init_image, mask_image=mask_image, ip_adapter_image=ip_image,
        guidance_scale=guidance_scale, num_inference_steps=int(steps), strength=strength,
        generator=generator,
        )

    output_image = output.images[0]
    # print(f"[Diffusion Inpainting] original_size: {original_size} output_size: {output_image.size}")
    output_image = output_image.resize(original_size)
    return output_image


def init_blocks():
    with gr.Blocks(title="Inpainting") as app:
        gr.Markdown("# Inpainting")
        with gr.Row():
            with gr.Column():
                crop = gr.ImageEditor(sources='upload', elem_id="crop", type="pil", label="Crop", layers=False, height=600,
                    eraser=False,  brush=False)
                prompt = gr.Textbox(label="Prompt", elem_id="prompt")
                # generate = gr.Button(value="Generate")
                ip_adapter_scale = gr.Slider(value=0.6, minimum=0, maximum=1.0, step=0.1, label="IP Adapter Scale")
                guidance_scale = gr.Slider(value=8.5, minimum=0, maximum=20.0, step=0.1, label="Guidance Scale")
                strength = gr.Slider(value=1.0, minimum=0, maximum=1.0, step=0.1, label="Strength")
            with gr.Column():
                mask = gr.ImageMask(sources='upload', elem_id="mask", type="pil", label="Mask", layers=False, height=600,
                    transforms=())
            with gr.Column():
                image_out = gr.Image(label="Output", elem_id="output-img", height=600, format="png")

        prompt.submit(fn=predict, inputs=[crop, mask, prompt, ip_adapter_scale, guidance_scale, strength], outputs=[image_out])
        # generate.click(fn=predict, inputs=[crop, mask, prompt, ip_adapter_scale, guidance_scale, strength], outputs=[image_out])

    return app


if __name__ == "__main__":
    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0',
        server_port=8060, show_api=False, share=False)
