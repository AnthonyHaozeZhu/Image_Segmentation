# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：mask.py
@Author ：AnthonyZ
@Date ：2022/10/4 13:41
"""

import clip
import torch
from torchvision import transforms
from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from utils import *


class MaskMaker:
    def __init__(self, args):
        self.args = args
        self.metrics_accumulator = MetricsAccumulator()
        self.clip = clip.load("ViT-B/16", device=self.args.device, jit=False)[0].eval().requires_grad_(False)
        self.clip_size = self.clip.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )
        self.model, self.diffusion = create_model_and_diffusion(
            **self.model_config
        )
        self.model.load_state_dict(
            torch.load(
                "checkpoints/256x256_diffusion_uncond.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu"
            )
        )
        self.model.requires_grad_(False).eval().to(self.args.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def clip_loss(self, x_in, text_embed):
        clip_loss = torch.tensor(0)
        augmented_input = x_in.add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        # 使用的CLIP模型在这里只能接受224*224的输入的，所以在这里必须要对其大小进行转换
        print(clip_in.shape)
        image_embeds = self.clip.encode_image(clip_in).float()  # 这一行有问题呗就是
        dists = d_clip_loss(image_embeds, text_embed)

        # We want to sum over the averages
        for i in range(self.args.batch_size):
            # We want to average at the "augmentations level"
            clip_loss = clip_loss + dists[i:: self.args.batch_size].mean()

        return clip_loss

    def make_mask_by_language(self, image, text):
        text_embed = self.clip.encode_text(
            clip.tokenize(text).to(self.args.device)
        ).float()

        def grad_controller(x, t, y=None):
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)
                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False,
                    model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)  # 获得的生成的中间图像，可以理解为我们需要的mask
                mask = x_in  # .repeat(1, 3, 1, 1) > 0.5
                item = mask * image

                loss = torch.tensor(0)
                if self.args.clip_guidance_lambda != 0:
                    clip_loss = self.clip_loss(item, text_embed) * self.args.clip_guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                return -torch.autograd.grad(loss, x)[0]

        for iteration_number in range(self.args.iterations_num):
            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                model=self.model,
                shape=(
                    self.args.batch_size,
                    3,  # 生成的mask只需要一个通道即可
                    self.model_config["image_size"],
                    self.model_config["image_size"]
                ),
                clip_denoised=False,
                model_kwargs={}  # 传给模型额外的关键字儿
                if self.args.model_output_size == 256
                else {
                    "y": torch.zeros([self.args.batch_size],
                                     device=self.args.device,
                                     dtype=torch.long)
                },
                cond_fn=grad_controller,  # 控制生成条件的方法
                progress=True,  # 显示一个tqdm的进度条随时查看进度
                # skip_timesteps=self.args.skip_timesteps,
                # init_image=self.init_image,  # 需要进行操作的图片
                # postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
                # randomize_class=True,
            )
            intermediate_samples = [[] for _ in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

            # 以下直接复制尝试以下这个东西能不能跑

            # TODO: 看看生成的是个什么样子的东西
            for j, sample in enumerate(samples):
                for index in range(self.args.batch_size):
                    # pred_mask = sample["pred_xstart"][index]
                    print("hah")
            return # pred_mask










