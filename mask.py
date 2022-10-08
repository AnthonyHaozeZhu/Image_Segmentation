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
        # TODO：这一部分参数还需要进行细致研究一下
        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                # 用来跳过一些步骤的，我目前把这一部分删掉看看是什么样的结果，删除之后我们的模型需要迭代1000次才可以获得一个结果，和我们预料的一样，这样的操作有点慢，后面再仔细阅读论文看看怎么办
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
        # TODO：现在这里text_embedding不能处理batch的东西，这里还需要进行一定的处理
        clip_loss = torch.tensor(0)
        augmented_input = x_in.add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        # 使用的CLIP模型在这里只能接受224*224的输入的，所以在这里必须要对其大小进行转换
        resize = transforms.Resize([self.clip_size, self.clip_size])
        clip_in = resize(clip_in)
        image_embeds = self.clip.encode_image(clip_in).float()
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
                # 如果生成的东西是三通道的，那我们将所有通道中的值加权平均后可以得到一个一通道的
                mask = torch.mean(input=x_in, dim=1).unsqueeze(1)
                mask = mask.repeat(1, 3, 1, 1) > 0.5
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

        # 先不进行多次迭代，正常操作
        # for iteration_number in range(self.args.iterations_num):
        # print("hahah:", iteration_number)
        sample_func = (
            self.diffusion.ddim_sample_loop_progressive
            if self.args.ddim
            else self.diffusion.p_sample_loop_progressive
        )
        samples = sample_func(
            model=self.model,
            shape=(
                self.args.batch_size,
                # TODO：这一部分写的是3通道的，如果改成1通道将会报错，需要进行修改
                3,  # 生成的mask只需要一个通道即可
                self.model_config["image_size"],
                self.model_config["image_size"]
            ),  # 输入内容的形状
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
            # TODO：这些参数还没有弄明白具体意思，还需要继续代码阅读进行更改
            # skip_timesteps=self.args.skip_timesteps,
            # init_image=self.init_image,  # 需要进行操作的图片
            # postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
            # randomize_class=True,
        )

        # TODO: 看看生成的是个什么样子的东西，暂时还有一定的报错，这一部分还没有使用，但是上面已经进行了控制之类的操作，还需要具体研究一下代码的流程步骤
        total_mask = []
        for index in range(self.args.batch_size):
            # TODO：generator类型元素，不能直接读取最后一个，必须进行迭代
            for j, sample in enumerate(samples):
                if j == 99:
                    print(self.diffusion.num_timesteps)
                    print(self.args.skip_timesteps)
                    print(j)
                    soft_mask = sample["pred_xstart"][index].add(1).div(2).clamp(0, 1)  # 这些操作是为了解决生成的图像中有数值小于0的问题
                    soft_mask = np.array(soft_mask.to("cpu")).transpose((1, 2, 0)) * 255
                    # print(soft_mask))
                    mask_pil = Image.fromarray(np.uint8(soft_mask))
                    mask_pil.save("./test_mask.png")
                    total_mask.append(sample["pred_xstart"][index])

        return total_mask










