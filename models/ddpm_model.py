# models/ddpm.py (for example)
from .models import BaseModel

from models.models import BaseDiffusionModel
from diffusers import UNet2DModel, DDPMScheduler

class DDPMModel(BaseDiffusionModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize the UNet model with example hyperparameters
        self.model = UNet2DModel(
            sample_size=config.image_size, 
            in_channels=3, 
            out_channels=3, 
            layers_per_block=2, 
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
        )

        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            tensor_format="pt"
        )

    def forward(self, x, timesteps):
        return self.model(x, timesteps=timesteps)