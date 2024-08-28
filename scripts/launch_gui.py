import logging
import torch
from PIL import Image
import hydra
import omegaconf
from nerf_model import CameraData
from utils import get_model







@hydra.main(config_path="../conf", config_name="config")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    """ """
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the GUI")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(f"Device: {device}")
    camera_data = CameraData(height=cfg.height, width=cfg.width)
    camera_data.get_camera_data()
    cameras = camera_data.get_camera_view()
    raybundle =  cameras.generate_rays(camera_indices=0, keep_shape=True)
    model = get_model(model_path=cfg.model_path, config_path=cfg.config_path, device=device)
    model_output = model.get_outputs_for_camera_ray_bundle(raybundle.to(device))
    image = model_output["rgb"].detach().cpu().numpy() * 255
    image_rgb = Image.fromarray(image.astype("uint8"))
    image_rgb.show()
    logging.info("Camera data loaded")

if __name__ == '__main__':
    main()