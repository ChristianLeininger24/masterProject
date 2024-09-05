import yaml
import torch
import numpy as np
from pathlib import Path
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.data.scene_box import SceneBox

from f3rm.model import FeatureFieldModelConfig, FeatureFieldModel




def rotate_object(x_angle, y_angle):
    """
    Rotate an object in 3D space around the x-axis and y-axis.
    
    Parameters:
    x_angle (float): The angle of rotation around the x-axis (in radians).
    y_angle (float): The angle of rotation around the y-axis (in radians).
    
    Returns:
    np.ndarray: The combined rotation matrix.
    """
    # Rotation matrix around the x-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(x_angle), -np.sin(x_angle)],
                    [0, np.sin(x_angle), np.cos(x_angle)]])
    
    # Rotation matrix around the y-axis
    R_y = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                    [0, 1, 0],
                    [-np.sin(y_angle), 0, np.cos(y_angle)]])
    
    # Combined rotation matrix
    R = np.dot(R_y, R_x)

    return R


def load_state_dict(model, state_dict):
    """   """
    is_ddp_model_state = True
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("_model."):
            # remove the "_model." prefix from key
            model_state[key[len("_model.") :]] = value
            # make sure that the "module." prefix comes from DDP,
            # rather than an attribute of the model named "module"
            if not key.startswith("_model.module."):
                is_ddp_model_state = False
    if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError:
        model.load_state_dict(model_state, strict=False)
    return model


def get_model(model_path, config_path, device="cuda"):
    """ This loads a pre trained model from the path specified in the config file
    
    Args:
        model_path (str): The path to the model
        config_path (str): The path to the config file
        device (str, optional): The device to use. Defaults to "cuda".
    Returns:
        NerfactoModel: The model loaded from the path specified in the config file
    """
    
    #config_path = Path("/home/programmer/master_project_experiments/22_11_2023/2023-11-22_093943/config.yml")
    config_path = Path(config_path)
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    aabb_scale = config.pipeline.datamanager.dataparser.scene_scale
    scene_box = SceneBox(
    aabb=torch.tensor(
        [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
        )
    ) 
    model_config=NerfactoModelConfig()
    print("Loading model")
    model = NerfactoModel(config=model_config, scene_box= scene_box, num_train_data=794).to(device)
    
    loaded_state = torch.load(model_path, map_location="cpu")
    model.update_to_step(loaded_state["step"])
    model = load_state_dict(model, loaded_state["pipeline"])
    return model


def get_f3rm_model(config_path: str, load_path: str, device):
    """ This loads a pre trained model from the path specified in the config file"""
    #config_path = Path("/home/programmer/master_project_experiments/07_12_2023/outputs/nerf_output/f3rm/2023-12-07_072525/config.yml")
    config_path = Path(config_path)
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    model_config=FeatureFieldModelConfig(eval_num_rays_per_chunk=1 << 14)
    aabb_scale = config.pipeline.datamanager.dataparser.scene_scale
    scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
    #TODO num data train needs to be from the file
    model = FeatureFieldModel(model_config, scene_box=scene_box, num_train_data=838, metadata={"feature_dim": 768, "feature_type":"CLIP"}, device=device).to(device)
    # load_path  = "/home/programmer/master_project_experiments/07_12_2023/outputs/nerf_output/f3rm/2023-12-07_072525/nerfstudio_models/step-000029999.ckpt"
    loaded_state = torch.load(load_path, map_location="cpu")
    model.update_to_step(loaded_state["step"])
    model = load_state_dict(model, loaded_state["pipeline"])
    return model