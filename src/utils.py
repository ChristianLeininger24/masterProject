import yaml
import torch
from pathlib import Path
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.data.scene_box import SceneBox


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