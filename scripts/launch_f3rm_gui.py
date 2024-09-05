import logging
import torch
from PIL import Image
import hydra
import omegaconf
from nerf_model import CameraData
from utils import get_model, get_f3rm_model, rotate_object

from segment_anything import sam_model_registry, SamPredictor
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import get_normalized_directions
from f3rm.feature_field import FeatureFieldHeadNames

from f3rm.model import ViewerUtils
import time
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm

def sam_model(device):
    """   """
    sam_checkpoint = "/home/programmer/master_project_experiments/sam_model_weights/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return  SamPredictor(sam)


def get_mask(mask, random_color=False):
    """  """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    return mask.reshape(h, w, 1) * color.reshape(1, 1, -1)



def create_mask(sim_array, rgb_image, device):
    """  """
    row, column = np.unravel_index(np.argmax(sim_array), sim_array.shape)
    predictor = sam_model(device)   # some how needs to be cuda
    predictor.set_image(rgb_image)
    input_point = np.array([[column, row]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
        )
    mask =  get_mask(masks[scores.argmax()])
    summed_mask = np.sum(mask, axis=2)
    return summed_mask != 0




def create_ray_bundle(image_width=320, image_height=240, x=0.2, y=0, z=-0.2, x_angle=0, y_angle=1, device="cpu", mask=None):
    """  """
    # ensure mask and image have the same shape
    if mask is not None:
        image_height, image_width = mask.shape[:2]
    camera_data = CameraData(width=image_width, height=image_height)
    camera_data.get_camera_data()
    rotate_matrix = rotate_object(x_angle, y_angle)
    camera = camera_data.get_camera_view(x=x, y=y, z=z, rotation_matrix=rotate_matrix)
    if mask is not None:
        camera.clip_density = torch.from_numpy(np.expand_dims(mask, axis=-1)).to(device)
    return camera.generate_rays(camera_indices=0, keep_shape=True).to(device)




def get_data(model, ray_samples, outputs_lists, weights_list):
    """ """
   
    density, density_embedding = model.field.get_density(ray_samples)
        
    if ray_samples.camera_indices is None:
        raise AttributeError("Camera indices are not provided.")
    camera_indices = ray_samples.camera_indices.squeeze()
    directions = get_normalized_directions(ray_samples.frustums.directions)
    directions_flat = directions.view(-1, 3)
    d = model.field.direction_encoding(directions_flat)
    outputs_shape = ray_samples.frustums.directions.shape[:-1]
    if model.field.use_average_appearance_embedding:
            embedded_appearance = torch.ones(
                (*directions.shape[:-1], model.field.appearance_embedding_dim), device=directions.device
            ) * model.field.embedding_appearance.mean(dim=0)
    else:
            embedded_appearance = torch.zeros(
                (*directions.shape[:-1], model.field.appearance_embedding_dim), device=directions.device
            )
    h = torch.cat(
        [
            d,
            density_embedding.view(-1, model.field.geo_feat_dim),
            embedded_appearance.view(-1, model.field.appearance_embedding_dim),
        ],
        dim=-1,
    )
    rgb = model.field.mlp_head(h).view(*outputs_shape, -1)
    outputs_lists[FieldHeadNames.DENSITY].append(density.cpu())
    weights = ray_samples.get_weights(density)
    weights_list.append(weights)
    rgb = model.renderer_rgb(rgb=rgb, weights=weights)
    outputs_lists[FieldHeadNames.RGB].append(rgb.cpu())
    ff_outputs = model.feature_field(ray_samples) # 16384, 48, 768
    feature =  model.renderer_feature(ff_outputs["feature"].cpu(), weights=weights.cpu())
    outputs_lists[FeatureFieldHeadNames.FEATURE].append(feature.cpu())
    # features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)

    return outputs_lists, weights_list

def get_output(raybundle, model, text_positiv, device):
    """  """
    ray_bundle =  model.collider(raybundle)
    # density needs ray_samples first 
    num_rays_per_chunk = model.config.eval_num_rays_per_chunk
    image_height, image_width =ray_bundle.origins.shape[:2]
    num_rays = len(ray_bundle)
    # outputs_lists = defaultdict(list)
    outputs_lists = defaultdict(list)
    weights_list = []
    for i in range(0, num_rays, num_rays_per_chunk):
        start_idx = i
        end_idx = i + num_rays_per_chunk
        ray_bundle_slice = ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
        ray_samples, weights_list, ray_samples_list = model.proposal_sampler(ray_bundle_slice.to(device), density_fns=model.density_fns)

        if ray_samples.shape[0] == 0:
             continue
        
        outputs_lists, weights_list = get_data(model, ray_samples, outputs_lists, weights_list)
    
    rgb_data =  torch.cat(outputs_lists[FieldHeadNames.RGB]).cpu().view(image_height, image_width, -1).detach().numpy()
    
    rgb_image = (rgb_data * 255).astype(np.uint8)
    clip_features = outputs_lists[FeatureFieldHeadNames.FEATURE]
    clip_features = torch.cat(clip_features).cpu().view(image_height, image_width, -1).detach()
    clip_features /= clip_features.norm(dim=-1, keepdim=True)
    
    viewer_utils = ViewerUtils()
    viewer_utils.handle_language_queries(text_positiv, is_positive=True)
    if not viewer_utils.has_negatives:
        sims = clip_features @ viewer_utils.pos_embed.T.cpu()
    return rgb_image, sims.squeeze(2).numpy(), clip_features



def create_sim_image(sim_data):
    """ """
    colormap = cm.get_cmap('hot')  # You can change 'hot' to any colormap you like
    normed_data = (sim_data - np.min(sim_data)) / (np.max(sim_data) - np.min(sim_data))
    heatmap_rgba = colormap(normed_data)
    heatmap = (heatmap_rgba * 255).astype(np.uint8)
    return Image.fromarray(heatmap)





@hydra.main(config_path="../conf", config_name="config_f3rm")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    """ """
    debug_info = False
    start_time = time.time()  # Capture the start time
    colormap = cm.get_cmap('hot')  # You can change 'hot' to any colormap you li
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the GUI")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    logging.debug(f"Device: {device}")
    camera_data = CameraData(height=cfg.height, width=cfg.width)
    camera_data.get_camera_data()
    rotate_matrix = rotate_object(x_angle=cfg.x_angle, y_angle=cfg.y_angle)
    cameras = camera_data.get_camera_view(x=cfg.x, y=cfg.y, z=cfg.z, rotation_matrix=rotate_matrix)
    raybundle =  cameras.generate_rays(camera_indices=0, keep_shape=True)
    model = get_f3rm_model(config_path=cfg.config_path, load_path=cfg.model_path, device=device)
    #model_output = model.get_outputs_for_camera_ray_bundle(raybundle.to(device), text_positives=input_string)
    #sim_data = model_output["similarity"].squeeze(2).cpu().numpy()
    
    rgb_image, sim_data, clip_feature = get_output(raybundle, model, text_positiv="Laptop", device=device)    
    
    if debug_info:
        sim_image = create_sim_image(sim_data)
        sim_image.show()
        image_rgb = Image.fromarray(rgb_image)
        image_rgb.show()
    boolean_mask = create_mask(sim_data, rgb_image.astype("uint8"), device)
    raybundle = create_ray_bundle(device=device, mask=boolean_mask)
    rgb_image, sim_data, clip_feature = get_output(raybundle, model, text_positiv="Can", device=device)  
    # create images  
    if debug_info:
        sim_image = create_sim_image(sim_data)
        sim_image.show()
        image_rgb = Image.fromarray(rgb_image)
        image_rgb.show()
        mask_image = Image.fromarray((boolean_mask * 255).astype("uint8"))
        mask_image.show()
    logging.info(f"Time taken: {time.time() - start_time}")
    return
    """ 
    # ray_samples, weights_list, ray_samples_list = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
    model_output = model.get_outputs_for_camera_ray_bundle(raybundle.to(device), text_positives=input_string)
    # keys of the model_output
    # dict_keys(['rgb', 'accumulation', 'depth', 'expected_depth', 'feature', 'density', 'prop_depth_0', 'prop_depth_1', 'similarity'])    
    import matplotlib.cm as cm
    import numpy as np

    sim_data = model_output["similarity"].squeeze(2).cpu().numpy()
    colormap = cm.get_cmap('hot')  # You can change 'hot' to any colormap you like
    normed_data = (sim_data - np.min(sim_data)) / (np.max(sim_data) - np.min(sim_data))
    heatmap_rgba = colormap(normed_data)
    heatmap = (heatmap_rgba * 255).astype(np.uint8)
    # show the image
    rgb_image = model_output["rgb"].detach().cpu().numpy() * 255
    
    mask = create_mask(sim_data, rgb_image.astype("uint8"), device)
    # use the ray 
    mask_image = Image.fromarray((mask * 255).astype("uint8"))
    mask_image.show()
    image_sim = Image.fromarray(heatmap)
    image_sim.show()
    
    image_rgb = Image.fromarray(rgb_image.astype("uint8"))
    image_rgb.show()
    model.feature_field.get_density(raybundle)
    return
    """ 
    
if __name__ == '__main__':
    main()