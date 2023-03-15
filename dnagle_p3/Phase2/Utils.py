
import torch
import cv2


def get_minibatches(inputs, chunksize=1024 * 8):
  """
  Function to split a tensor into chunks of a given size.
  :param inputs: Tensor to be split.
  :param chunksize: Size of the chunks.
  :return: List of chunks.
  """
  return torch.split(inputs, chunksize) # Split the tensor into chunks of size chunksize

def cumulative_product(tensor):
    """
    Function to compute the cumulative product of a tensor.
    :param tensor: Tensor to compute the cumulative product of.
    :return: Exclusive cumulative product of the tensor.
    """
    # Compute the cumulative product of the tensor along the last dimension
    cumprod = torch.cumprod(tensor, dim=-1)
    
    # Shift the elements of the cumulative product tensor to the right
    shifted_cumprod = torch.roll(cumprod, 1, dims=-1)
    
    # Replace the first element with 1
    shifted_cumprod[..., 0] = 1
    
    # Compute the exclusive cumulative product by dividing the cumulative product
    # by the shifted cumulative product
    exclusive_cumprod = cumprod / shifted_cumprod
    
    return exclusive_cumprod

def normalize(image_mat):
    return cv2.normalize(image_mat,dst=None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

def accumulatingTransmittance(alphavals):
        accTrans = torch.cumprod(alphavals, 1)
        return torch.cat((torch.ones((accTrans.shape[0], 1)), accTrans[:,:-1]), dim=-1) #device=alphavals.device

def get_ray_bundle(height, width, focal_length, camera_to_world):
    """
    Compute the bundle of rays through all image pixels.
    :param height: Height of the image.
    :param width: Width of the image.
    :param focal_length: Focal length of the camera.
    :param camera_to_world: Camera-to-world transformation matrix.
    :return: Tuple containing the ray origins and directions.
    """
    # Compute a meshgrid of pixel coordinates
    x = torch.arange(width, dtype=torch.float32, device=camera_to_world.device)
    y = torch.arange(height, dtype=torch.float32, device=camera_to_world.device)
    x, y = torch.meshgrid(x, y)
    xx, yy = x.transpose(-1, -2), y.transpose(-1, -2)

    # Compute pixel directions
    pixel_directions = torch.stack(
        [
            (xx - width * .5) / focal_length,
            -(yy - height * .5) / focal_length,
            -torch.ones_like(xx),
        ],
        dim=-1,
    )

    # Transform pixel directions to world directions
    world_directions = torch.sum(
        pixel_directions[..., None, :] * camera_to_world[:3, :3], dim=-1
    )

    # Compute ray origins as camera centers
    ray_origins = camera_to_world[:3, -1].expand(world_directions.shape)

    return ray_origins, world_directions



def compute_query_points_from_rays(ray_origins, ray_directions, near_plane, far_plane, num_samples, randomize=True):
    """
    Function to compute query points along rays.
    :param ray_origins: Origin of each ray in the "bundle" as returned by the `get_ray_bundle()` method.
    :param ray_directions: Direction of each ray in the "bundle" as returned by the `get_ray_bundle()` method.
    :param near_plane: Near plane for the depth values.
    :param far_plane: Far plane for the depth values.
    :param num_samples: Number of depth samples along each ray.
    :param randomize: Whether to randomize the depth samples.
    :return: Tuple containing the query points and depth values.
    """
    device = ray_origins.device # Get the device of the ray origins
    batch_size = ray_origins.shape[0] # Get the batch size
    num_rays = ray_origins.shape[1] # Get the number of rays

    # Generate a list of depth values for each ray
    depth_values_list = []
    for _ in range(batch_size):
        for _ in range(num_rays):
            depth_values = torch.linspace(near_plane, far_plane, num_samples, device=device) # Generate a list of depth values
            if randomize: 
                noise = torch.rand((num_samples,), device=device) # Generate a list of random numbers
                depth_values += noise * (far_plane - near_plane) / num_samples # Add the random numbers to the depth values
            depth_values_list.append(depth_values) # Append the depth values to the list

    # Stack the list of depth values into a tensor
    depth_values = torch.stack(depth_values_list, dim=0).reshape(batch_size, num_rays, num_samples)
    
    # Compute query points from ray origins, ray directions, and depth values
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]

    return query_points, depth_values


def render_volume_density(radiance_field, ray_origins, depth_values):
    """
    Function to render the density of a volume.
    :param radiance_field: Radiance field of the volume.
    :param ray_origins: Origin of each ray in the "bundle" as returned by the `get_ray_bundle()` method.
    :param depth_values: Depth values along each ray as returned by the `compute_query_points_from_rays()` method.
    :return: Tuple containing the density map, depth map, and accumulated density map.
    """
    attenuation = torch.nn.functional.relu(radiance_field[..., 3]) # Get the attenuation values
    color = torch.sigmoid(radiance_field[..., :3]) # Get the color values
    max_depth = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device) # Get the maximum depth value
    ray_lengths = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], max_depth.expand(depth_values[..., :1].shape)), dim=-1) # Compute the length of each ray segment
    ray_alphas = 1. - torch.exp(-attenuation * ray_lengths) # Compute the alpha values for each ray segment 
    ray_weights = ray_alphas * cumulative_product(1. - ray_alphas + 1e-10) # Compute the weights for each ray segment
    color_map = (ray_weights[..., None] * color).sum(dim=-2) # Compute the color map
    depth_map = (ray_weights * depth_values).sum(dim=-1) # Compute the depth map
    weight_sum = ray_weights.sum(-1) # Compute the accumulated density map
    return color_map, depth_map, weight_sum # Return the color map, depth map, and accumulated density map


def compute_positional_encoding(input_tensor, num_encoding_functions=6, include_input=True, use_log_sampling=True):
    """
    Computes the positional encoding of an input tensor.
    :param input_tensor: Input tensor.
    :param num_encoding_functions: Number of encoding functions to use.
    :param include_input: Whether to include the input tensor in the encoding.
    :param use_log_sampling: Whether to use logarithmic sampling of the encoding functions.
    :return: Positional encoding tensor.
    """

    encoding = [input_tensor] if include_input else [] # Initialize the encoding list

    device = input_tensor.device
    
    # Compute the frequency bands
    if use_log_sampling:
        freq_bands = 2.0 ** torch.linspace( 
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=input_tensor.dtype,
            device=device,
        )
    else:
        freq_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=input_tensor.dtype,
            device=device,
        )

    for freq in freq_bands:
        encoding.append(torch.sin(input_tensor * freq))
        encoding.append(torch.cos(input_tensor * freq))

    return torch.cat(encoding, dim=-1) # Return the positional encoding
