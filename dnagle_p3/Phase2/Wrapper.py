# Import all the good stuff


import os
import numpy as np
import torch
import wget
import matplotlib.pyplot as plt
from Utils import *
from typing import Optional
from Network import *

if not os.path.exists('tiny_nerf_data.npz'):
    url = 'https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz'
    filename = wget.download(url)


def run_iter(height, width, focal_length, tform_cam2world,
             near_clip, far_clip, num_samples_per_ray,
             encoding_fn, batch_fn, model):
    # Compute the bundle of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length,
                                                 tform_cam2world)

    # Sample query points along each ray.
    query_points, depth_values = compute_query_points_from_rays(ray_origins, ray_directions,
                                                                near_clip, far_clip,
                                                                num_samples_per_ray)

    # Flatten the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points.
    encoded_points = encoding_fn(flattened_query_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = batch_fn(encoded_points, chunksize=128)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flat = torch.cat(predictions, dim=0)

    # Reshape the radiance field.
    radiance_field = torch.reshape(radiance_field_flat, list(query_points.shape[:-1]) + [4])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)

    return rgb_predicted


def load_data():
    """
    Function to load the TinyNeRF dataset.
    :return: images, poses, focal
    """
    data = np.load('tiny_nerf_data.npz')
    images = data['images'].astype(np.float32)
    poses = data['poses'].astype(np.float32)
    focal = np.array(data["focal"])

    return images, poses, focal


def main():
    def trans_t(t):
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

    def phi(angle):
        angle = torch.tensor(angle)
        return torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle), 0],
            [0, torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

    def theta(th):
        th = torch.tensor(th)
        return torch.tensor([
            [torch.cos(th), 0, -torch.sin(th), 0],
            [0, 1, 0, 0],
            [torch.sin(th), 0, torch.cos(th), 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

    # def pose_spherical(th, angle, radius):
    #     c2w = trans_t(radius)
    #     c2w = phi(angle/180*np.pi) @ c2w
    #     c2w = theta(th/180*np.pi) @ c2w
    #     c2w = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.float32) @ c2w
    #     return c2w
    def pose_spherical(th, angle, radius):
        c2w = torch.Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, radius],
            [0, 0, 0, 1],
        ])
        c2w = phi(angle / 180 * np.pi) @ c2w
        c2w = theta(th / 180 * np.pi) @ c2w
        c2w = torch.Tensor([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]) @ c2w
        return c2w

    names = [
        ['theta', [100, 0, 360]],
        ['phi', [-30, -90, 0]],
        ['radius', [4, 3, 5]],
    ]
    import imageio
    f = 'video.mp4'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, tform_cam2world, focal_length = load_data()  # Load the data
    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)  # Convert to torch tensors
    focal_length = torch.from_numpy(focal_length).to(device)  # Convert to torch tensors

    height, width = images.shape[1:3]  # Get the image dimensions

    near_thresh = 2.  # Near and far clipping planes
    far_thresh = 6.  # Near and far clipping planes

    testimg, testpose = images[101], tform_cam2world[101]  # Get the test image and pose
    testimg = torch.from_numpy(testimg).to(device)  # Convert to torch tensor

    images = torch.from_numpy(images[:100, ..., :3]).to(device)  # Convert to torch tensor

    num_encoding_functions = 6  # Number of encoding functions
    encode = lambda x: compute_positional_encoding(x,
                                                   num_encoding_functions=num_encoding_functions)  # Encoding function
    depth_samples_per_ray = 32  # Number of depth samples per ray

    # Optimizer parameters
    lr = 5e-3  # Learning rate
    num_iters = 1000  # Number of iterations

    # Misc parameters
    display_every = 100  # Number of iters after which stats are displayed

    model = Nerf(num_encoding_functions=num_encoding_functions)  # Initialize the model
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Initialize the optimizer

    # Lists to store the loss and error values
    list_of_error = []  # List to store  error values
    iterations = []

    from tqdm import tqdm_notebook as tqdm

    angles = torch.linspace(0, 360, 300, dtype=torch.float32)

    for i in range(num_iters):

        target_img_idx = np.random.randint(images.shape[0])  # Randomly select a target image
        target_img = images[target_img_idx].to(device)  # Get the target image
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)  # Get the target pose

        rgb_predicted = run_iter(height, width, focal_length,
                                 target_tform_cam2world, near_thresh,
                                 far_thresh, depth_samples_per_ray,
                                 encode, get_minibatches, model)  # Run the model

        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)  # Compute the loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Reset the gradients

        if i % display_every == 0:
            rgb_predicted = run_iter(height, width, focal_length,
                                     testpose, near_thresh,
                                     far_thresh, depth_samples_per_ray,
                                     encode, get_minibatches, model)  # Run the model
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)  # Compute the loss
            print("Loss Value:", loss.item())
            logloss = -10. * torch.log10(loss)

            list_of_error.append(logloss.item())
            iterations.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.title(f"{i}th iteration")
            plt.subplot(122)
            plt.plot(iterations, list_of_error)
            plt.title("Loss Plot")
            plt.show()

        print('Done!')

    # Generate the list of images
    images1 = []
    for th in angles:
        # Compute the camera-to-world transformation matrix
        c2w = pose_spherical(th, -30, 4).to(device)

        # Render the scene using TinyNeRF
        rgb = run_iter(height, width, focal_length, c2w[:3, :4], 2, 6, depth_samples_per_ray, encode, get_minibatches,
                       model)

        # Convert the image to a numpy array and append it to the list of images
        image = (255 * np.clip(rgb.clone().detach().cpu().numpy(), 0, 1)).astype(np.uint8)
        images1.append(image)
        plt.imshow(image)
        plt.show()

    # Write the list of images to a video file
    with imageio.get_writer(f, fps=27, quality=9) as writer:
        for image in images1:
            writer.append_data(image)


if __name__ == "__main__":
    main()