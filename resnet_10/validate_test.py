import torch
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from typing import Tuple, Dict
from test import ResNet10
from convert_jax_to_pytroch import load_resnet10_params
from hil_serl.serl_launcher.serl_launcher.networks.reward_classifier import create_classifier
import os
import flax.linen as nn
from functools import partial
from hil_serl.serl_launcher.serl_launcher.vision.resnet_v1 import MyGroupNorm, ResNetBlock

from jax import config
config.update('jax_disable_jit', True)

def get_encoder_def_output(classifier, image_key="image"):
    def forward_fn(params, obs):
        return classifier.apply_fn(
            {"params": params},
            obs,  # Input is wrapped in a dict
            method=lambda module, x: module.encoder_def.encoder[image_key].pretrained_encoder(
                x[image_key]  # Extract tensor from dict
            )
        )
    # return jax.jit(forward_fn)
    return forward_fn

def load_and_preprocess_image(image_path: str) -> Tuple[torch.Tensor, jnp.ndarray]:
    """
    Load and preprocess an image for both PyTorch and JAX models.
    Returns both PyTorch and JAX tensors.
    """
    # Standard ImageNet preprocessing
    torch_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                    std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=[0, 0, 0], 
        #                    std=[1, 1, 1])
    ])
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')

    torch_img = torch_transform(image)

    # torch_img = torch.zeros((3, 128, 128), dtype=torch.float32)

    # Convert to JAX array with same preprocessing
    # np_img = np.array(image)
    # jax_img = jnp.array(np_img)
    # back_ = np.array(jax_img)
    # print("JAX after numpy conversion : ", jax_img.mean().item())
    
    # Add batch dimension
    torch_img = torch_img.unsqueeze(0)
    # torch_img = torch_img.half()
    print("TORCh after numpy squeeze : ", torch_img.mean().item())
    # jax_img = jax_img[None, ...]

    jax_img = jnp.transpose(jnp.array(torch_img.numpy(), dtype=jnp.float32), (0, 2, 3, 1))
    print("JAX after numpy conversion : ", jax_img.mean().item())

    jax_input = {'image': jax_img}
    
    return torch_img, jax_input

def compute_feature_similarity(torch_features: torch.Tensor, 
                             jax_features: jnp.ndarray) -> Dict[str, float]:
    """
    Compute various similarity metrics between PyTorch and JAX features.
    """

    torch_features = torch_features.permute(0, 2, 3, 1)

    # Convert to numpy for consistent computation
    torch_features = torch_features.detach().cpu().numpy()
    jax_features = np.array(jax_features)
    
    # Flatten features
    torch_flat = torch_features.reshape(-1)
    jax_flat = jax_features.reshape(-1)

    assert torch_features.shape == jax_features.shape, \
        f"Shape mismatch: {torch_features.shape} vs {jax_features.shape}"
    
    # Compute metrics
    mse = np.mean((torch_flat - jax_flat) ** 2)
    cosine_sim = np.dot(torch_flat, jax_flat) / (np.linalg.norm(torch_flat) * np.linalg.norm(jax_flat))
    max_diff = np.max(np.abs(torch_flat - jax_flat))
    
    return {
        'mse': float(mse),
        'cosine_similarity': float(cosine_sim),
        'max_absolute_difference': float(max_diff)
    }


def benchmark_models(torch_model, jax_func, image_paths: list, num_runs: int = 100):
    """
    Comprehensive benchmark comparing PyTorch and JAX models.
    """
    torch_model.eval()  # Set PyTorch model to evaluation mode

    results = {
        'feature_similarities': [],
        'torch_times': [],
        'jax_times': []
    }
    
    for image_path in image_paths:
        torch_img, jax_img = load_and_preprocess_image(image_path)
        
        print("computing ....")

        # # Warmup run
        # with torch.no_grad():
        #     _ = torch_model(torch_img)
        # _ = jax_func(classifier.params, jax_img)
        
        # Benchmark runs
        torch_features_list = []
        jax_features_list = []
        
        for _ in range(num_runs):
            # Time PyTorch forward pass
            start_time = time.perf_counter()
            with torch.no_grad():
                torch_features = torch_model(torch_img)
            torch_time = time.perf_counter() - start_time
            results['torch_times'].append(torch_time)
            torch_features_list.append(torch_features)

            print("+====================================+")
            
            # Time JAX forward pass
            start_time = time.perf_counter()
            jax_features = jax_func(classifier.params, jax_img)
            jax_time = time.perf_counter() - start_time
            results['jax_times'].append(jax_time)
            jax_features_list.append(jax_features)
        
        # Compute feature similarities
        for t_feat, j_feat in zip(torch_features_list, jax_features_list):
            similarity = compute_feature_similarity(t_feat, j_feat)
            results['feature_similarities'].append(similarity)
    
    # Compute summary statistics
    summary = {
        'mean_torch_time': np.mean(results['torch_times']),
        'mean_jax_time': np.mean(results['jax_times']),
        'mean_mse': np.mean([s['mse'] for s in results['feature_similarities']]),
        'mean_cosine_sim': np.mean([s['cosine_similarity'] for s in results['feature_similarities']]),
        'mean_max_diff': np.mean([s['max_absolute_difference'] for s in results['feature_similarities']])
    }
    
    return results, summary
    # return results

def print_benchmark_results(summary: Dict[str, float]):
    """
    Print benchmark results in a readable format.
    """
    print("\n=== Benchmark Results ===")
    print(f"\nTiming Comparison:")
    print(f"PyTorch mean inference time: {summary['mean_torch_time']*1000:.2f} ms")
    print(f"JAX mean inference time: {summary['mean_jax_time']*1000:.2f} ms")
    
    print(f"\nFeature Similarity Metrics:")
    print(f"Mean MSE: {summary['mean_mse']:.8f}")
    print(f"Mean Cosine Similarity: {summary['mean_cosine_sim']:.8f}")
    print(f"Mean Maximum Absolute Difference: {summary['mean_max_diff']:.8f}")
    
    print("\nAnalysis:")
    if summary['mean_cosine_sim'] > 0.9999:
        print("✓ Features are nearly identical (cosine similarity > 0.9999)")
    else:
        print("⚠ Features show some differences")
        
    if summary['mean_max_diff'] < 1e-5:
        print("✓ Maximum differences are negligible")
    else:
        print(f"⚠ Notable maximum differences detected: {summary['mean_max_diff']:.8f}")
        
    speedup = summary['mean_torch_time'] / summary['mean_jax_time']
    print(f"\nSpeed Comparison: JAX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch")


# Example usage
if __name__ == "__main__":
    import argparse
    from transformers import ResNetConfig
    
    parser = argparse.ArgumentParser(description='Benchmark JAX and PyTorch ResNet models')
    # parser.add_argument('--image_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for timing benchmark')
    args = parser.parse_args()

    image_dir = "resnet_10/images/"

    # Create a sample dictionary
    sample = {
        "image": jnp.ones((1, 128, 128, 3))  # Single image, shape: NHWC format
    }

    classifier = create_classifier(jax.random.PRNGKey(0), sample, ["image"], n_way=2)
    jax_func = get_encoder_def_output(classifier)
    
    # Create PyTorch model
    config = ResNetConfig(
        num_channels=3,
        embedding_size=64,
        hidden_act="relu",
        hidden_sizes=[64, 128, 256, 512],
        depths=[1, 1, 1, 1],
    )
    torch_model = ResNet10(config)
    
    # Load weights
    jax_params = load_resnet10_params()
    torch_model.load_jax_weights(jax_params)
    
    # Get list of image paths
    import glob
    import os
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    # Run benchmark
    results, summary = benchmark_models(torch_model, jax_func, image_paths, args.num_runs)
    print_benchmark_results(summary)
