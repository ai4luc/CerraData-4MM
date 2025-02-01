import torch
import matplotlib.pyplot as plt
import os
import cv2
from dataset import MyDataset


def save_binary_image(image, path):
    binary_image = (~torch.isnan(image)).to(torch.uint8) * 255
    binary_image = binary_image[0].cpu().numpy() 
    cv2.imwrite(path, binary_image)


def plot_distribution(data, data_normalized, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].hist(data.ravel(), bins=256, color='blue', alpha=0.7)
    axes[0].set_title('Non normalized image')
    axes[0].set_xlabel('Pixels values')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True)

    axes[1].hist(data_normalized.ravel(), bins=256, color='orange', alpha=0.7)
    axes[1].set_title('Normalized image')
    axes[1].set_xlabel('Pixels values')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True)

    plt.savefig(save_path)
    plt.close()


#def calculate_statistics(data):
#    valid_data = data[~torch.isnan(data)]
#    data_mean = torch.mean(valid_data)
#    data_std = torch.std(valid_data)
#    data_min = torch.min(valid_data)
#    data_max = torch.max(valid_data)
#    return data_mean, data_std, data_min, data_max

def calculate_statistics(data):
    means = []
    stds = []
    mins = []
    maxs = []

    for band in range(data.shape[0]):
        valid_data = data[band][~torch.isnan(data[band])]
        means.append(torch.mean(valid_data))
        stds.append(torch.std(valid_data))
        mins.append(torch.min(valid_data))
        maxs.append(torch.max(valid_data))

    return means, stds, mins, maxs


binary_image_dir = "binary_images"
os.makedirs(binary_image_dir, exist_ok=True)

histogram_dir = "opt_histograms"
os.makedirs(histogram_dir, exist_ok=True)

device = 'cpu'
data_path = "../datasets/cerradata4mm_exp/train/"
total_data = MyDataset(dir_path=data_path)

sum_pixels = torch.zeros(total_data.in_chan, dtype=torch.float32, device=device)
sum_squared_pixels = torch.zeros(total_data.in_chan, dtype=torch.float32, device=device)
total_pixels = 0
total_valid_pixels = torch.zeros(total_data.in_chan, dtype=torch.float32, device=device)

all_images_stats = {
    'mean': [],
    'std': [],
    'min': [],
    'max': []
}

all_normalized_stats = {
    'mean': [],
    'std': [],
    'min': [],
    'max': []
}

# Durante o loop para calcular as estatísticas
for idx, (images, _) in enumerate(total_data):
    nan_mask = torch.isnan(images)
    if nan_mask.any():
        images[nan_mask] = 0
    
    images_log = torch.log1p(images)
    
    valid_pixels = ~nan_mask

    sum_pixels += images_log.sum(dim=(1, 2))
    sum_squared_pixels += (images_log ** 2).sum(dim=(1, 2))
    total_valid_pixels += valid_pixels.sum(dim=(1, 2))

    # Calcula estatísticas para cada banda
    img_means, img_stds, img_mins, img_maxs = calculate_statistics(images_log)
    all_images_stats['mean'].append(img_means)
    all_images_stats['std'].append(img_stds)
    all_images_stats['min'].append(img_mins)
    all_images_stats['max'].append(img_maxs)

    #norm_means, norm_stds, norm_mins, norm_maxs = calculate_statistics(normalized_img)
    #all_normalized_stats['mean'].append(norm_means)
    #all_normalized_stats['std'].append(norm_stds)
    #all_normalized_stats['min'].append(norm_mins)
    #all_normalized_stats['max'].append(norm_maxs)

    #plot_distribution(images.cpu().numpy(), normalized_img.cpu().numpy(), os.path.join(histogram_dir, f"histogram_{idx}.png"))

# Calcule as estatísticas médias para o dataset
dataset_stats = {
    'mean': torch.mean(torch.tensor(all_images_stats['mean']), dim=0),
    'std': torch.mean(torch.tensor(all_images_stats['std']), dim=0),
    'min': torch.min(torch.tensor(all_images_stats['min']), dim=0)[0],
    'max': torch.max(torch.tensor(all_images_stats['max']), dim=0)[0]
}

normalized_stats = {
    'mean': torch.mean(torch.tensor(all_normalized_stats['mean']), dim=0),
    'std': torch.mean(torch.tensor(all_normalized_stats['std']), dim=0),
    'min': torch.min(torch.tensor(all_normalized_stats['min']))[0],
    'max': torch.max(torch.tensor(all_normalized_stats['max']))[0]
}

# Impressão das estatísticas para cada banda
print('Statistic Report')
print('Non-normalized images statistics:')
for band in range(len(dataset_stats['mean'])):
    print(f'Band {band + 1}:')
    print(f'Mean: {dataset_stats["mean"][band]}')
    print(f'Std Dev: {dataset_stats["std"][band]}')
    print(f'Min: {dataset_stats["min"][band]}')
    print(f'Max: {dataset_stats["max"][band]}')

print('Normalized images statistics:')
for band in range(len(normalized_stats['mean'])):
    print(f'Band {band + 1}:')
    print(f'Mean: {normalized_stats["mean"][band]}')
    print(f'Std Dev: {normalized_stats["std"][band]}')
    print(f'Min: {normalized_stats["min"][band]}')
    print(f'Max: {normalized_stats["max"][band]}')

""" 
for idx, (images, normalized_img, _) in enumerate(total_data):
    nan_mask = torch.isnan(images)
    if nan_mask.any():
        images[nan_mask] = 0
    
    images_log = torch.log1p(images)
    
    valid_pixels = ~nan_mask

    sum_pixels += images_log.sum(dim=(1, 2))

    sum_squared_pixels += (images_log ** 2).sum(dim=(1, 2))

    total_valid_pixels += valid_pixels.sum(dim=(1, 2))

    img_mean, img_std, img_min, img_max = calculate_statistics(images_log)
    all_images_stats['mean'].append(img_mean)
    all_images_stats['std'].append(img_std)
    all_images_stats['min'].append(img_min)
    all_images_stats['max'].append(img_max)


    norm_mean, norm_std, norm_min, norm_max = calculate_statistics(normalized_img)
    all_normalized_stats['mean'].append(norm_mean)
    all_normalized_stats['std'].append(norm_std)
    all_normalized_stats['min'].append(norm_min)
    all_normalized_stats['max'].append(norm_max)
    
    plot_distribution(images.cpu().numpy(), normalized_img.cpu().numpy(), os.path.join(histogram_dir, f"histogram_{idx}.png"))

sar_i_mean = sum_pixels / total_valid_pixels

sar_i_var = (sum_squared_pixels / total_valid_pixels) - (sar_i_mean ** 2)
sar_i_std = torch.sqrt(sar_i_var)

dataset_stats = {
    'mean': torch.mean(torch.tensor(all_images_stats['mean'])),
    'std': torch.mean(torch.tensor(all_images_stats['std'])),
    'min': torch.min(torch.tensor(all_images_stats['min'])),
    'max': torch.max(torch.tensor(all_images_stats['max']))
}

normalized_stats = {
    'mean': torch.mean(torch.tensor(all_normalized_stats['mean'])),
    'std': torch.mean(torch.tensor(all_normalized_stats['std'])),
    'min': torch.min(torch.tensor(all_normalized_stats['min'])),
    'max': torch.max(torch.tensor(all_normalized_stats['max']))
}

print('Statistic Report')
print(f'Sum of pixels: {sum_pixels}')
print(f'Sum of squared pixels: {sum_squared_pixels}')
print(f'Total valid pixels: {total_valid_pixels}')

print('Non-normalized images statistics:')
print(f'Mean: {dataset_stats["mean"]}')
print(f'Std Dev: {dataset_stats["std"]}')
print(f'Min: {dataset_stats["min"]}')
print(f'Max: {dataset_stats["max"]}')

print('Normalized images statistics:')
print(f'Mean: {normalized_stats["mean"]}')
print(f'Std Dev: {normalized_stats["std"]}')
print(f'Min: {normalized_stats["min"]}')
print(f'Max: {normalized_stats["max"]}')

"""