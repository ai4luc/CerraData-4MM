import cv2
import os
import glob

def crop_image(image_path, output_folder, patch_size=512):

    for raster in glob.iglob(image_path):

        name_file = os.path.basename(raster).split('.tif')
        print(name_file[0])

        # Read the input image
        image = cv2.imread(raster)

        # Get the dimensions of the input image
        height, width, _ = image.shape

        # Check if the image is square
        if height != width:
            raise ValueError("Input image is not square")

        # Calculate the number of patches in each dimension
        num_patches = height // patch_size

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Iterate through each patch
        for i in range(num_patches):
            for j in range(num_patches):
                # Calculate the coordinates of the top-left corner of the patch
                start_row, start_col = i * patch_size, j * patch_size

                # Extract the patch from the image
                patch = image[start_row:start_row+patch_size, start_col:start_col+patch_size]

                # Save the patch to the output folder
                patch_filename = name_file[0]+f"_{i}_{j}.tif"
                patch_path = os.path.join(output_folder, patch_filename)
                print('Salvando...', patch_path)
                cv2.imwrite(patch_path, patch)

if __name__ == "__main__":
    # Specify the input image path
    input_image_path = "Potsdam/5_Labels_all_noBoundary/label/*.tif"

    # Specify the output folder for patches
    output_folder = "Potsdam/split_dataset/label/"

    # Specify the patch size
    patch_size = 512

    # Crop the image into patches
    crop_image(input_image_path, output_folder, patch_size)

    print("Image cropped into patches successfully.")
