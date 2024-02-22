import splitfolders

### Split data into Train and testing

input_folder = '../dataset/Potsdam/Potsdam_dataset2/'
output_folder = "../dataset/Potsdam/Potsdam_split_data2/"

# ratio of split are in order of train/val/test.
splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(.9, .0, .1))

