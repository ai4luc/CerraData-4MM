import splitfolders

### Split data into Train and testing

input_folder = '/hr/home/deso_mt/projeto_v2/dataset/cerradata4mm/novas/edge/'
output_folder = '/hr/home/deso_mt/projeto_v2/dataset/cerradata4mm/mask/'

# ratio of split are in order of train/val/test.
splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(.9, .0, .1))

