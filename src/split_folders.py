import splitfolders

input_folder = r"C:\WoodAI\data\data_crop"
output_folder = r"C:\WoodAI\data\data_split"

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.75, 0.15, 0.1), move=False)