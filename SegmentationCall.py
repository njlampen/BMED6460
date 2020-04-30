modelfilepath = '___Model Filepath ___ .h5'
path = '___Dataset Filepath___'
output_path = '__Results Filepath___  /01_RES'
output_path_2 = '___Results Filepath for UVA-NL tracking __ /01_RES_2'
predict_dataset(path, '01', modelfilepath , output_path, viz=True, init=False)
predict_dataset_2(output_path, output_path_2, threshold=0.15)