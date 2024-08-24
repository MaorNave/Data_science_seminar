import shutil
from data_loader_final_ver import DataLoaderLocal
from seg_class import SegmentationModel
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader


class PredictModels():

    @staticmethod
    def map_image_to_rgb(image, color_dict):
        # Convert the image to a numpy array if it's not already
        image_array = np.array(image)
        # Create an empty array for the output RGB image
        rgb_image = np.zeros((*image_array.shape, 3), dtype=np.uint8)
        # Map each value in the image array to the corresponding RGB value
        for key, color in color_dict.items():
            mask = image_array == key
            rgb_image[mask] = color

        return rgb_image

    def predict_data(self, model_name, lr_val, amount_of_frames):
        # relevant path for model weights
        path_to_trained_weights = os.path.join("C:\Maor Nanikashvili\data_science_seminar\Pipeline\input\\net_weights",
                                               model_name,
                                               f'ds_seminar_{model_name}_new_model_weights_lr_{lr_val}.pth')
        output_path = os.path.join("C:\Maor Nanikashvili\data_science_seminar\Pipeline\output", model_name.lower())
        # data loader
        # load the data --> for the first time and initilize the loader class
        dataset_test = DataLoaderLocal('test')
        dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

        # number of segmentation classes
        num_classes = 6
        # load the classifier and model weights
        vgg = SegmentationModel(num_classes, False, model_name)
        vgg.load_state_dict(torch.load(path_to_trained_weights,  map_location=torch.device('cpu')))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg.to(device)

        vgg.eval()

        for batch_num, test_batch in tqdm(enumerate(dataloader_test)):
            if batch_num!=0:
                continue
            test_inputs = test_batch[0][:amount_of_frames]
            test_masks = test_batch[1][:amount_of_frames]
            test_inputs, test_masks = test_inputs.to(device), test_masks.to(device)

            with torch.no_grad():
                test_outputs = vgg(test_inputs)
                test_predicted_labels = torch.argmax(test_outputs, dim=1)


            for ind , pred_mask in enumerate(test_predicted_labels):
                rgb_pred_mask = self.map_image_to_rgb(pred_mask, dataset_test.config['color_dict'])
                rgb_pred_mask_as_image = Image.fromarray(rgb_pred_mask, 'RGB')
                rgb_pred_mask_as_image.save(os.path.join(output_path, 'pred', dataloader_test.dataset.images_list_paths[ind].split('\\')[-1].replace('.jpg', '.png')))
                shutil.copy2(dataloader_test.dataset.images_list_paths[ind], os.path.join(output_path, 'input', dataloader_test.dataset.images_list_paths[ind].split('\\')[-1]))
                shutil.copy2(dataloader_test.dataset.masks_list_paths[ind], os.path.join(output_path, 'mask', dataloader_test.dataset.masks_list_paths[ind].split('\\')[-1]))
                plt.clf()
                plt.close()



