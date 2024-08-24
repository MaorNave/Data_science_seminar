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
        """
        Maps a single-channel segmentation image to an RGB image using a color dictionary.
        Args:
            image (ndarray or PIL Image): The single-channel segmentation image.
            color_dict (dict): A dictionary mapping label indices to RGB colors.
        Returns:
            ndarray: The RGB image.
        """
        image_array = np.array(image)  # Convert to numpy array if not already
        rgb_image = np.zeros((*image_array.shape, 3), dtype=np.uint8)  # Initialize an empty RGB array

        # Map each label in the image to its corresponding RGB color
        for key, color in color_dict.items():
            mask = image_array == key
            rgb_image[mask] = color

        return rgb_image

    def predict_data(self, model_name, lr_val, amount_of_frames):
        """
        Predicts segmentation masks for a set number of frames from the test dataset and saves the results.
        Args:
            model_name (str): Name of the model to load.
            lr_val (float): The learning rate associated with the trained model.
            amount_of_frames (int): Number of frames to process in the prediction.
        """
        # Path to the trained model weights
        path_to_trained_weights = os.path.join(
            "Pipeline\input\\net_weights",
            model_name,
            f'ds_seminar_{model_name}_new_model_weights_lr_{lr_val}.pth'
        )

        # Output directory for saving the prediction results
        output_path = os.path.join("Pipeline\output", model_name.lower())

        # Load the test data
        dataset_test = DataLoaderLocal('test')
        dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

        # Number of segmentation classes
        num_classes = 6

        # Initialize the model and load its weights
        vgg = SegmentationModel(num_classes, False, model_name)
        vgg.load_state_dict(torch.load(path_to_trained_weights, map_location=torch.device('cpu')))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg.to(device)
        vgg.eval()  # Set the model to evaluation mode

        # Process each batch (only the first batch is processed based on the condition)
        for batch_num, test_batch in tqdm(enumerate(dataloader_test)):
            if batch_num != 0:
                continue  # Skip all but the first batch

            test_inputs = test_batch[0][:amount_of_frames].to(device)
            test_masks = test_batch[1][:amount_of_frames].to(device)

            # Make predictions
            with torch.no_grad():
                test_outputs = vgg(test_inputs)
                test_predicted_labels = torch.argmax(test_outputs, dim=1)

            # Save the predicted masks as RGB images
            for ind, pred_mask in enumerate(test_predicted_labels):
                rgb_pred_mask = self.map_image_to_rgb(pred_mask, dataset_test.config['color_dict'])
                rgb_pred_mask_as_image = Image.fromarray(rgb_pred_mask, 'RGB')
                rgb_pred_mask_as_image.save(os.path.join(
                    output_path, 'pred',
                    dataloader_test.dataset.images_list_paths[ind].split('\\')[-1].replace('.jpg', '.png')
                ))
                # Copy the original input and mask images to the output directory
                shutil.copy2(dataloader_test.dataset.images_list_paths[ind], os.path.join(
                    output_path, 'input', dataloader_test.dataset.images_list_paths[ind].split('\\')[-1]
                ))
                shutil.copy2(dataloader_test.dataset.masks_list_paths[ind], os.path.join(
                    output_path, 'mask', dataloader_test.dataset.masks_list_paths[ind].split('\\')[-1]
                ))

