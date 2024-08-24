from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import yaml
import torch.nn.functional as F
from skimage.util import view_as_blocks
import torchvision.transforms as transforms
import torch
import random
import shutil



class DataGenerator(Dataset):

    main_output_path = "C:\Maor Nanikashvili\data_science_seminar\Pipeline\input\data"


    @staticmethod
    def move_data_randomly(input_folder_path, output_folder_path, files_perc):
        # Define the image and mask folder names
        image_folder = 'images'
        mask_folder = 'masks'

        # Define the full paths for the image and mask folders in input and output directories
        input_images_path = os.path.join(input_folder_path, image_folder)
        input_masks_path = os.path.join(input_folder_path, mask_folder)

        output_images_path = os.path.join(output_folder_path, image_folder)
        output_masks_path = os.path.join(output_folder_path, mask_folder)


        # Get a list of all image files in the input images folder
        image_files = os.listdir(input_images_path)

        # Determine the number of files to move (25%)
        num_files_to_move = int(len(image_files) * files_perc)

        # Randomly select the files to move
        files_to_move = random.sample(image_files, num_files_to_move)

        for file_name in files_to_move:
            # Define the full file paths for the images and masks
            src_image = os.path.join(input_images_path, file_name)
            dest_image = os.path.join(output_images_path, file_name)

            # Move the image file
            shutil.move(src_image, dest_image)

            # Get the corresponding mask file name (assuming it has the same name as the image)
            src_mask = os.path.join(input_masks_path, file_name.replace('jpg', 'png'))
            dest_mask = os.path.join(output_masks_path, file_name.replace('jpg', 'png'))

            # Move the mask file
            shutil.move(src_mask, dest_mask)

        print(f"Moved {num_files_to_move} images and their corresponding masks.")


    @staticmethod
    def move_candidates_to_dataset():
        for folder in ['train', 'val', 'test']:
            main_full_path = os.path.join(DataGenerator.main_output_path, folder)

            for file in np.sort(os.listdir(os.path.join(main_full_path, 'images'))):
                full_image_path = os.path.join(main_full_path, 'images', file)
                full_mask_path = os.path.join(main_full_path, 'masks', file.replace('.jpg', '.png'))
                image = Image.open(full_image_path)
                mask = Image.open(full_mask_path)
                if image.size!=(224,224):
                    continue
                else:
                    if not os.path.exists(os.path.join(main_full_path, 'images').replace(folder, folder+'_2')):
                        os.makedirs(os.path.join(main_full_path, 'images').replace(folder, folder+'_2'))
                        os.makedirs(os.path.join(main_full_path, 'masks').replace(folder, folder+'_2'))
                    image.save(os.path.join(main_full_path, 'images', file).replace(folder, folder+'_2'))
                    mask.save(os.path.join(main_full_path, 'masks', file.replace('.jpg', '.png')).replace(folder, folder + '_2'))


    @staticmethod
    def save_images(file_name, tiles_set_name, image):
        if 'jpg' in file_name:
            if os.path.exists(os.path.join(DataGenerator.main_output_path, tiles_set_name.split('_')[0], 'images')):
                image.save(os.path.join(DataGenerator.main_output_path, tiles_set_name.split('_')[0], 'images' ,file_name))
            else:
                os.makedirs(os.path.join(DataGenerator.main_output_path,  tiles_set_name.split('_')[0], 'images'))
                image.save(os.path.join(DataGenerator.main_output_path, tiles_set_name.split('_')[0],'images' ,file_name))
        else:
            if os.path.exists(os.path.join(DataGenerator.main_output_path, tiles_set_name.split('_')[0], 'masks')):
                image.save(os.path.join(DataGenerator.main_output_path, tiles_set_name.split('_')[0],'masks' ,file_name))
            else:
                os.makedirs(os.path.join(DataGenerator.main_output_path,  tiles_set_name.split('_')[0], 'masks'))
                image.save(os.path.join(DataGenerator.main_output_path, tiles_set_name.split('_')[0], 'masks',file_name))


    def crop_image_or_mask_to_closest_size(self, frame):
        SIZE_X = (frame.shape[1] // self.patch_size) * self.patch_size  # Nearest size divisible by our patch size
        SIZE_Y = (frame.shape[0] // self.patch_size) * self.patch_size  # Nearest size divisible by our patch size
        frame = Image.fromarray(frame)
        frame = frame.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
        frame = np.array(frame)

        return frame

    def rotate_and_flip_frames(self, image, mask, tiles_set_name, names_tup):
        patch_image_name, patch_mask_name = names_tup
        angle_list = [0,90,180,270]
        # Save transformed patches
        for i, angle in enumerate(angle_list):  # 4 rotations
            rotated_patch_image = image.rotate(angle)
            rotated_mask = mask.rotate(angle)
            for j in range(4):  # 2 flips
                frame_number = str((i * len(angle_list)) + (j + 1))
                if j == 0:
                    flipped_image = rotated_patch_image.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped_mask = rotated_mask.transpose(Image.FLIP_LEFT_RIGHT)
                elif j == 1:
                    flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)
                    flipped_mask = flipped_mask.transpose(Image.FLIP_TOP_BOTTOM)
                elif j ==2:
                    flipped_image = rotated_patch_image.transpose(Image.FLIP_TOP_BOTTOM)
                    flipped_mask = rotated_mask.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    flipped_image = rotated_patch_image
                    flipped_mask = rotated_mask

                self.save_images('.'.join([patch_image_name.split('.')[0]+'_'+frame_number, patch_image_name.split('.')[-1]])
                                 , tiles_set_name, flipped_image)
                self.save_images('.'.join([patch_mask_name.split('.')[0]+'_'+frame_number, patch_mask_name.split('.')[-1]])
                                 , tiles_set_name, flipped_mask)



    def get_data_from_paths(self, paths_list_name):
        paths_list = getattr(self, paths_list_name)
        patched_images = []
        patched_onehot_masks = []
        for tile_num, paths_batch in enumerate(paths_list):
            img_paths, mask_paths = paths_batch
            for img_path, mask_path in zip(img_paths, mask_paths):
                image = np.array(Image.open(img_path).convert("RGB"))
                mask = np.array(Image.open(mask_path).convert("RGB"))

                crop_image = self.crop_image_or_mask_to_closest_size(image)
                crop_mask = self.crop_image_or_mask_to_closest_size(mask)

                self.save_images('tile_num_'+str(tile_num)+'_'+img_path.split('\\')[-1] ,paths_list_name, Image.fromarray(crop_image))
                self.save_images('tile_num_'+str(tile_num)+'_'+mask_path.split('\\')[-1] ,paths_list_name, Image.fromarray(crop_mask))

                patches_image = view_as_blocks(crop_image,
                                               block_shape=(self.patch_size, self.patch_size, crop_image.shape[2]))
                patches_mask = view_as_blocks(crop_mask, block_shape=(self.patch_size, self.patch_size, crop_mask.shape[2]))

                for i in range(patches_image.shape[0]):
                    for j in range(patches_image.shape[1]):
                        frame_number = str((i * patches_image.shape[1]) + (j + 1))
                        patch_image = patches_image[i, j, 0]
                        patch_mask = patches_mask[i, j, 0]
                        patch_image_name = '.'.join(['tile_num_'+str(tile_num)+'_'+img_path.split('\\')[-1].split('.')[0]+'_'+frame_number,
                                                     img_path.split('\\')[-1].split('.')[-1]])
                        patch_mask_name = '.'.join(['tile_num_'+str(tile_num)+'_'+mask_path.split('\\')[-1].split('.')[0] + '_' + frame_number,
                                                   mask_path.split('\\')[-1].split('.')[-1]])
                        self.save_images(patch_image_name
                                         , paths_list_name, Image.fromarray(patch_image))
                        self.save_images(patch_mask_name
                                         , paths_list_name, Image.fromarray(patch_mask))

                        self.rotate_and_flip_frames(Image.fromarray(patch_image), Image.fromarray(patch_mask), paths_list_name,(patch_image_name, patch_mask_name))



    def __init__(self, root_dir):
        self.patch_size = 224
        self.root_dir = root_dir
        self.tiles = [f"Tile {i}" for i in range(1, 9)]  # Assuming tiles are named tile_1, tile_2, ..., tile_8
        self.train_tiles = self.tiles[3:5] + self.tiles[6:]  # Tiles for training
        self.val_tiles = self.tiles[5:6]
        self.test_tiles = self.tiles[:3]  # Tiles for testing
        self.train_data_paths = []
        self.val_data_paths = []
        self.test_data_paths = []


        # Load images and masks
        for tile in self.tiles:
            images = []
            masks = []
            images_dir = os.path.join(root_dir, tile, "images")
            masks_dir = os.path.join(root_dir, tile, "masks")

            image_files = sorted(os.listdir(images_dir))
            mask_files = sorted(os.listdir(masks_dir))

            for img_file, mask_file in zip(image_files, mask_files):
                img_path = os.path.join(images_dir, img_file)
                mask_path = os.path.join(masks_dir, mask_file)

                images.append(img_path)
                masks.append(mask_path)

            if tile in self.train_tiles:
                self.train_data_paths.append([images, masks])
            elif tile in self.val_tiles:
                self.val_data_paths.append([images, masks])
            else:
                self.test_data_paths.append([images, masks])

        for data_type_path in ['train_data_paths', 'val_data_paths', 'test_data_paths']:
            self.get_data_from_paths(data_type_path)



