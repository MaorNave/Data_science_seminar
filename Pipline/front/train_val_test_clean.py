from data_loader_final_ver import DataLoaderLocal
from seg_class import SegmentationModel
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader ,TensorDataset
from torch.utils.tensorboard import SummaryWriter


class TrainValModels():

    def test_model(self, model_name, lr_val):
        # Initialize TensorBoard
        log_dir = "runs/segmentation_experiment"
        writer = SummaryWriter(log_dir)

        #relevant path for model weights
        path_to_trained_weights = os.path.join("C:\Maor Nanikashvili\data_science_seminar\Pipeline\input\\net_weights", model_name,
                                               f'ds_seminar_{model_name}_new_model_weights_lr_{lr_val}.pth')
        #data loader
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
        test_correct_predictions = 0
        test_total_indecies_predictions = 0

        for batch_num, test_batch in tqdm(enumerate(dataloader_test)):
            test_inputs = test_batch[0]
            test_masks = test_batch[1]
            test_inputs, test_masks = test_inputs.to(device), test_masks.to(device)

            with torch.no_grad():
                test_outputs = vgg(test_inputs)
                # Compute test accuracy
                test_predicted_labels = torch.argmax(test_outputs, dim=1)
                test_correct_predictions += torch.sum(test_predicted_labels == torch.argmax(test_masks, dim=1))
                test_total_indecies_predictions += test_inputs.size(0) * test_inputs.size(2) * test_inputs.size(3)

                test_accuracy = test_correct_predictions / test_total_indecies_predictions

                print(f"Batch test number {batch_num + 1}, Test Accuracy: {test_accuracy * 100}%")

                # Log training and validation loss/accuracy to TensorBoard
                writer.add_scalar(f'{model_name}_Accuracy_lr_{lr_val}/Test', test_accuracy * 100, batch_num)




    def train_val_model(self, model_name):
        # Initialize TensorBoard
        log_dir = "runs/segmentation_experiment"
        writer = SummaryWriter(log_dir)
        #data loader
        # load the data --> for the first time and initilize the loader class
        dataset_train = DataLoaderLocal('train')
        dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
        dataset_val = DataLoaderLocal('val')
        dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)
        # number of segmentation classes
        num_classes = 6

        vgg = SegmentationModel(num_classes, True, model_name)
        # Define loss function (example: CrossEntropyLoss for segmentation)
        criterion = nn.CrossEntropyLoss()
        # assert learning rates list
        lr_list = [0.00003, 0.0001, 0.0003, 0.001, 0.003]
        for lr_val in lr_list:
          # Define optimizer (adjust learning rate and other parameters as needed)
          optimizer = torch.optim.Adam(vgg.parameters(), lr=lr_val)

          # Training loop
          num_epochs = 10
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          vgg.to(device)

          for epoch in tqdm(range(num_epochs)):
              vgg.train()
              running_loss = 0.0
              correct_predictions = 0
              total_masks_predictions = 0
              total_indecies_predictions = 0

              for batch in tqdm(dataloader_train):
                  inputs = batch[0]
                  masks = batch[1]
                  inputs, masks = inputs.to(device), masks.to(device)
                  optimizer.zero_grad()
                  outputs = vgg(inputs)
                  loss = criterion(outputs, torch.argmax(masks, dim=1))
                  # Compute accuracy
                  predicted_labels = torch.argmax(outputs, dim=1)
                  correct_predictions += torch.sum(predicted_labels == torch.argmax(masks, dim=1))
                  total_masks_predictions += inputs.size(0)  # the total number of inputs(or outputs) that the model have triend on in each epoch
                  total_indecies_predictions += inputs.size(0)*inputs.size(2)*inputs.size(3) # the total number of inecies in each frame that the model tring to predict (224*224)

                  loss.backward()
                  optimizer.step()

                  running_loss += loss.item()

              # Validation loop
              vgg.eval()  # Set model to evaluation mode
              val_running_loss = 0.0
              val_correct_predictions = 0
              val_total_masks_predictions = 0
              val_total_indecies_predictions = 0

              for val_batch in tqdm(dataloader_val):
                  val_inputs = val_batch[0]
                  val_masks = val_batch[1]
                  val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)

                  with torch.no_grad():
                      val_outputs = vgg(val_inputs)
                      val_loss =  criterion(val_outputs, torch.argmax(val_masks, dim=1))
                      # Compute validation accuracy
                      val_predicted_labels = torch.argmax(val_outputs, dim=1)
                      val_correct_predictions += torch.sum(val_predicted_labels == torch.argmax(val_masks, dim=1))

                      val_total_masks_predictions += val_inputs.size(0)
                      val_total_indecies_predictions += val_inputs.size(0) * val_inputs.size(2) * val_inputs.size(3)

                      val_running_loss += val_loss.item()

              epoch_loss = running_loss / total_masks_predictions
              epoch_accuracy = correct_predictions / total_indecies_predictions
              val_epoch_loss = val_running_loss / val_total_masks_predictions
              val_epoch_accuracy = val_correct_predictions / val_total_indecies_predictions

              print(f"Epoch {epoch+1}, Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy * 100}%, "
                    f"Validation Loss: {val_epoch_loss}, Validation Accuracy: {val_epoch_accuracy * 100}%")

              # Log training and validation loss/accuracy to TensorBoard
              writer.add_scalar(f'{model_name}_Loss_lr_{lr_val}/Train', epoch_loss, epoch)
              writer.add_scalar(f'{model_name}_Loss_lr_{lr_val}/Validation', val_epoch_loss, epoch)
              writer.add_scalar(f'{model_name}_Accuracy_lr_{lr_val}/Train', epoch_accuracy*100, epoch)
              writer.add_scalar(f'{model_name}_Accuracy_lr_{lr_val}/Validation', val_epoch_accuracy*100, epoch)



          # Save the new weights of the model
          torch.save(vgg.state_dict(), f"/content/drive/MyDrive/ds_seminar_{model_name}_new_model_weights_lr_{lr_val}.pth")



