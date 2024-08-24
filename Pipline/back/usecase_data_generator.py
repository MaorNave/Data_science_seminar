from data_generator import DataGenerator



def data_generator_to_specific_dirs(main_path):
    dg = DataGenerator(main_path)

def move_rel_data_to_copy():
    DataGenerator.move_candidates_to_dataset()


def main():
    pass
    # data generator --> case to initilaize the DG class and make the relevant data folders for Train, val and test
    # main_path = "C:\Maor Nanikashvili\data_science_seminar\Pipeline\input\data\Semantic segmentation dataset"
    # data_generator_to_specific_dirs(main_path)

    # move relevant data files to copy of a train val and test folders (only relevant cutted (224*224) and augmented files --> without source files)
    # move_rel_data_to_copy()

    # move and shuffle data from different folders of train val and test (with size of shuffle).
    # DataGenerator.move_data_randomly('C:\Maor Nanikashvili\data_science_seminar\Pipeline\input\data\\test', 'C:\Maor Nanikashvili\data_science_seminar\Pipeline\input\data\\train', 0.1)

            