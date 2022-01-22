import argparse
import matplotlib as plt
import matplotlib.pyplot as pl
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
import numpy as np
import os
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_util import *
from model import _CNN

# This is a color map that you can use to plot the SHAP heatmap on the input MRI
colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30. / 255, 136. / 255, 229. / 255, l))
for l in np.linspace(0, 1, 100):
    colors.append((255. / 255, 13. / 255, 87. / 255, l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

dict_heatmap_MRI_data = {"0": [], "1": []}
sv_AD = []
sv_not_AD = []
dict_average_region = {}
# Make model instance global to be visible for test purposes
device = torch.device('cpu')
model = _CNN(fil_num=20, drop_rate=0.1)


# Returns two data loaders (objects of the class: torch.utils.data.DataLoader) that are
# used to load the background and test datasets.
def prepare_dataloaders(bg_csv, test_csv, bg_batch_size=8, test_batch_size=1, num_workers=1):
    '''
    Attributes:
        bg_csv (str): The path to the background CSV file.
        test_csv (str): The path to the test data CSV file.
        bg_batch_size (int): The batch size of the background data loader
        test_batch_size (int): The batch size of the test data loader
        num_workers (int): The number of sub-processes to use for dataloader
    '''
    # YOUR CODE HERE

    # Background dataloader initialization
    bg_dataset = CNN_Data(bg_csv)
    bg_dataloader = DataLoader(bg_dataset, batch_size=bg_batch_size, shuffle=False, num_workers=num_workers)

    # Test dataloader initialization
    test_dataset = CNN_Data(test_csv)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return bg_dataloader, test_dataloader


# Generates SHAP values for all pixels in the MRIs given by the test_loader
def create_SHAP_values(bg_loader, test_loader, mri_count, save_path):
    '''
    Attributes:
        bg_loader (torch.utils.data.DataLoader): Dataloader instance for the background dataset.
        test_loader (torch.utils.data.DataLoader): Dataloader instance for the test dataset.
        mri_count (int): The total number of explanations to generate.
        save_path (str): The path to save the generated SHAP values (as .npy files).
    '''
    # YOUR CODE HERE
    num_explanations = 0

    # Background dataset instance
    bg_MRI_data, _, _ = next(iter(bg_loader))
    bg_MRI_data = bg_MRI_data.unsqueeze(1)

    for test_MRI_data, test_filename, test_label in test_loader:
        test_MRI_data = test_MRI_data.unsqueeze(1)
        predict_label_vector = model(test_MRI_data)
        # Get vector with probabilities sum to 1
        probability_label_vector = torch.softmax(predict_label_vector, dim=1)
        for i in range(len(predict_label_vector)):
            predicted_index = probability_label_vector[i, :].data.cpu().numpy().argmax()
            # Explain only correctly predicted instances
            if predicted_index == test_label[i].item():
                # Use of GradientExplainer
                explainer = shap.GradientExplainer(model=model, data=bg_MRI_data)
                shap_values = explainer.shap_values(X=test_MRI_data, nsamples=5, rseed=1000)
                np.save(save_path + test_filename[0] + '.npy', shap_values)
                # Preparation for TASK III & IV.
                # Save information about MRI instances classified as "AD" and as "Not AD"
                # Classified as "AD"
                if test_label[i].item() == 1:
                    dict_heatmap_MRI_data["1"].append(["../data/Assignment_3_data/ADNI3/" + test_filename[0] + '.npy',
                                                       save_path + test_filename[0] + '.npy'])
                    sv_AD.append(test_filename)
                # Classified as "Not AD"
                else:
                    dict_heatmap_MRI_data["0"].append(["../data/Assignment_3_data/ADNI3/" + test_filename[0] + '.npy',
                                                       save_path + test_filename[0] + '.npy'])
                    sv_not_AD.append(test_filename)

                num_explanations += 1
                # Generate "mri_count" explanations for True Positive/Negative instances
                if num_explanations == mri_count:
                    return 1
    return 1


# Aggregates SHAP values per brain region and returns a dictionary that maps
# each region to the average SHAP value of its pixels. 
def aggregate_SHAP_values_per_region(shap_values, seg_path, brain_regions):
    '''
    Attributes:
        shap_values (ndarray): The shap values for an MRI (.npy).
        seg_path (str): The path to the segmented MRI (.nii). 
        brain_regions (dict): The regions inside the segmented MRI image (see data_utl.py)
    '''
    # YOUR CODE HERE
    dict_average_shap_values = {}
    segmented_image = nib.load(seg_path)
    segmented_data = segmented_image.get_fdata()

    for key, value in brain_regions.items():
        # Map a value from dictionary with all pixel share the same value in the MRI image.
        coordinates = np.where(segmented_data == key)
        coordinate_x = coordinates[0]
        coordinate_y = coordinates[1]
        coordinate_z = coordinates[2]
        sum_shap_values = 0
        # Iterate over all possible pixel and retrieve their Shapley value for average calculation.
        for i in range(len(coordinates[0])):
            sum_shap_values = shap_values[coordinate_x[i], coordinate_y[i], coordinate_z[i]]
        average_shap_values = sum_shap_values / len(coordinates)
        dict_average_shap_values.update({value: average_shap_values})
    return dict_average_shap_values


# Returns a list containing the top-5 most contributing brain regions to each predicted class (AD/NotAD).
def output_top_5_lst(csv_file):
    '''
    Attribute:
        csv_file (str): The path to a CSV file that contains the aggregated SHAP values per region.
    '''
    # YOUR CODE HERE
    with open(csv_file, 'r+') as csv_f:
        reader = csv.reader(csv_f)
        # Convert dictionary values to float64 from str
        dict_average_region = {key: np.float64(value) for key, value in dict(reader).items()}
        # Get 5 regions with the greatest average of average SHAP values per region for all MRIs classified as AD/NotAD
        return sorted(dict_average_region, key=dict_average_region.get, reverse=True)[:5]


# Plots SHAP values on a 2D slice of the 3D MRI.
def plot_shap_on_mri(subject_mri, shap_values):
    '''
    Attributes:
        subject_mri (str): The path to the MRI (.npy).
        shap_values (str): The path to the SHAP explanation that corresponds to the MRI (.npy).
    '''
    # YOUR CODE HERE
    # Load image and shapley values
    figure_list = []
    MRI_data_image = np.load(subject_mri)
    MRI_data_image = np.expand_dims(MRI_data_image, axis=(0, 1))
    sv = np.load(shap_values)

    # Fix dimensions to be 2D.

    # Plot without x-axis
    sv_transposed = []
    sv_transposed.append(fix_dimensions(sv[0], exclude_dim='x'))
    sv_transposed.append(fix_dimensions(sv[1], exclude_dim='x'))
    MRI_data_image_transposed = fix_dimensions(MRI_data_image, exclude_dim='x')
    shap.image_plot(sv_transposed, pixel_values=MRI_data_image_transposed, show=False)
    figure_list.append(pl.gcf())

    # Plot without y-axis
    sv_transposed = []
    sv_transposed.append(fix_dimensions(sv[0], exclude_dim='y'))
    sv_transposed.append(fix_dimensions(sv[1], exclude_dim='y'))
    MRI_data_image_transposed = fix_dimensions(MRI_data_image, exclude_dim='y')
    shap.image_plot(sv_transposed, pixel_values=MRI_data_image_transposed, show=False)
    figure_list.append(pl.gcf())

    # Plot without z-axis
    sv_transposed = []
    sv_transposed.append(fix_dimensions(sv[0], exclude_dim='z'))
    sv_transposed.append(fix_dimensions(sv[1], exclude_dim='z'))
    MRI_data_image_transposed = fix_dimensions(MRI_data_image, exclude_dim='z')
    shap.image_plot(sv_transposed, pixel_values=MRI_data_image_transposed, show=False)
    figure_list.append(pl.gcf())

    return figure_list


def probe_model(dataloader):
    correctly_classified = 0
    with torch.no_grad():
        for batch_MRI_data, MRI_filename, batch_label in dataloader:
            batch_MRI_data = batch_MRI_data.unsqueeze(1)  # Change dimensions to [batch_size_dataloader,1,182,218,182]
            predict_label_vector = model(batch_MRI_data)
            # Get vector with probabilities sum to 1
            probability_label_vector = torch.softmax(predict_label_vector, dim=1)
            for i in range(len(predict_label_vector)):
                predicted_index = probability_label_vector[i, :].data.cpu().numpy().argmax()
                if predicted_index == batch_label[i].item():
                    correctly_classified += 1
    return correctly_classified


def fix_dimensions(np_array, exclude_dim):
    if exclude_dim == 'x':
        return np_array[:, :, :, 100, :].transpose(0, 2, 3, 1)
    elif exclude_dim == 'y':
        return np_array[:, :, :, :, 100].transpose(0, 2, 3, 1)
    elif exclude_dim == 'z':
        return np_array[:, :, 100, :, :].transpose(0, 2, 3, 1)


if __name__ == '__main__':
    # TASK I: Load CNN model and instances (MRIs)
    #         Report how many of the test MRIs are classified correctly
    # YOUR CODE HERE

    # Preprocess the data and constructing the dataloaders for test and background datasets
    split_csv(csv_file='../data/Assignment_3_data/ADNI3/ADNI3.csv')
    bg_dataloader, test_dataloader = prepare_dataloaders(
        bg_csv='../data/Assignment_3_data/ADNI3/background_MRI_metadata.csv',
        test_csv='../data/Assignment_3_data/ADNI3/test_MRI_metadata.csv')
    # Load pretrained model
    checkpoint = torch.load('../data/Assignment_3_data/ADNI3/cnn_best.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Write on classified_AD.txt file how many MRIs are classified correctly from test and background datasets.
    with open('../data/Assignment_3_data/ADNI3/output/classified_AD.txt', 'w+') as f:
        result = 'Number of test MRIs correctly classified: ' + str(probe_model(test_dataloader)) + \
                 ' out of ' + str(test_dataloader.dataset.__len__())
        f.write("%s\n" % result)
        result = 'Number of background MRIs correctly classified: ' + str(probe_model(bg_dataloader)) + \
                 ' out of ' + str(bg_dataloader.dataset.__len__())
        f.write("%s\n" % result)

    # TASK II: Probe the CNN model to generate predictions and compute the SHAP
    #          values for each MRI using the DeepExplainer or the GradientExplainer. 
    #          Save the generated SHAP values that correspond to instances with a
    #          correct prediction into output/SHAP/data/
    # YOUR CODE HERE

    save_path = '../data/Assignment_3_data/ADNI3/output/SHAP/data/'
    create_SHAP_values(bg_loader=bg_dataloader, test_loader=test_dataloader,
                       mri_count=test_dataloader.dataset.__len__(), save_path=save_path)

    # TASK III: Plot an explanation (pixel-based SHAP heatmaps) for a random MRI. 
    #           Save heatmaps into output/SHAP/heatmaps/
    # YOUR CODE HERE

    # Plot for patient classified as "Not AD"
    figure_list = plot_shap_on_mri(dict_heatmap_MRI_data["0"][0][0], dict_heatmap_MRI_data["0"][0][1])
    # Save figures
    figure_list[0].savefig('../data/Assignment_3_data/ADNI3/output/SHAP/heatmaps/NotAD/pic_x.png')
    figure_list[1].savefig('../data/Assignment_3_data/ADNI3/output/SHAP/heatmaps/NotAD/pic_y.png')
    figure_list[2].savefig('../data/Assignment_3_data/ADNI3/output/SHAP/heatmaps/NotAD/pic_z.png')

    # Plot for patient classified as "AD"
    figure_list = plot_shap_on_mri(dict_heatmap_MRI_data["1"][0][0], dict_heatmap_MRI_data["1"][0][1])
    # Save figures
    figure_list[0].savefig('../data/Assignment_3_data/ADNI3/output/SHAP/heatmaps/AD/pic_x.png')
    figure_list[1].savefig('../data/Assignment_3_data/ADNI3/output/SHAP/heatmaps/AD/pic_y.png')
    figure_list[2].savefig('../data/Assignment_3_data/ADNI3/output/SHAP/heatmaps/AD/pic_z.png')

    # TASK IV: Map each SHAP value to its brain region and aggregate SHAP values per region.
    #          Report the top-5 most contributing regions per class (AD/NC) as top5_{class}.csv
    #          Save CSV files into output/top5/
    # YOUR CODE HERE

    # CLASS AD = 1
    for filename in sv_AD:
        shap_values = np.load(save_path + filename[0] + '.npy')
        shap_values = shap_values.squeeze()  # after squeeze size [2,182,218,182]
        segmented_path = '../data/Assignment_3_data/ADNI3/seg/' + filename[0] + '.nii'
        dict_average_shap_values = aggregate_SHAP_values_per_region(shap_values[1], segmented_path, brain_regions)
        if not dict_average_region:
            for key in dict_average_shap_values:
                dict_average_region.update({key: [dict_average_shap_values[key]]})
        else:
            for key in dict_average_region:
                dict_average_region[key].append(dict_average_shap_values[key])

    # Calculate the average value of average SHAP values per region for all MRIs classified as "AD"
    for key, value in dict_average_region.items():
        dict_average_region[key] = np.mean(dict_average_region[key])

    with open('../data/Assignment_3_data/ADNI3/output/top5/top_5_AD.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_average_region.items():
            writer.writerow([key, value])

    # Write on top_5_AD.txt file the 5 regions with the
    # greatest average of average SHAP values per region for all MRIs classified as "AD"
    AD_list = output_top_5_lst('../data/Assignment_3_data/ADNI3/output/top5/top_5_AD.csv')
    with open('../data/Assignment_3_data/ADNI3/output/top5/top_5_AD.txt', 'w+') as f:
        for item in AD_list:
            f.write("%s\n" % item)

    # CLASS AD = 0
    dict_average_region = {}
    for filename in sv_not_AD:
        shap_values = np.load(save_path + filename[0] + '.npy')
        shap_values = shap_values.squeeze()  # after squeeze size [2,182,218,182]
        segmented_path = '../data/Assignment_3_data/ADNI3/seg/' + filename[0] + '.nii'
        dict_average_shap_values = aggregate_SHAP_values_per_region(shap_values[0], segmented_path, brain_regions)
        if not dict_average_region:
            for key in dict_average_shap_values:
                dict_average_region.update({key: [dict_average_shap_values[key]]})
        else:
            for key in dict_average_region:
                dict_average_region[key].append(dict_average_shap_values[key])

    # Calculate the average value of average SHAP values per region for all MRIs classified as "Not AD"
    for key, value in dict_average_region.items():
        dict_average_region[key] = np.mean(dict_average_region[key])

    # Get 5 regions with the greatest average of average SHAP values per region for all MRIs classified as "Not AD"
    with open('../data/Assignment_3_data/ADNI3/output/top5/top_5_NotAD.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_average_region.items():
            writer.writerow([key, value])

    # Write on top_5_NotAD.txt file the 5 regions with the
    # greatest average of average SHAP values per region for all MRIs classified as "Not AD"
    AD_list = output_top_5_lst('../data/Assignment_3_data/ADNI3/output/top5/top_5_NotAD.csv')
    with open('../data/Assignment_3_data/ADNI3/output/top5/top_5_NotAD.txt', 'w+') as f:
        for item in AD_list:
            f.write("%s\n" % item)
