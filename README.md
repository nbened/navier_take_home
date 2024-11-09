# Navier Take Home
- The inferred purpose of this takehome is to prepare data to predict velocity from SDF.

## Deliverable
- The deliverables (3 data sets) are generated using 01_Data_Preparation_and_EDA.ipynb
- For a non train/test split, use prepared_tensor_data.pt
- For a train/test split, use the respective files in the data folder. 
    - train data has input headers - (x,y,sdf)
    - test data has target headers - (x,y,x_v,y_v)


# (Read First) design choice updates:
- Pytorch Geometric is discussed a lot in the planned approach, but it's actually not used because it's not needed. Airfrans' Simulation object was enough to get the data in the right headers and into PyTorch tensors.
- Only 50 airfoils were prepared due to RAM restrictions going for 1000 sims in google colab, but the code should support the whole set. 
    - Look in "Data preparation" section in 01_Data_Preparation.ipynb to change.
- the prepared data is saved as tensors as that's the most convienient for a pytorch model
- Tensors are not batched because they're different sizes. The batching method depends on the model requirements using said data.


# Initial design choices
Look in thought_stream.pdf for: 
- design choice explanations
- approach for problem
- statistics on process and bandwidth division
- a complete braindump of in the moment thinking, including wrong turns

# Testing
- to test this code, running the Data_Preparation.ipynb notebook in Google Collab is recommended.

# Data
After running Data_Preparation notebook, the data folder will contain:
- the dataset with all 5 features ready for training, not yet split between test and train (including x,y,sdf,x_v,and y_v)
- an 80/20 test and train random split of the prepared dataset, where the input is (x,y,sdf) and output is (x,y,x_v,y_v)

# Notebooks
01_Data_Preparation_and_EDA.ipynb is the main script for this assignment. Use to:
- download airfrans set
- convert airfrans set to desired tensor file
- normalize values
- download pytorch dataframe for training
- (EDA) visualization of correlations within the processed dataset
