#########################################################################################################
# Import ################################################################################################
#########################################################################################################

import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import pydicom
import glob
import imageio
from IPython.display import Image
import warnings
import plotly.figure_factory as ff
warnings.filterwarnings('ignore')

#########################################################################################################
# Get Data ##############################################################################################
#########################################################################################################
# **Data Description**
#
# * train.csv - the training set, contains full history of clinical information
# * test.csv - the test set, contains only the baseline measurement
# * train/ - contains the training patients' baseline CT scan in DICOM format
# * test/ - contains the test patients' baseline CT scan in DICOM format
# * sample_submission.csv - demonstrates the submission format

list(os.listdir("../input/osic"))
train_df = pd.read_csv('../input/osic/train.csv')
test_df = pd.read_csv('../input/osic/test.csv')
train_df.head()

#########################################################################################################
# Explore CSV Data ######################################################################################
#########################################################################################################
# **What does the columns represent?**
# 
# * **Patient**- a unique Id for each patient (also the name of the patient's DICOM folder)
# * **Weeks**- the relative number of weeks pre/post the baseline CT (may be negative)
# * **FVC** - the recorded lung capacity in ml
# * **Percent**- a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
# * **Age**- Age of the patient
# * **Sex**
# * **SmokingStatus**
# # Data Exploration
# **Shape of the Data**

# Interesting to note that the test set consists of only 5 images. Moreover, as given in the description, the provided test set is a small representative set of files (**copied from the training set**) to demonstrate the format of the private test set.
# **Null count and Datatype**

print('Shape of Training data: ', train_df.shape)
print('Shape of Test data: ', test_df.shape)

train_df.info()
test_df.info()

print(f"The total patient ids are {train_df['Patient'].count()}")
print(f"Number of unique ids are {train_df['Patient'].value_counts().shape[0]} ")

#########################################################################################################
# Plot CSV Data #########################################################################################
#########################################################################################################
# There are multiple records of the same patient as the number of unique ids are less than total patient ids record.
# # Visualizing the Data

new_df = train_df.groupby([train_df.Patient,train_df.Age,train_df.Sex, train_df.SmokingStatus])['Patient'].count()
new_df.index = new_df.index.set_names(['id','Age','Sex','SmokingStatus'])
new_df = new_df.reset_index()
new_df.rename(columns = {'Patient': 'freq'},inplace = True)
new_df.head()

# The 'freq' column represents number of observations for that patient

fig = px.bar(new_df, x='id',y ='freq',color='freq')
fig.update_layout(xaxis={'categoryorder':'total ascending'},title='No. of observations for each patient')
fig.update_xaxes(showticklabels=False)
fig.show()

# The number of oberservations for every unique patient in the train csv ranges from 6 to 10 wherein most of them have 9 observations.

fig = px.histogram(new_df, x='Age',nbins = 42)
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                 marker_line_width=1.5, opacity=0.6)
fig.update_layout(title = 'Distribution of Age for unique patients')
fig.show()

# We notice the range of age to be between 48-88 where we have more records for patients in the age range 64-74.

fig = px.histogram(new_df, x='Sex')
fig.update_traces(marker_color='rgb(202,158,225)', marker_line_color='rgb(48,8,107)',
                 marker_line_width=2, opacity=0.8)
fig.update_layout(title = 'Distribution of Sex for unique patients')
fig.show()

# More number of male patients than female patients.

fig = px.histogram(new_df, x='SmokingStatus')
fig.update_traces(marker_color='rgb(202,225,158)', marker_line_color='rgb(48,107,8)',
                 marker_line_width=2, opacity=0.8)
fig.update_layout(title = 'Distribution of SmokingStatus for unique patients')
fig.show()


# A big chunk of data is of patients who are Ex-smokers whereas very few patients who currently smoke.

fig = px.histogram(new_df, x='SmokingStatus',color = 'Sex')
fig.update_traces(marker_line_color='black',marker_line_width=2, opacity=0.85)
fig.update_layout(title = 'Distribution of SmokingStatus for unique patients')
fig.show()

# Records with patient who have never smoked have equal distribution of male and female patients whereas a large majority of ex-smokers are males.

fig = px.histogram(new_df, x='Age',color = 'Sex',color_discrete_map={'Male':'#EB89B5','Female':'#330C73'},marginal = 'rug',hover_data = new_df.columns)
fig.update_layout(title = 'Distribution of Age w.r.t Sex for unique patients')
fig.update_traces(marker_line_color='black',marker_line_width=1.5, opacity=0.85)
fig.show()

# Male and female records are almost distributed throughout the age range.

fig = px.histogram(new_df, x='Age',color = 'SmokingStatus',color_discrete_map={'Male':'#EB89B5','Female':'#330C73'},marginal = 'rug',hover_data = new_df.columns)
fig.update_layout(title = 'Distribution of Age w.r.t SmokingStatus for unique patients')
fig.update_traces(marker_line_color='black',marker_line_width=1.5, opacity=0.85)
fig.show()

# > **NOTE: Double click on the side legend to isolate a category**
# Patients who currently smoke show only a few occurrence along the age range.

df1 = train_df.groupby('SmokingStatus').get_group('Ex-smoker')
df2 = train_df.groupby('SmokingStatus').get_group('Never smoked')
df3 = train_df.groupby('SmokingStatus').get_group('Currently smokes')

hist_data = [df1['FVC'], df2['FVC'], df3['FVC']]

group_labels = ['Ex-Smokers', 'Never Smoked', 'Current Smokers']
colors = ['#393E46', '#2BCDC1', '#F66095']

fig = ff.create_distplot(hist_data, group_labels, colors=colors,bin_size=50,
                         show_curve=True)

# Add title
fig.update(layout_title_text='Distribution of FVC categorized by Smoking Status')
fig.update_layout( width=700,height=600)
fig.show()

# * The value of FVC for current smokers is mostly concentrated around 3000.
# * For patients who never smoked, the value remains below 4400.
# * For Ex-smokers we see a few higher values around 6000 and a large number of records are between 2000 and 3000

df1 = train_df.groupby('SmokingStatus').get_group('Ex-smoker')
df2 = train_df.groupby('SmokingStatus').get_group('Never smoked')
df3 = train_df.groupby('SmokingStatus').get_group('Currently smokes')
hist_data = [df1['Percent'], df2['Percent'], df3['Percent']]
group_labels = ['Ex-Smokers', 'Never Smoked', 'Current Smokers']
colors = ['#393E46', '#2BCDC1', '#F66095']
fig = ff.create_distplot(hist_data, group_labels, colors=colors,bin_size=1, show_curve=True)

# Add title
fig.update(layout_title_text='Distribution of Percent categorized by Smoking Status')
fig.update_layout( width=700,height=600)
fig.show()

# Where most values for 'Percent'(the patient's FVC as a percent of the typical FVC for a person of similar characteristics) ranges from 40-120, there is a chunk of data for Current smokers which have values above 140.

patient1 = train_df[train_df.Patient == 'ID00007637202177411956430']
patient2 = train_df[train_df.Patient == 'ID00012637202177665765362']
patient3 = train_df[train_df.Patient == 'ID00082637202201836229724']

patient1['text'] ='ID: ' + (patient1['Patient']).astype(str) + '<br>FVC ' + patient1['FVC'].astype(str) + '<br>Percent ' + patient1['Percent'].astype(str) + '<br>Week ' + patient1['Weeks'].astype(str)
patient2['text'] ='ID: ' + (patient2['Patient']).astype(str) + '<br>FVC ' + patient2['FVC'].astype(str)+ '<br>Percent ' + patient2['Percent'].astype(str)  + '<br>Week ' + patient2['Weeks'].astype(str)
patient3['text'] ='ID: ' + (patient3['Patient']).astype(str) + '<br>FVC ' + patient3['FVC'].astype(str) + '<br>Percent ' + patient3['Percent'].astype(str) + '<br>Week ' + patient3['Weeks'].astype(str)

fig = go.Figure()
fig.add_trace(go.Scatter(x=patient1['Weeks'], y=patient1['FVC'],hovertext = patient1['text'],
                    mode='lines+markers',marker=dict(size = 12,line_width = 2),
                    name='Ex-smoker'))
fig.add_trace(go.Scatter(x=patient2['Weeks'], y=patient2['FVC'],hovertext = patient2['text'],
                    mode='lines+markers',marker=dict(size = 12,line_width = 2),
                    name='Never smoked'))
fig.add_trace(go.Scatter(x=patient3['Weeks'], y=patient3['FVC'],hovertext = patient3['text'],
                    mode='lines+markers',marker=dict(size = 12,line_width = 2), name='Currently smokes'))

fig.update(layout_title_text='FVC vs Weeks for 3 different patients')
fig.update_layout( width=700,height=500)
fig.show()

patient1['text'] ='ID: ' + (patient1['Patient']).astype(str) + '<br>Percent ' + patient1['Percent'].astype(str) + '<br>FVC ' + patient1['FVC'].astype(str) + '<br>Week ' + patient1['Weeks'].astype(str)
patient2['text'] ='ID: ' + (patient2['Patient']).astype(str) + '<br>Percent ' + patient2['Percent'].astype(str) + '<br>FVC ' + patient2['FVC'].astype(str) + '<br>Week ' + patient2['Weeks'].astype(str)
patient3['text'] ='ID: ' + (patient3['Patient']).astype(str) + '<br>Percent ' + patient3['Percent'].astype(str) + '<br>FVC ' + patient3['FVC'].astype(str) + '<br>Week ' + patient3['Weeks'].astype(str)

fig = go.Figure()
fig.add_trace(go.Scatter(x=patient1['Weeks'], y=patient1['Percent'],hovertext = patient1['text'],
                    mode='lines+markers',marker=dict(size = 12,line_width = 2),
                    name='Ex-smoker'))
fig.add_trace(go.Scatter(x=patient2['Weeks'], y=patient2['Percent'],hovertext = patient2['text'],
                    mode='lines+markers',marker=dict(size = 12,line_width = 2),
                    name='Never smoked'))
fig.add_trace(go.Scatter(x=patient3['Weeks'], y=patient3['Percent'],hovertext = patient3['text'],
                    mode='lines+markers',marker=dict(size = 12,line_width = 2), name='Currently smokes'))

fig.update(layout_title_text='Percent vs Weeks for 3 different patients')
fig.update_layout( width=700,height=500)
fig.show()

#########################################################################################################
# Explore Image Data ####################################################################################
#########################################################################################################

# Notice that the **'FVC'** value for **Non-smoker** was the **highest** but trends for **'Percent'** for Non-smoker is less than that of the patient of **currently smokes**.
# # Visualizing Images
# We have been provided with DICOM files or "Digital Imaging and Communications in Medicine" format. It contains an image from a medical scan, like a CT scan + information about the patient. . A DICOM file has two parts: the header and the dataset. The header contains information on the encapsulated dataset. It consists of a File Preamble, a DICOM prefix, and the File Meta Elements.
# **HAVING A FIRST LOOK AT THE IMAGE**

img = "../input/osic/train/ID00009637202177434476278/100.dcm"
ds = pydicom.dcmread(img)
plt.figure(figsize = (10,10))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

# **Let's view all of the images for the first patient**

image_dir = '../input/osic/train/ID00007637202177411956430'

fig=plt.figure(figsize=(10,10))
columns = 5
rows = 6
image_list = os.listdir(image_dir)
for i in range(1, columns*rows +1):
    filename = image_dir + "/" + str(i) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

# > Looking at the previous two visuals, we notice that the **scans** for both the patients are **different**. Let's have a closer look at the scans of both patients.
image1 = '../input/osic/train/ID00007637202177411956430/8.dcm'
image2 = "../input/osic/train/ID00009637202177434476278/100.dcm"

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ds = pydicom.dcmread(image1)
ax[0].set_title('Patient 1')
ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)

ds = pydicom.dcmread(image2)
ax[1].set_title('Patient 2')
ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)

plt.show
#########################################################################################################
# Animate? GIFs? ########################################################################################
#########################################################################################################
# > Notice that the first scan has a circular border and the second one is regular.
# **Creating An Animation!!**
# 
# > If you have seen my previous works, you must know how much I love experimenting with animations. Thanks to Dan Presli for his work https://www.kaggle.com/danpresil1/dicom-basic-preprocessing-and-visualization that I was able to include this animation in my notebook.

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def set_lungwin(img, hu=[-1200., 600.]):
    lungwin = np.array(hu)
    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


scans = load_scan('../input/osic/train/ID00007637202177411956430/')
scan_array = set_lungwin(get_pixels_hu(scans))

imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)
Image(filename="/tmp/gif.gif", format='png')

#########################################################################################################
# Complete ##############################################################################################
#########################################################################################################