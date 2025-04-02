# Medical Imaging (Project Summary)

## Project background
Skin cancer is one of the most common cancers around the world. It affects approximately 300 thousand people every year, with Denmark being in the top 5 countries with the most cases per population (Source: https://www.wcrf.org/preventing-cancer/cancer-statistics/skin-cancer-statistics/).

Thankfully, even though skin cancer can be fatal, early detection dramatically improves the survival rates of patients. Ideally, regular screenings would be available to everyone, but in reality, this is not always feasible. Dermatologists are not accessible everywhere and if they are, frequent dermatologist visits are also not desirable due to economic reasons. 

This project is focused on improving the skin detection process by optimizing the time medical professionals spend evaluating whether skin lesions are cancerous.  Instead of relying solely on human assessment, the goal it to use a machine learning algorithm to automate and quicken the diagnosis process, making early detection more efficient and accessible.

## Dataset description
As said prior, a key part of early detection is a review of the skin lesion that is thought to be cancerous. Eventhough, this is predominantly done by dermatologists, developing an automated programme can result in a more efficient and accessible check up for the patient. To create a useful model, we need it to be highly accurate. In order to do so we must analyse diverse types of skin lesions and this is represented in the dataset we are provided. The skin lesions photographed are of different shapes, colors and textures, which will expose the model to different types of cancerous lesions thus making its verdict more robust. The images in the dataset are also all of similar quality which is ideal for the model. To efficiently analyse the skin lesion dataset, first we will need to cleanup any distractions off the skin lesions which can degrade the model's accuracy. Most common example of this, is the hair present in the region of the patient's skin lesion. The patient's hair can obstruct the lesion and confuse the model, or thus comprimising the models verdict. To combat this problem, we will need to remove the hair off the images before we can allow the model to evaluate it. 

## Image annotations
An important component of this project is the annotation of the amount of hair in the provided pictures. All 5 members of our group have individually rated all 200 pictures on a scale from 0-2, 0 meaning no hair in the picture, 1 meaning some hair, and 2 meaning a lot of hair. Later in the project, these annotations will serve as input for model training and further analysis. Also, from these annotations we can measure inter-observer consistency between different annotators and intra-observer consistency for the same annotator. For measuring the inter-observer consistency, Cohens Kappa can be used to measure how much the annotations of two annotators agree.

## Segmentation of skin lesions
 text here 

 ## Conclusions and reflection

**Strengths:**

The presence of hair in an image can potentially influence the performance of a machine learning algorithm. To avoid this issue, we applied a hair removal algorithm to preprocess the images. The algorithm processes a selected set of images and returns hair-free versions while preserving other relevant details.

Upon reviewing the results, we observed that the algorithm performs well in some cases but encounters challenges in others. Specifically, it is effective at removing dark hairs, both in images with minimal hair (see Example 1) and those with significant hair coverage (see Example 2).

(examples here 1186
1191 - lot of hair removed)

That being said, it is crucial that the mole remains visible and retains all features necessary for accurate classification. 
To check for any modifications of essential features, which can potentially affecting diagnostic accuracy we have revised and compared the 'before' and 'after' version of each of the 200 images.

After careful examination the following insights were uncovered:

We have noticed that some successfully processed images appear slightly blurred, with certain areas looking lighter than in the original. 

(example of fading - 1180 - we can use blur cleaner to fix it)

This isssue can potentially affect the feature 'shape', which might prove to be important in diagnostics. To address this problem, incorporating a denoising step in future processing may help improve image clarity. 

**Weaknesses:** 
**White hairs not removed**

| Before | After |
|---------|---------|
| ![Alt text](example_photos/img_1287.png) | ![Alt text](example_photos/img_after_1287.png) |

One key issue we noticed in the segmented photos was that, even though the process worked well with dark hairs next to skin lesions, it often only partially removed or completely ignored white hairs in the image. This leads to a limitation, as white hairs blocking the lesions can still affect the validity and accuracy of the program.

In this case, the problem is caused by the code in inpaint_util.py, which relies on blackhat morphology. Blackhat morphology looks for dark features against a light or bright background, meaning only dark hairs are detected during segmentation. As a result, white hairs are excluded from the mask and improperly removed in the hair segmentation process.

To fix this, we can also use MORPH_TOPHAT, which detects white hairs by identifying bright features in the image.

**Size and shape of mole changed due to hair segmentation**

While reviewing our images, we also noticed that some of the moles change size and shape after hair removal. In some cases, previously round moles take on a cross-like appearance, as seen in image 1362.

| Before | After |
|---------|---------|
| ![Alt text](example_photos/img_1362.png) | ![Alt text](example_photos/img_after_1362.png) |

As the pictures show, something in the hair segmentation process is recognizing some parts of the mole as hair. This could have to do with the mole having the same color as the hair in the photo, or it can have to do with erosion. The fact that we are using a cross-shaped structuring element (MORPH_CROSS), which results in the removal of pixels in a cross-like pattern.


cross examples (adds asymetry)
img 1362
img 1369 
Lot of hair removed but size changed
img 1181
 

 
