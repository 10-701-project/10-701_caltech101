The file, baseline-feature.mat, contains four variables that describe all 3060 images in the dataset:

'classes' is a vector of cell that contains the string of name for 102 categories (the first being background)
'classes_num' is a vector of cell that contains the number of sample images in each category (in this balanced case, 30 for all 102 categories)
'imageClass' is a vector of the classification of each image: 1 row (range from 1 to 102), 3060 columns (images)
'psix' is a matrix that contains the mapped feature using VL algorithm: 36000 rows (features), 3060 columns (images)



Maintained by Jueheng Zhu
