# HPC-Project2

## Summary 

In ~/src, there are three files:
1. src/read_download.ipynb for reading and generating the matrices
2. src/Section_2-4.ipynb code for section 2 to 4
3. src/Section_5.py code for section 5

And in ~/Dataset, we provide pictures and original dataset for section 2, 4, and 5.
*Note* Due to moodle submission restriction, we'll delete the dataset.
To clone the project for more information on dataset, use the following link:
https://github.com/CartmanRolex/HPC-Project2.git


In the code, we often call SRHT by FJLT.
## Notes

Overleaf link for project report(only for review): [HPC Project](https://www.overleaf.com/read/txxrhxfnrmqf#6b7896)

## Dataset Setup
Due to file size limitations, the datasets are not included in this repository. 
Please follow these steps:

1. Run the download script: `./download_datasets.sh`
2. Download the required datasets:
   - denseData_test1.csv (510.31 MB)
   - mnist.scale (109.51 MB)
   - mnist_780 (113.03 MB)
3. Place the downloaded files in the `Dataset/` folder 


## Section 2-4
Please see src/Section_1-4.ipynb

We obtained three matrix A_1, A_2, and A_3. 
Each one is stored at:
/Dataset/A_1_polyDecayMatrix.npy in .npy format.

And we summarise the results about errors for each matrix at: 
Dataset/Section_2/Section_2_test_3_Gaussian_for_loadMatrixA_1.csv 

In section4:  
Similarly, all .png and .csv files are stored, for example as:
- Dataset/Section_4/Section_4_A1_fjlt.csv
- Dataset/Section_4/Section_4_A1_fjlt.png

Please be noted that the current accurate path name has been modified.

## Section 5

Please see src/Section_5.py