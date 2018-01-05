# MRI-Dementia-Predicting
Predicting dementia from MRI images using Neural Networks and Neuroevolution of Augmenting Topologies

## Team
Miloš Pavlić (sw29-2014) <br/>
Petar Radošević (sw31-2014) <br/>
Ognjen Francuski (sw8-2014) <br/>

## Dataset
Data is acquired on [kaggle](https://www.kaggle.com/jboysen/mri-and-alzheimers) which further referenced to [oasis-brains.org](http://www.oasis-brains.org/app/template/Tools.vm). Data is given in NIfTI-1 format (img - scan data, hdr - metadata). Using [nibabel](http://nipy.org/nibabel/) from NIfTI-1 files, we acquired slices in three axes, and saved it in png format.
