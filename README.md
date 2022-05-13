# Final_Project_MachineLearning


This code and the project is created for a mandatory project ACIT4630 Advanced machine learning and deep learning (https://student.oslomet.no/en/studier/-/studieinfo/emne/ACIT4630/2022/HØST) Master course at Oslo Metropolitan university. The data set for this project is taken from the link (https://drive.google.com/open?id=1gqS1fkTvtuAaKj_0cn9n04ng1qDAoZ2t) 


## Project Titile "Fall detection using wearable sensor" 
## Problem Definition
Automatic fall detection (FD) devices have been developed as assertive technology throughout the last two decades. The primary purpose of fall detection systems is to detect critical falls and warn medical professionals or carers as soon as possible. Furthermore, these solutions can help the elderly and carers cope with psychological stress. Basically the problem to be solved is to distinguish fall from activities of daily life by using a wireless sensor fitted to the persons body. There are various types of fall action on the dataset and the task is to identify some of the non fall actions which are high impact events with falls. For this, we used machine learning classifiers to train the dataset and predit the fall activities by using Decision Tree, K-Nearest Neighbors, Random Forest, Support Vector Machine and a deep neural network (LSTM, CNN+LSTM). 


In the aged, falls are majorly responsible for severe injuries and death. According to the World Health Organization [1], over 30\% of the elderly, aged 64 and more, fall at least once a year. A previous study showed that nearly half of the elderly population died after laying on the floor for more than an hour within six months of a fall. Also, around 420,000 falls result in death each year. As a result of this statistic, falls are the second most significant cause of unintentional injury mortality.

Falls, particularly among the elderly, are a significant health concern worldwide. Fall detection systems that are reliable can help reduce the detrimental effects of falls. Making a fair comparison between fall detection systems and machine learning algorithms for detection is one of the significant challenges and issues identified in the research.

After falling, elderly persons and people with epilepsy may require assistance, but they may be unable to summon help owing to injuries or loss of consciousness. Several wearable fall detection systems have been created, but not everyone at risk uses them.

## Data set
The data set consists of 57.96\% falls and 42.04\% of activities of daily life. In addition, there are altogether 1570 number of records, 910 falls and 660 daily life activities. 

## Data exploration [This section is copied from data source provider]

Ten males and seven females participated in a study. A wireless sensor unit was fitted to the subject's waist and right thigh among other body parts as can be seen in Figure 1. The sensor unit comprises three tri-axial devices: accelerometer, gyroscope, and magnetometer/compass. Raw motion data was recorded along three perpendicular axes (x, y, z) from the unit with a sampling frequency of 25 Hz yielding Acc_X, Acc_Y, Acc_Z ($m/s^2$), Gyr_X, Gyr_Y, Gyr_Z (°/s) and Mag_X, Mag_Y, Mag_Z (Gauss). A set of trials consists of 20 fall actions (see list 'Fall Actions' above) and 16 activities of daily living (see list 'Non-Fall Actions' above). Each trial lasted about 15s on average. The 17 volunteers repeated each test five times. Then the peak of the total acceleration vector $\sqrt{\text{Acc_X}^2 + \text{Acc_Y}^2 + \text{Acc_Z}^2}$ was detected, and two seconds of the sequence before and after the peak acceleration were kept.



## User guide

I will suggest taking a look at the notebook [Example project](https://github.com/Shailendra995/Final_Project_MachineLearning/blob/master/fall_detection_wearable_sensor.ipynb). To run code you have to save the directory "falldetection" on running directory. 

## Technologies
Project is created with:
* Python version: 3.8 
* sklearn
* keras
* Tensorflow


## Setup
This project uses sklearn, keras and tensorflow. The best way of installing sklearn, keras and tensorlow is by using pip: `$ pip install sklearn` ,  `$ pip install keras` and `$ pip install tensorflow` respectively. 


## References
.[1] World Health Organization: Global report on falls prevention in older age. http://www.who.int/ageing/publications/Falls_prevention7March.pdf 

