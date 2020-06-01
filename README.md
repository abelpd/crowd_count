### <u>Crowd Count Project</u>

The  goal of this project was to build a convolutional neural network with Keras to count the number people in an image of a mall webcam.

#### Example of image


<img src="https://github.com/abelpd/crowd_count/blob/master/data_preprocessing/color_frame_example.PNG?raw=true">


#### Data Source

The data was sourced from professor Chen Change Loy's personal website.
More information on the dataset can be found <a href=http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html>here.</a>

<p><u><b>For specifics on the project, refer <a href=https://github.com/abelpd/crowd_count/blob/master/Capstone%202%20-%20final.pdf> to the final report.</a></b></u></p>

---

### Key Takaways from the project
* Model complexity (in terms of number of parameters) does not translate to model performance. A model with ~60MM parameters performed better than one with ~110MM parameters.
* Improving the sample size did not boost performance in a way that was expected. Boosting the sample size by flipping the images makes the model more generalizable and helps with overfitting, however, more complexity or fine tuning of the model is required to achieve the same performance.
* Adding additional data in the form of a three dimensional tensor (RGB/HSV) did not boost performance significantly. A one dimensional tensor containing the "value" data from HSV performed almost as well as RGB channel tensor.
* Ill devised models with many parameters were sensitive to vanishing gradient or exploding gradient. In some instances, adding a LeakyRelu activation function fixed this issue.
