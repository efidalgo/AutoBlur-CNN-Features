# AutoBlur_CNN_Features
Based on the code from: https://gogul09.github.io/software/flower-recognition-deep-learning  

Script to run all the tests over Soccer, Birds, 17flowers, ImageNet-6Weapons and ImageNet-7Arthropods. 

Use the VGG16 extracted features or MobileNet into an SVM classifier.  Allows to compare the difference between the use of full images or filtered with AutoBlur method

Together with the code, it has been provided the Soccer [1] dataset, so it can be tested easily: 
- Soccer: original images 
- SoccerAutoBlurBB: original images after apply AutoBlur filtering technique and cropped them with the corresponding Bounding Box

References:
[1] J. van de Weijer, C. Schmid, Coloring Local Feature Extraction, Proc. ECCV2006, Graz, Austria, 2006.
URL: http://lear.inrialpes.fr/people/vandeweijer/data.html
