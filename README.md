# AutoBlur_CNN_Features
Based on the code from: https://gogul09.github.io/software/flower-recognition-deep-learning  

Script to extract CNN deep features with different ConvNets, and then use them for an Image Classification task with a SVM classifier with lineal kernel over the following small datasets: Soccer [1], Birds [2], 17flowers [3], ImageNet-6Weapons[4] and ImageNet-7Arthropods[4]. 

Use the VGG16 extracted features or MobileNet into an SVM classifier.  Allows to compare the difference between the use of full images or filtered with AutoBlur method

Together with the code, it has been provided the Soccer  dataset, so it can be tested easily: 
- Soccer: original images 
- SoccerAutoBlurBB: original images after apply AutoBlur filtering technique and cropped them with the corresponding Bounding Box

References:
[1] J. van de Weijer, C. Schmid, Coloring Local Feature Extraction, Proc. ECCV2006, Graz, Austria, 2006.
URL: http://lear.inrialpes.fr/people/vandeweijer/data.html

[2] Svetlana Lazebnik, Cordelia Schmid, and Jean Ponce. A Maximum Entropy Framework for Part-Based Texture and Object Recognition. Proceedings of the IEEE International Conference on Computer Vision, Beijing, China, October 2005, vol. 1, pp. 832-838.
URL: http://www-cvr.ai.uiuc.edu/ponce_grp/data/

[3] Nilsback, M-E. and Zisserman, A. A Visual Vocabulary for Flower Classification, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2006) 
URL: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

[4] VARP Research Group Datasets.
http://pitia.unileon.es/VARP/galleries
