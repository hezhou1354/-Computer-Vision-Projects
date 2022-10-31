# Spatial Enabled Artificial Intelligence Projects in Python

### Project 1 -  Query on PostgreSQL and Apache Sedone

- Use existing spatial data to prepare feature vectors capturing geographic context to support spatial artificial intelligence tasks.
- Use PostgreSQL with PostGIS and Apache Sedona (formerly GeoSpark) to perform spatial queries. 
- Compare their performance in terms of query efficiency using the same query tasks on both platforms. 

### Project 2 - Air Quality Prediction

- Use machine learning methods to support real-world spatial artificial intelligence tasks. 
- Implement the approach proposed in the paper “Mining Public Datasets for Modeling Intra-City PM2.5 Concentrations at a Fine Spatial Resolution” [Lin et al. 2017]. The approach aims to generate fine-grained air quality prediction over a region using sensor observations (PM2.5 concentrations) and geographic feature data. 
- Apply clustering, classification, and regression techniques to solve the problem. 

### Project 3 - Road Semantic Segmentation

- Work on the semantic segmentation task to detect road pixels from imagery. 
- Train a segmentation model called Fully Convolutional Network (FCN). 
- Implement a UNet model from scratch, in addition to using the FCN from the torchvision package.

### Project 4 - Plane Imagery Detection

- Perform object detection for detecting planes in imagery.
- Adapt a pretrained Faster-RCNN model using the Plane dataset with transfer learning. 

### Project 5 -  Spatio-temporal Prediction 

- Forecast the future content of a video conditioned on previous frames using moving MNIST dataset and sea surface temperature dataset.
- Implement the Convolutional LSTM Network ([ConvLSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)) for the task.
- Compare the performance (i.e., convergence speed and accuracy) w/wo the teacher forcing strategy. 