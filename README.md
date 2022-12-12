# Thymio
## Global navigation with the use of a camera to guide a Thymio robot around obstacles

This project was developed as part of the Micro-450 course at EPFL. The aim of this project was to guide a small robot in a map with some obstacles, the main obstacles are detected with the camera and the small obsacles are avoided by the robot itself directly, a djikstra algorithm is used to find the ideal path, the position of the robot can be estimated for a short periode of time in order to go under a bridge or recover out of the camera's field of view.



This project contain a part computer vision in order to create a map and quide the robot through it.

A part of the project is the local avoidance directly programmed on the thymio. 

A global navigation is also implemented in order to follow the path previously calculated.

An extended Kalman filter was developped for the purpose of processing all measurements and estimating the position.


## Video demo


https://user-images.githubusercontent.com/102581647/207149962-0e1a1da0-2240-4995-995a-955adc73e7f9.mp4

Axel Praplan, Oliver Ghisen, Romain Bianci, Albias Havoli
