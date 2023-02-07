# BottleCaps

## Summary
Bottle caps is an script which uses computer vision *(mainly OpenCV)* among other algorithms to detect and match bottle caps from a databse. *Currently only works locally but the plan is to make it work with AWS servers with a lambda function.* It uses the following OpenCV methods: __Simple Blob Detector__, __Hough Transform Circles__ and __Scale-Invariant Feature Transform__. Also a different preprocess on every step of the image is necessary.

## Flow

My first intution was to use directly hough transform circles, but this didn't work as expected, as it detected multiple false positives as the program didn't know the size of the circles. As more than one cap can be in the photo, the radius of the circle may be different. This is the reason why I first apply __Simple Blob detector__. But first I reduce the color of the photo to 3 levels __(it is suggested to take the photo above an white paper or caps with high contrast from the background)__. Then we get the average size 
