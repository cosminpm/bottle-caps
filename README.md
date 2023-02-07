# BottleCaps

The idea of this app is to have a database with all the caps you have then you take a photo of some caps someone has or a cap you had found on the street, with this you would compare all the caps with this photo and it would tell if you have the cap or not.

## Summary
Bottle caps is an script which uses computer vision *(mainly OpenCV)* among other algorithms to detect and match bottle caps from a databse. *Currently only works locally but the plan is to make it work with AWS servers with a lambda function.* It uses the following OpenCV methods: __Simple Blob Detector__, __Hough Transform Circles__ and __Scale-Invariant Feature Transform__. Also a different preprocess on every step of the image is necessary. 

## Flow

Here I am going to explain how the code works, I won't go into the code too much, just explaining the main actions.

### Creation of the database

### Blobs detection
My first intution was to use directly hough transform circles, but this didn't work as expected, as it detected multiple false positives as the program didn't know the size of the circles. As more than one cap can be in the photo, the radius of the circle may be different. This is the reason why I first apply __Simple Blob detector__. But first I reduce the color of the photo to 3 levels __(it is suggested to take the photo above an white paper or caps with high contrast from the background)__. From all the blobls we remove the overlapping ones, as we don't need them and might be multiple detection of the same cap. Then we get the median size for all the blobls.

- __Why median over average?__ Because sometimes there might be a big blob that is a false positive and we are not interested. If we took the average this would increase the radius and won't be the output we wanted. The same issue would be for small false positives. Also all bottle caps have the same diameter __37 mm__.

### Hough Transform Circles
Now we have an approximate radius for gettings the caps. We use this as an approximator and we take a low boud and an upper bound from this radius. Also a preprocess is applied here. Then Hough Transform Circles is applied and we have the positions of all circles.

### Scale-Invariant Feature Transform

Now we have the positions of the circles and their radius. With the database and cropping the image, we compare each image with an entry of the database. We apply our comparison method and we decide if it is a match or not, we keep the best match.

__TO DO: This tasks are currently under development or we have a plan for this__
- Have a tree structure of the database or use clustering for grouping the images, so we don't dont have to compare all the caps from the photo with all the caps from the database. Because this is an $O(n^2)$ algorithm, but if we use trees this would change to $O(nlogn)$, also clustering would be a good option as we would only have $O(nk)$ being $k$ the number of clusters we have.

- Improve the decision of a match or not by having the maximum number of keypoints detected in an image. In this way we can decide where to place the threshold. If an image has low quality it would detect less keypoints and it would not make sense to keep the threshold for a big quality image. 

- Create an app for Android and iOS app. (Currently under development)
