# bottle-caps

The idea of this project it's to have a machine learning app that will detect and identify bottlecaps from a source.

## Overview

bottle-caps uses computer vision techniques, primarily based on OpenCV, to identify and match bottle caps. While the app currently operates locally, future plans include transitioning to cloud hosting. The app utilizes key OpenCV methods including **Simple Blob Detector**, **Hough Transform Circles**, and **Scale-Invariant Feature Transform (SIFT)**, each requiring specific preprocessing steps.

## Code Flow

Below is an overview of how the code operates.

### Database Creation

Currently there is a script that generates all the data and uploads it to Pinecone, the vector database used to compare the vector result from the model and the ones from the database: `scripts/generate_model.py`.

### Blob Detection

Initially, the _Hough Transform Circles_ method was used, but it resulted in multiple false positives due to the variability in circle sizes. To resolve this, the _Simple Blob Detector_ was applied first. This process involves reducing the image's color depth to three levels _(photos should ideally be taken on a white background or with high-contrast caps)_. Overlapping blobs, which might represent multiple detections of the same cap, were removed. The median size of the blobs was calculated to ensure accuracy.

- **Why Median Over Average?** Using the median rather than the average prevents large or small false positives from skewing the results. Since all bottle caps have a diameter of 37mm, the median provides a more reliable measurement.

### Hough Transform Circles

Based on the approximate radius determined from the blobs, upper and lower bounds were set. Preprocessing was applied before using Hough Transform Circles to identify the positions of all circles in the image.

### RESTNet

Currently using a RESTNet model to identify the caps, basically transform them in vector format and then saving them in Pinecone. It is a vector database great for comparing AI content.

## Example

Below is an example output of the full model. The results show areas for improvement: while the first two caps are correctly identified, the bottom two are labeled incorrectly. Speed optimizations are also needed. For additional notes on testing, refer to `ScriptsMain/note_tests/README.md`.

![image](https://user-images.githubusercontent.com/51784809/232329262-603f6ff8-a1df-423a-bb16-22454085e084.png)
