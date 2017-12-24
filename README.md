# digibill

## TODO

### Line/Word Detection
- [x] MSER segmentation
- [x] Kmeans clustering
- [ ] Kmeans++ clustering
- [x] Hierarchical Clustering
- [ ] Stopping criteria for hierarchical clustering
- [ ] DBSCAN (most likely to yield reasonable results)
- [ ] OPTICS

### Dataset Generation
- [ ] Generate images of text
- [ ] Generate images of text with a given font (color, size, type, bold/italics/underline)
- [ ] Find an appropriate list of fonts to use
- [ ] Find an appropriate list of transformations to make data realistic
- [ ] Find a corpus
- [ ] Generate the synthetic dataset using the above steps

### Design/write the training procedure
- [ ] CNN to extract features from word images
- [ ] Bi-LSTM takes sequence of CNN extracted features
- [ ] CTC classifies
- [ ] Batch norm where appropriate

## Important links
* [Dropbox Article](https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/)
* [Convert TF to Coreml](https://github.com/tf-coreml/tf-coreml)
* [OpenCV](https://opencv.org/releases.html)
