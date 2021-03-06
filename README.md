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
- [ ] findCountours from openCV (see dzone article)

### Dataset Generation
- [x] Generate images of text
- [x] Generate images of text with a given font (color, size, type, bold/italics/underline)
- [x] Find an appropriate list of fonts to use
- [ ] Find an appropriate list of transformations to make data realistic
- [x] Find a corpus (UPC database)
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
* [Dropbox apply](https://www.dropbox.com/jobs/listing/794772)
* [Dzone article](https://dzone.com/articles/using-ocr-for-receipt-recognition)
* [CNN](http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/)
* [CNN Batch Norm](https://www.kaggle.com/sarvesh278/cnn-and-batch-normalization-in-tensorflow)
* [CNN LSTM CTC 2](https://github.com/weinman/cnn_lstm_ctc_ocr/blob/master/src/test.py)
* [CNN LSTM CTC 1](https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow)
