#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <cmath>
#include "clustering/src/cluster/optics.hpp"
#include <map>

using namespace std;
using namespace cluster_analysis;

pair<cv::Rect*, cv::Rect> rectPair(cv::Rect* r1_ptr, cv::Rect r2);
cv::Rect* nearestMean(cv::Rect to, vector <cv::Rect> &means, double &error);
double distanceBetween(cv::Rect r1, cv::Rect r2);
int getRandomRect(vector<cv::Rect> &rects);
void clusterToLinesWithKmeans(vector<cv::Rect> &boxes, vector<cv::Rect> lines, double &error);
void groupOverlaps(vector<cv::Rect> &boxes, vector<cv::Rect> &reducedBoxes);

pair<cv::Rect*, cv::Rect> rectPair(cv::Rect* r1_ptr, cv::Rect r2)
{
    return pair<cv::Rect*, cv::Rect>(r1_ptr, r2);
}


cv::Rect* nearestMean(cv::Rect to,
                     vector <cv::Rect> &means,
                     double &error)
{
    cv::Rect* nearest;
    double nearestDist = DBL_MAX;
    for (cv::Rect& mean:means){
        double dist = distanceBetween(mean, to);
        if (dist < nearestDist){
            nearestDist = dist;
            nearest = &mean;
        }
    }
    error = nearestDist;
    return nearest;
}

double distanceBetween(cv::Rect r1, cv::Rect r2)
{
    float x = sqrt(abs(r1.x - r2.x) + pow(r1.y - r2.y, 8));
    return x;
}

// todo: improve this to implement kmeans++
int getRandomRect(vector<cv::Rect> &rects)
{
    int randomIndex = rand() % rects.size();
    return randomIndex;
}

void clusterToLinesWithKmeans(vector<cv::Rect> &boxes,
                              vector<cv::Rect> &lines,
                              int k,
                              double &error)
{

    vector <cv::Rect> means;
    multimap <cv::Rect*, cv::Rect> clusters;

    // initialize means as random rects from boxes
    for (int i = 0; i < k; i++){
        cv::Rect mean = boxes[getRandomRect(boxes)];
        means.push_back(mean);
    }

    int max_iter = 1000;
    int iter = 0;
    double prevError = DBL_MAX;

    while (iter++ < max_iter){

        double currentError = 0;

        // empty the clusters to make room for the new assignments
        clusters.clear();

        // assign each rect to the cluster with the nearest mean
        for (cv::Rect box:boxes){
            double err;
            cv::Rect *mean = nearestMean(box, means, err);
            clusters.insert( rectPair(mean, box));
            currentError += err;
        }

        // recalculate the means
        for (int i = 0; i < means.size(); i++){
            cv::Rect* mean = &means[i];
            double meanx = 0;
            double meany = 0;
            for(auto it = clusters.begin(); it != clusters.end(); it++){

                if (mean == it->first){
                    meanx += it->second.x;
                    meany += it->second.y;
                }
            }
            meanx = meanx/clusters.count(mean);
            meany = meany/clusters.count(mean);
            means[i] = cv::Rect(meanx, meany, 0, 0);
        }

        if (currentError > 0.995 * prevError) {
            // printf("Error is %f\n", currentError);
            // printf("Error not decreasing enough to justify more loops\n");
            prevError = currentError;
            break;
        }

        prevError = currentError;
        // printf("Finished iteration %i, error is now %f\n", iter, currentError);
    }

    error = prevError;

    // create the lines
    lines.clear();
    for (int i = 0; i < means.size(); i++){
        cv::Rect mean = means[i];
        cv::Rect unionRect = cv::Rect(0,0,0,0);
        for(auto it = clusters.begin(); it != clusters.end(); it++){
            if (mean == *(it->first)){
                if (unionRect.area() == 0) unionRect = it->second;
                else unionRect = unionRect | it->second;
            }
        }
        lines.push_back(unionRect);
    }

}

/**
given vector of cv::MSER bounding boxes, return a vecor subset of boxes that
represent the overlapping regions in the original vector. So if 3 boxes overlapped
in the original vector, they would be found as one entry (the union of the 3)
 in the output vector
*/
void groupOverlaps(vector<cv::Rect> &boxes,
                   vector<cv::Rect> &reducedBoxes)
{

    // create an array to keep track of whether a cv::Rect was already grouped
    int n = boxes.size();
    bool grouped[n];
    for (int i = 0; i < n; i++) grouped[i] = false;

    for (int i = 0; i < n; i++)
        if (grouped[i] == false){
            grouped[i] = true;
            cv::Rect grouping = boxes[i];
            for (int j = i+1; j < n; j++){
                bool intersects = (boxes[i] & boxes[j]).area() > 0;
                if (intersects){
                    grouping = grouping | boxes[j]; // union
                    grouped[j] = true;
                }
            }
            reducedBoxes.push_back(grouping);
        }

}


int main(int argc, char *argv[])
{

    const double connectivityRadius = 50;
    const size_t minNeighbours = 1;
    optics opticObj = optics(connectivityRadius, minNeighbours);

    // the regions vector passed to detectRegions is the set of points/pixels
    // that belong to a maximally stable extremal region, we don't need it
    vector<vector<cv::Point> > regions;
    vector<cv::Rect> boxes, reducedBoxes, lineBoxes, bestLines, reducedLines;
    cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Ptr<cv::MSER> ms = cv::MSER::create();
    double error = DBL_MAX;
    double minError = DBL_MAX;
    const double connectivityRadius = 50;
    const size_t minNeighbours = 1;
    optics opticObj = optics(connectivityRadius, minNeighbours);

    // default is 60, smaller means high recall low percision
    ms->setMinArea(15);
    ms->detectRegions(img, regions, boxes);
    groupOverlaps(boxes, reducedBoxes);



    printf("MSER box count is %i\n", int(boxes.size()));
    printf("After special kmeans, we have %i lines\n", int(reducedLines.size()));

    for (int i = 0; i < reducedLines.size(); i++)
    {
        cv::rectangle(img, reducedLines[i], CV_RGB(0, 255, 0));
    }

    cv::imshow("mser", img);
    cv::waitKey(0);
    return 0;
}
