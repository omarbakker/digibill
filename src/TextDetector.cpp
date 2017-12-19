#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <cmath>
#include <map>

using namespace std;

void printRect(cv::Rect &rect);
pair<cv::Rect*, cv::Rect> rectPair(cv::Rect* r1_ptr, cv::Rect r2);
cv::Rect* nearestMean(cv::Rect to, vector <cv::Rect> &means, double &error);
double distanceBetweenLetters(cv::Rect r1, cv::Rect r2);
int getRandomRect(vector<cv::Rect> &rects);
void clusterToLinesWithKmeans(vector<cv::Rect> &boxes,
                              vector<cv::Rect> &lines,
                              double &error);
void clusterToLinesWithHierarchical(vector<cv::Rect> &boxes,
                                    vector<cv::Rect> &lines,
                                    double &error);
void overlapFilter(vector<cv::Rect> &boxes);
void outlierFilter(vector<cv::Rect> &boxes, double cutoff);

void printRect(cv::Rect &rect)
{
    printf("rect x:%i y:%i w:%i h:%i, area: %i\n",
            rect.x, rect.y, rect.width, rect.height, rect.area());
}

pair<cv::Rect*, cv::Rect> rectPair(cv::Rect* r1_ptr, cv::Rect r2)
{
    return pair<cv::Rect*, cv::Rect>(r1_ptr, r2);
}

/*
 agglomorative hierarchical clustering => O(n^3), needs optimization
 https://en.wikipedia.org/wiki/Hierarchical_clustering
 Linkage criteria: single linkage
*/
void clusterToLinesWithHierarchical(vector<cv::Rect> &boxes,
                                    vector<cv::Rect> &lines,
                                    double &error)
{
    double minDist;
    int minDistC1, minDistC2;

    // start with all the rects as individual clusters => O(n)
    lines.clear();
    for (auto & box:boxes) lines.push_back(box);

    int i = 1;
    double lastSS = DBL_MAX;
    double totalSS;
    do {

        minDist = DBL_MAX;
        totalSS = 0;

        // find the closest 2 boxes => O(n^2)
        for (int i = 0; i < lines.size()-1; i++){
            for (int j = i+1; j < lines.size(); j++){
                double dist = distanceBetweenLetters(lines[i], lines[j]);
                totalSS += dist;
                if (dist < minDist){
                    minDist = dist;
                    minDistC1 = i;
                    minDistC2 = j;
                }
            }
        }

        // merge the 2 closest boxes into one
        cv::Rect c1 = lines[minDistC1];
        cv::Rect c2 = lines[minDistC2];
        cv::Rect merged = c1 | c2;
        lines[minDistC1] = merged;
        lines.erase(lines.begin()+minDistC2);

        double diff = lastSS - totalSS;
        printf("diff %f\n", diff);
        lastSS = totalSS;

        cv::Mat img1 = cv::imread("tst.png");
        for (int i = 0; i < lines.size(); i++)
            cv::rectangle(img1, lines[i], CV_RGB(50, 200, 50));
        cv::rectangle(img1, merged, CV_RGB(0, 0, 0));
        cv::rectangle(img1, c1, CV_RGB(255, 0, 50));
        cv::rectangle(img1, c2, CV_RGB(0, 0, 255));
        cv::imshow("mser", img1);
        cv::waitKey(0);

        i++;
    } while (lines.size() > 1);

}

/*
 A special distance function optimized to give the distance between letters
 of the same line as small a distance value as possible
*/
double distanceBetweenLetters(cv::Rect r1, cv::Rect r2)
{
    // penalize vertical distance more, and use the rect base as y
    float dy = abs(pow((r1.y + r1.height) - (r2.y + r2.height), 8.0));
    float dx = abs(pow((r1.x + (r1.width/2.0)) - (r2.x + (r2.width/2.0)), 1.0));
    return dx + sqrt(dy);
}

cv::Rect* nearestMean(cv::Rect to,
                     vector <cv::Rect> &means,
                     double &error)
{
    cv::Rect* nearest;
    double nearestDist = DBL_MAX;
    for (cv::Rect& mean:means){
        double dist = distanceBetweenLetters(mean, to);
        if (dist < nearestDist){
            nearestDist = dist;
            nearest = &mean;
        }
    }
    error = nearestDist;
    return nearest;
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
represent the overlapping regions in the original vector. So if 3 boxes
 overlapped in the original vector, they would be found as one entry (the
 union of the 3) in the output vector
*/
void overlapFilter(vector<cv::Rect> &boxes )
{
    bool shouldStop = false;

    while (!shouldStop){
        shouldStop = true;
        for (int i = 0; i < boxes.size()-1; i++){
            for (int j = i+1; j < boxes.size(); j++){
                bool intersects = (boxes[i] & boxes[j]).area() > 0;
                if (intersects){
                    boxes[i] = boxes[i] | boxes[j];
                    boxes.erase(boxes.begin() + j);
                    j--; shouldStop = false;
                }
            }
        }
    }
}

/**
Given a vector of type cv:rect remove the outliers by excluding the rects
that are not within cutoff standard deviations from the average
Params:
    boxes: input cv rect vector
    cutoff: exclude rects not within this many sd from the mean
**/
void outlierFilter(vector<cv::Rect> &boxes,
                   double cutoff)
{
    double sum, mean, sd, n;

    // calculate the mean
    n = (double) boxes.size();
    sum = 0;
    for (int i = 0; i < n; i++) sum += boxes[i].area();
    mean = sum/n;

    // caluclate the standard deviation
    sum = 0;
    for (int i = 0; i < n; i++)
        sum += pow(boxes[i].area() - mean, 2.0);
    sd = sqrt(sum/(n - 1));

    // erase if too far from sd (z-score)
    boxes.erase(remove_if(boxes.begin(),
                          boxes.end(),
                          [mean, sd, cutoff](cv::Rect r){
                               return abs(((r.area() - mean)/sd)) > cutoff;
                           }),
                boxes.end());
}



int main(int argc, char *argv[])
{

    // the regions vector passed to detectRegions is the set of points/pixels
    // that belong to a maximally stable extremal region, we don't need it
    vector<vector<cv::Point> > regions;
    vector<cv::Rect> boxes, reducedBoxes, lineBoxes, bestLines, reducedLines;
    cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Ptr<cv::MSER> ms = cv::MSER::create();
    double error = DBL_MAX;
    double minError = DBL_MAX;

    // default is 60, smaller means high recall low percision
    ms->setMinArea(15);
    ms->detectRegions(img, regions, boxes);

    printf("MSER box count is %i\n", int(boxes.size()));
    outlierFilter(boxes, 1.0);
    printf("Outlier filter reduced count is %i\n", int(boxes.size()));
    overlapFilter(boxes);
    printf("Overlap filter reduced count is %i\n", int(boxes.size()));
    clusterToLinesWithHierarchical(boxes, lineBoxes, error);


    cv::Mat img2 = cv::imread(argv[1]);
    for (int i = 0; i < lineBoxes.size(); i++) {
        cv::rectangle(img2, lineBoxes[i], CV_RGB(50, 200, 50));
    }

    cv::imshow("mser", img2);
    cv::waitKey(0);
    return 0;
}
