#ifndef PATHCHASER_H
#define PATHCHASER_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <string>
#include <stack>
#include <fstream>
#include <map>
#include <ctime>
using namespace std;
using namespace cv;

// #define test_algorithm

#define debug

// #define debug_colorsearch
// #define debug_rmnoise_mask
// #define debug_chunk_trapazoid

// #define debug_birdview
// #define debug_preprocess
#define debug_noiseremove

/* graph 1: debug segmentation, clustering, fitline and fitline intersection */
// #define debug_graph1
// #define debug_segment
// #define debug_clustering
// #define debug_fitline
// #define debug_intersect

/* graph 2: debug cluster grouping */
// #define debug_graph2
// #define debug_cluster_grouping
// #define debug_threeway_detect

// #define show_cloak
// #define shop_clot

class PathChaser
{
private:

    double *isobox;
    double *clrbox;

    // frame size
    Size fsize;
    clock_t fps;
    clock_t current_ticks, delta_ticks;
    
    // lower color and upper color range
    int *min, *max;
    Scalar lower, upper;
    int upr, upg, upb;

    
    // morpho close in refineMask()
    int mc;

    // birdview params
    int bdAlpha, bdBeta, bdGam, bdF, bdDist;

    // preprocess params
    int ppLowThr, ppHighThr, ppBlurW, ppBlurH;

    // noise removal params
    int ppEroW, ppEroH, ppEroIt, ppEroType;
    int ppDltW, ppDltH, ppDltIt, ppDltType;

    // frame segments
    int cpMaxlv; // number of segments total
    int cpLimlv; // number of segments to use
    double cpSegmH; // segment height
    int cpLimDt; // maximum of points to calculate min distance
    int cpMinDt; // minimum distance between point in a cluster
    int cpDviDt; // additional distance diviation

    // line segments
    int cpLimFlH; // number of high segments bound line
    int  cpLimFlL; // number of low segments bound line
    double cpFlmHB; // high bound segment height px
    double cpFlmLB; // low bound segment height px

    // cluster params
    float clsAgliL;
    float clsAgliH;
    Point2f pcp, plp, prp;

    int mgAgl; // debug value for clsAglFm
    float clsAglFm; // min acceptable angle

    // cluster point by line params
    int cpblAgl; // angle of line to Ox
    int cpbDist; // distance of point to line

    // color
    Scalar clrred, clrgrn, clrblu, clrylw, clrwht;
    RNG rng;

    // color box trap params
    int tbxLW, tbxHW, tbxH, tbxX, tbxY;



public:
    bool clingleft;

    // params container config file
    std::map<string, string> params;

    // frame counter
    int frc;

    // frame start: start read video at a certain frame (debug feature)
    int frs;
    
    // frame seeder: number of frame to get color range
    int fcp;

    // convolution
    enum ConvolutionType {   
        /* Return the full convolution, including border */
        CONVOLUTION_FULL, 
        
        /* Return only the part that corresponds to the original image */
        CONVOLUTION_SAME,
        
        /* Return only the submatrix containing elements that were not influenced by the border */
        CONVOLUTION_VALID
    };


    // color search mask
    string dbwn6;

    // color search polygon mask
    string dbwn8;

    // remove noise mask
    string dbwn7;

    // birdview params
    string dbwn;

    // image preprocessing params
    string dbwn2;

    // noise removal params
    string dbwn3;

    // debug graph params
    string dbwn4;
    Mat raw_mat;

    // debug graph2 params
    string dbwn5;

    // debug cluster point by line
    string dbwn9;

    explicit PathChaser();
    // setup
    void settrackbar();

    // read file config
    bool readconfig(string fn);
    void trim(string &line);
    void rmcmt(string &line);
    vector<string> getpair(string str, string dl);
    void fillparam();
    float param(string key);
    
    // line
    vector<Point2f> getlinepoint(Vec4f line, Size s, int hb, int lb);
    vector<vector<Point2f>> linetopoint(vector<Vec4f> lines, Size size, int hb, int lb);
    vector<Vec4f> fitlinesMult(vector<vector<Point2f>> clusters);
    Vec4f fitlines(vector<Point2f> points);
    vector<double> findVectorsAngles(
        vector<vector<Point2f>> pointsline, 
        vector<Point2f> its,
        vector<vector<int>> couples
    );
    vector<Vec4i> findHoughLines(Mat mask, Mat &output);

    // vector
    Point2f fvect(Point2f a, Point2f b); // find vector from two points
    double lvect(Point2f v); // find length of a vectors
    double pvect(Point2f v1, Point2f v2); // find dot product of two vectors
    double avect(Point2f v1, Point2f v2); // find angle of two vectors
    Point2f mpoint(Point2f a, Point2f b);

    // cluster point
    void groupByLine(vector<Point2f> points);
    vector<vector<Point2f>> gencluster(vector<Mat> parts);
    vector<vector<Point2f>> filterclusters(vector<vector<Point2f>> clusters, int minpoint);
    vector<vector<Point2f>> groupclusters(
        vector<vector<Point2f>> clusters,
        vector<vector<int>> couples,
        vector<double> vtangles,
        double minAngle);
    double closerLeft(vector<Point2f> points, double x);
    double closerRight(vector<Point2f> points, double x);
    void findside(vector<Point2f> &left, vector<Point2f> &right, vector<Mat> frame);

    // draw
    Scalar randclr();
    void drawftlIntersect(
        vector<vector<Point2f>> pointsline, 
        vector<Point2f> its,
        vector<vector<int>> couples,
        Mat frame
    );
    void drawsegments(Mat frame);    
    void drawlines(vector<Vec4f> line, Mat frame);
    void drawcloudpoints(vector<vector<Point2f>> points, Mat frame);
    void drawclusters(vector<vector<Point2f>> clusters, Mat frame);
    void drawpoints(vector<Point2f> points, Mat frame, Scalar color);
    void drawpoint(Point2f point, Mat frame, Scalar color);
    void drawline(Vec4f line, Mat frame, Scalar color);
    void drawfc(Mat frame); // draw frame counter
    Mat draw(Mat frame, Rect box, String label);
    Mat findCtrMask(Mat bin, vector<Rect> &roirects);
    
    // points
    vector<vector<int>> kmeand(vector<int> items, vector<vector<int>> &indices, int md);
    vector<vector<Point2f>> dbscan(vector<Point2f> points, int epsilon);
    vector<vector<Point2f>> findIntersection(
        vector<vector<Point2f>> grp, 
        vector<vector<Point2f>> pvt,
        vector<Point2f> *its,
        vector<vector<int>> &couples
    );
    bool intersectLine(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);
    bool intersectLineSegment (vector<Point2f> p1, vector<Point2f> p2, Point2f &r);
    Point2f leftmostPoint(vector<Point2f> points, double startX);
    Point2f rightmostPoint(vector<Point2f> points, double endX);

    void getMeanDistance(vector<Point2f> points);
    double calcDistanceP(Point2f a, Point2f b);
    vector<Point2f> genpoints(vector<Mat> parts);
    static void debugger(int, void*);

    // segment
    vector<Point2f> converge(Mat frame);
    vector<Mat> segment(Mat frame, int n);

    // image preprocessing
    Mat bird(Mat source);
    Mat preprocess(Mat frame);
    void laplacian(Mat src, Mat &sharp, Mat &lapla);
    void thinningIteration(Mat& im, int iter);
    void thinning(Mat& im);
    Mat skeletonization(Mat inputImage);
    void conv2(
        const cv::Mat &img, const cv::Mat& kernel, 
        PathChaser::ConvolutionType type, cv::Mat& dest
    );

    // image refining
    void ctrclean(Mat frame);
    Mat noiseremove(Mat frame);
    Mat masknoisermv(Mat mask);

    // main
    double roadline(Mat frame);
    double roadlineOTSU(Mat frame);

    // debug
    void video(string video, int wk);
    string print(int *arr, int n);
    void show(string title, Mat frame);
    static void showframe(string title, Mat frame, void* x);

    // misc
    // void freevect(vector<x*> vect);

    // shape anaylsis
    Mat roadshape(Mat frame);

    // roi cropping
    Mat isolate(Mat frame);
    Rect mkbox(Mat frame, double *frontier);
    Mat trapeziumroi(Mat &frame);
    void magicwand(Mat image);

    Mat chunk(Mat frame);
    Rect mkcbox(Mat frame);

    // colors
    void minmaxroi(Mat roi, Scalar &min, Scalar &max);
    void minmax(Mat frame, bool e);
    void updsclr(); // update scalar masking

    // misc math and logic
    double sqr(double x);
    bool isInRange(double val, double l, double r);
    double calcPointDist(Point2f M);
    void countsort( vector<int> &v); // where A is in and B is out
    Point2f aglpoint(int y);

    // misc
    void testAlgorithm();

    // transform
    void waveletTransform(
        const cv::Mat& img, 
        cv::Mat& edge, 
        double threshold
    );

    // angle calculate
    double calcAngle(vector<Point2f> left, vector<Point2f> right);

    // convert radian to degree - phi is diviation of number
    double rtd(double rad, double phi); 
};

#endif // PATHCHASER_H


/*
lower 17 181 126
upper 25 249 179
*/