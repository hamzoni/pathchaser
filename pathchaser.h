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
using namespace std;
using namespace cv;

class PathChaser
{
private:

    double *isobox;
    double *clrbox;

    // frame size
    Size fsize;

    // lower color and upper color range
    int *min, *max;
    Scalar lower, upper;
    
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
    double cpLimFlH; // number of high segments bound line
    double cpLimFlL; // number of low segments bound line
    int cpFlmHB; // high bound segment height px
    int cpFlmLB; // low bound segment height px

    // cluster params
    float clsAgliL;
    float clsAgliH;
    Point2f pcp, plp, prp;

    // path history

    int mgAgl; // debug value for clsAglFm
    float clsAglFm; // min acceptable angle

    // color
    Scalar clrred, clrgrn, clrblu, clrylw, clrwht;
    RNG rng;

public:
    // params container config file
    std::map<string, string> params;

    // frame counter, frame seeder, frame start
    int frc, fcp, frs;

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

    // vector
    Point2f fvect(Point2f a, Point2f b); // find vector from two points
    double lvect(Point2f v); // find length of a vectors
    double pvect(Point2f v1, Point2f v2); // find dot product of two vectors
    double avect(Point2f v1, Point2f v2); // find angle of two vectors

    // cluster point
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
    
    // points
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

    // image refining
    void ctrclean(Mat frame);
    Mat noiseremove(Mat frame);

    // main
    Mat roadline(Mat frame);

    // debug
    void video(string video, int wk);
    string print(int *arr, int n);
    void show(string title, Mat frame);
    static void showframe(string title, Mat frame, void* x);

    // misc
    // void freevect(vector<x*> vect);


    // not updated
    Mat isolate(Mat frame);
    Rect mkbox(Mat frame, double *frontier);
    Mat chunk(Mat frame);
    void refineMask(Mat &frame);
    Rect mkcbox(Mat frame);
    void minmax(Mat frame);
    void updsclr(); // update scalar masking
    
};

#endif // PATHCHASER_H
