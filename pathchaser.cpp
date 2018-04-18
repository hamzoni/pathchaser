#include "pathchaser.h"
#define debug
// #define debug_birdview
// #define debug_preprocess
// #define debug_noiseremove

/* graph 1: debug segmentation, clustering, fitline and fitline intersection */
// #define debug_graph1
// #define debug_segment
// #define debug_clustering
// #define debug_fitline
// #define debug_intersect

/* graph 2: debug cluster grouping */
// #define debug_graph2
// #define debug_cluster_grouping
#define debug_threeway_detect

#define dev

PathChaser::PathChaser()
{
    this->isobox = new double[4 * sizeof(double)]
        {0.5, 0.0, 0.0, 0.0}; // T - R - B - L
    this->clrbox = new double[4 * sizeof(double)]
        {0, 15, 200, 50}; // X, Y, W, H

    #ifndef dev
    this->readconfig("pro.conf");

    #else
    this->readconfig("dev.conf");

    this->min = new int[3 * sizeof(int)] {5, 78, 56};
    this->max = new int[3 * sizeof(int)] {34, 255, 204};
    this->fcp = 0;
    this->frs = 1;

    this->pcp = Point2f(-1, -1); // previous center point
    this->plp = Point2f(-1, -1); // previous left point
    this->prp = Point2f(-1, -1); // previous right point


    #endif

    this->frc = 0;
    this->mc = 3;

    this->settrackbar();
}

Mat PathChaser::roadline(Mat frame) {
    if (this->cpLimlv <= 0 || this->cpLimlv > this->cpMaxlv) {
        this->cpLimlv = this->cpMaxlv;
    }

    this->fsize = Size(frame.cols, frame.rows);
    // set segment height
    double seg_h = ((double) frame.rows / this->cpMaxlv);
    this->cpSegmH = seg_h * (this->cpLimlv + 1);
    this->cpFlmHB = seg_h * (this->cpLimFlH + 1);
    this->cpFlmLB = seg_h * (this->cpLimFlL + 1);

    Mat image = frame.clone();
    this->raw_mat = image.clone();

    image = this->bird(frame);

    image = this->preprocess(image);

    image = this->noiseremove(image);

    vector<Mat> parts = this->segment(image, this->cpMaxlv);

    // gen points and clustering same time

    vector<Point2f> points = this->genpoints(parts);

    // vector<vector<Point2f>> gclusters = this->gencluster(parts);

    Mat can;
    Canny(image, can, 50, 100);
    show("can", can);


    // clustering #1
    vector<vector<Point2f>> clusters = this->dbscan(points, 40);

    // filter clustering 
    clusters = this->filterclusters(clusters, 3);

    vector<Vec4f> lines = this->fitlinesMult(clusters);
    
    // lane detection and classification
    vector<vector<Point2f>> pointsline = this->linetopoint(lines, this->fsize, this->cpFlmHB, this->cpFlmLB);
    
    vector<Point2f> its;
    vector<vector<int>> couples;

    this->findIntersection(clusters, pointsline, &its, couples);

    vector<double> vtangles = this->findVectorsAngles(pointsline, its, couples);

    // clustering 2
    vector<vector<Point2f>> clusters2 = this->groupclusters(clusters, couples, vtangles, this->clsAglFm);


    /* DEBUG PART */
    #ifdef debug

        #ifdef debug_graph1
        Mat graph = Mat::zeros(frame.size(), CV_8UC3);

        #ifdef debug_segment
        this->drawsegments(graph);
        #endif

        #ifdef debug_clustering
        this->drawclusters(clusters, graph);
        #endif

        #ifdef debug_fitline
        this->drawlines(lines, graph);
        #endif

        #ifdef debug_intersect
        
        // end point of fitline
        this->drawcloudpoints(pointsline, graph);

        // intersect point
        this->drawpoints(its, graph, this->clrred);

        // vector intersect line
        this->drawftlIntersect(pointsline, its, couples, graph);

        #endif
        
        show(this->dbwn4, graph);
        #endif // debug graph 1

        #ifdef debug_graph2
        Mat graph2 = Mat::zeros(frame.size(), CV_8UC3);
        this->drawclusters(clusters2, graph2);
        show(this->dbwn5, graph2);
        #endif
        

        // if (clusters2.size() >= 1) waitKey();
    #endif

    return image;
}

vector<vector<Point2f>> PathChaser::gencluster(vector<Mat> parts) {
    vector<vector<Point2f>> clusters;
    vector<Point2f> points;

    double w = parts.at(0).cols;
    double h = parts.at(0).rows;
    Size fsize = Size(w, h * parts.size());
   
    // get all points
    for (int i = parts.size() - 1; i >= 0; i--) {
        Mat part = parts.at(i);
        double y = h * i;
        vector<Point2f> comets = this->converge(part);
        for (int j = 0; j < comets.size(); j++) {
            points.push_back(Point2f(comets.at(j).x, y));
        }
    }

    /* analyze all points */

    vector<vector<Point2f>> groups = this->dbscan(points, 60);
    vector<vector<Point2f>> twcrps; // 3-ways cross road points
    vector<vector<Point2f>> sdpths; // side path road points

    int k = 0;
    for_each(groups.begin(), groups.end(), [&](vector<Point2f> group) {
        
        // find lowest point - v bottom
        Point2f vtp = group.at(0);
        for_each(group.begin(), group.end(), [&](Point2f gp) {
            if (gp.y > vtp.y) vtp = Point2f(gp.x, gp.y);
        });

        // find left cluster and right cluster
        vector<Point2f> lgp, rgp;
        Point tlc = group.at(0); // top left corner (xmin, ymin)
        Point brc = group.at(0); // bottom right corner (xmax, ymax)
        for_each(group.begin(), group.end(), [&](Point2f gp) {
            if (gp.x > vtp.x) {
                rgp.push_back(gp);
            } else {
                lgp.push_back(gp);
            }

            // get bound of shape
            if (gp.x < tlc.x) tlc.x = gp.x;
            if (gp.y < tlc.y) tlc.y = gp.y;
            if (gp.x > brc.x) brc.x = gp.x;
            if (gp.y > brc.y) brc.y = gp.y;
        });

        int minm = 3; // minimum points on each side of three way
        int mxnm = 10; // maximum points different on each side of three way
        
        // both side must be balance in a number of points
        int dfls = abs(lgp.size() - rgp.size());
        if (dfls < 0) dfls = -dfls;

        if (lgp.size() > minm && rgp.size() > minm && dfls < 7) {
            // find top left line (min y)
            // find top right line (min y)
            Point2f tl = lgp.at(0);
            Point2f tr = rgp.at(0);

            for (int i = 1; i < lgp.size(); i++) {
                if (lgp.at(i).y < tl.y) tl = lgp.at(i);
            }

            for (int i = 1; i < rgp.size(); i++) {
                if (rgp.at(i).y < tr.y) tr = rgp.at(i);
            }

            // check V-shape
            Vec4f ll = this->fitlines(lgp);
            Vec4f rl = this->fitlines(rgp);

            // rect that contains line
            Rect rbn = Rect(tlc, brc);
            Size s = Size(rbn.width, rbn.height);
            
            // normalize vectors

            Point2f vlp = this->fvect(tl, vtp);
            Point2f vrp = this->fvect(tr, vtp);

            // calculate angle
            double avp = this->avect(vlp, vrp);

            // means this group is V shape
            
            if (avp > 1) {
                Mat graph;

                #ifdef debug_threeway_detect
                Scalar clr = this->clrred;  
                graph = Mat::zeros(fsize, CV_8UC3);

                // draw lines
                line(graph, tl, vtp, clr, 1, 8, 0);              
                line(graph, tr, vtp, clr, 1, 8, 0);

                // draw rect bounding group
                this->drawpoints(points, graph, this->clrwht);

                this->drawpoints(lgp, graph, this->clrblu);
                this->drawpoints(rgp, graph, this->clrylw);

                Point center(fsize.width / 2, fsize.height);
                this->drawpoint(center, graph, this->clrred);
                
                rectangle(graph, rbn, this->clrred);

                show("debug three way crossroads points", graph);
                #endif

                // convert draw only path on screen
                graph = Mat::zeros(fsize, CV_8UC1);
                line(graph, tl, vtp, this->clrwht, 2, 8, 0);              
                line(graph, tr, vtp, this->clrwht, 2, 8, 0);

                vector<Mat> rggp = this->segment(graph, this->cpMaxlv);
                vector<Point2f> rgpp = this->genpoints(rggp);

                twcrps.push_back(rgpp);
                
            } else { // if the angle is not enough to make vshape
                sdpths.push_back(group);
            }
        } else { // if not enough points to make a V shape
            sdpths.push_back(group);
        }
        k++;
    });

    vector<Point2f> onlysides;
    for_each(sdpths.begin(), sdpths.end(), [&](vector<Point2f> group) {
        for_each(group.begin(), group.end(), [&](Point2f point) {
            onlysides.push_back(point);
        });
    });

    // create graph that has no 3 way line
    Mat graph3 = Mat::zeros(fsize, CV_8UC1);
    this->drawpoints(onlysides, graph3, this->clrwht);
    vector<Mat> parts2 = this->segment(graph3, this->cpMaxlv);

    vector<Point2f> leftside;
    vector<Point2f> rightside;
    this->findside(leftside, rightside, parts2);

    Mat graph5 = Mat::zeros(fsize, CV_8UC1);
    this->drawpoints(points, graph5, this->clrwht);
    show("before find side", graph5);

    return clusters;

}

void PathChaser::findside(vector<Point2f> &left, vector<Point2f> &right, vector<Mat> parts) {
    double w = parts.at(0).cols;
    double h = parts.at(0).rows;
    Size fsize = Size(w, h * parts.size());

    // seeding center point
    double ctp = (w / 2);

    // min and max distance between center point and path point
    double mnd = 60;
    double mxd = 90;

    // distance between center point and path point
    double d = mnd;

    // min different between two center points
    double mcd = 35;

    // previous path points
    double pl = -1, pr = -1;

    // reset center points container
    vector<Point2f> ctpoints;

    for (int i = parts.size() - 1; i >= 0; i--) {
    // for (int i = 0; i < parts.size(); i++) {
        
        Mat part = parts.at(i);
        
        // y position of center points
        double y = h * i;

        // new center point
        Point2f ncp(-1, y);

        // get points from segment
        vector<Point2f> comets = this->converge(part);

        // find closest point to center point on both side
        double cll = this->closerLeft(comets, ctp);
        double clr = this->closerRight(comets, ctp);

        // validate points: new path points must not be too different from previous path points
        double dmax = mnd;
        if (clr != -1 && pr != -1) {
            double dcr = abs(clr - pr);
            // different is too large, then it's not right
            if (dcr > dmax) clr = -1;
        }

        if (cll != -1 && pl != -1) {
            double dcl = abs(cll - pl);
            // different is too large, then it's not left
            if (dcl > dmax) cll = -1;
        }

        /* create new center point */

        // both points available, center point will be in middle
        if (cll != -1 && clr != -1) {
            double nd = ((clr - cll) / 2 + d) / 2;
            ncp.x = nd + cll; 

            // update min distance from center point to path points
            nd = abs(ncp.x - clr);
            if (nd > mnd && nd < mxd) d = (d + nd) / 2;

            // update previous path points
            pl = cll; pr = clr;
        } else {
            if (cll != -1) {
                ncp.x = cll + d;
                pl = cll;
            }
            if (clr != -1) {
                ncp.x = clr - d;
                pr = clr;
            }
        }

        // only push center point in if it's identified (x not -1)
        if (ncp.x != -1) {
            ctpoints.push_back(ncp);
            // center point x is updated
            ctp = ncp.x;
            
            if (cll != -1) left.push_back(Point2f(cll, y));
            if (clr != -1) right.push_back(Point2f(clr, y));
        }
    }

    Mat graph = Mat::zeros(fsize, CV_8UC3);
    this->drawpoints(ctpoints, graph, this->clrylw);
    this->drawpoints(left, graph, this->clrwht);
    this->drawpoints(right, graph, this->clrwht);
    show("multiple points", graph);
}

double PathChaser::closerLeft(vector<Point2f> points, double x) {
    if (points.size() == 0) return -1;
    double r = -1;
    for_each(points.begin(), points.end(), [&](Point2f p) {
        if (p.x < x) {
            if (r == -1) {
                r = p.x;
            } else {
                double px = abs(p.x - x);
                double rx = abs(r - x);
                if (px < rx) r = p.x;
            }
            
        }
        
    });
    return r;
}

double PathChaser::closerRight(vector<Point2f> points, double x) {
    if (points.size() == 0) return -1;
    double r = -1;
    for_each(points.begin(), points.end(), [&](Point2f p) {
        if (p.x > x) {
            if (r == -1) {
                r = p.x;
            } else {
                double px = abs(p.x - x);
                double rx = abs(r - x);
                if (px < rx) r = p.x;
            }
            
        }
        
    });
    return r;
}

Point2f PathChaser::leftmostPoint(vector<Point2f> points, double startX) {
    if (points.size() == 0) return Point2f(-1, -1);
    Point2f r = points.at(0);
    for_each(points.begin(), points.end(), [&](Point2f p) {
        if (p.x < r.x && p.x > startX) r = p;
    });
    return r;
}
Point2f PathChaser::rightmostPoint(vector<Point2f> points, double endX) {
    if (points.size() == 0) return Point2f(-1, -1);
    Point2f r = points.at(0);
    for_each(points.begin(), points.end(), [&](Point2f p) {
        if (p.x > r.x && p.x < endX) r = p;
    });
    return r;
}

void PathChaser::video(string v, int wk) {
    VideoCapture cap(v);

    cap.set(CV_CAP_PROP_POS_FRAMES, this->frs);

    Mat frame;
    this->frc = this->frs;
    bool start = false;
    bool pause = false;

    this->frc --;
    while (1) {
        cap >> frame;
    
        this->frc++;
        if (this->frc >= cap.get(cv::CAP_PROP_FRAME_COUNT)) {
            this->frc = 1;
            cap.set(CV_CAP_PROP_POS_FRAMES, 1);
        }

        if (frame.empty()) continue;
        
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

       show("s0", frame);
    //    Mat s1 = this->isolate(frame);
    //    show("s1", s1);

    //    Mat s2 = this->chunk(s1);
    //    show("s2", s2);

        Mat s4 = this->roadline(frame);
        // show("s4", s4);
        



        ///// END OF PROCESSING /////
        if (!start || pause) waitKey();
        start = true;

        int k = cv::waitKey(wk) &0xffff ;
        
        if(k == 27) break;
        if(k == 32) {
            pause = false;
            waitKey();
        }
        if (k == 110) { // pause + next frame
            pause = true;
        }
        
    }

}

//////////////////////////////////////////
/* BELOW IS THE CODE HAS BEEN COMPLETED */
//////////////////////////////////////////

void PathChaser::drawsegments(Mat frame) {
    Size s = frame.size();

    int a = this->cpMaxlv;
    int b = this->cpLimlv;
    int c = this->cpLimFlH;
    
    double sh = s.height / (double) a;

    for (int i = 0; i < a; i++) {
        Point m(0, sh * i);
        Point n(s.width, sh * i);
        line(frame, m, n, Scalar(158, 158, 158), 1, 8, 0);
        if (a - i - 1 == b) 
            line(frame, m, n, Scalar(107, 0, 142), 2, 8, 0);
        if (a - i - 1 == c) 
            line(frame, m, n, Scalar(196, 113, 183), 2, 8, 0);
    }
}

void PathChaser::drawcloudpoints(vector<vector<Point2f>> points, Mat frame) {
    for (int i = 0; i < points.size(); i++) {
        vector<Point2f> pl = points.at(i);
        Point a = pl.at(0);
        Point b = pl.at(1);
        
        // upper end point
        circle(frame, a, (int)3, this->clrylw, 2, 8, 0);

        // lower end point
        circle(frame, b, (int)3, this->clrblu, 2, 8, 0);
    }
}

void PathChaser::drawfc(Mat frame) {
    String val = "Frame: " + to_string(this->frc);
    Point pos = Point(10, frame.rows - 10);
    cv::putText(frame, val, pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, this->clrwht, 1, 8);
}

Mat PathChaser::draw(Mat frame, Rect box, String label) {
    Mat image = frame.clone();
    Point a(box.x, box.y);
    Point b(box.x + box.width, box.y + box.height);
    Scalar bclr(255,255,255);

    rectangle(image, a, b, bclr, 2);

    Scalar clrf = Scalar(255,0,0);
    putText(image, label, a, FONT_HERSHEY_SIMPLEX, 1, clrf, 3, 0, false);
    return image;
}

Rect PathChaser::mkcbox(Mat frame) {
    double W = frame.cols;
    double H = frame.rows;
    double x = this->clrbox[0];
    x = W / 2 - this->clrbox[2] / 2;
    double y = this->clrbox[1];
    double w =this->clrbox[2];
    double h =this->clrbox[3];
    double t = (H - (h + y)) / H; // T
    double r = (W - (w + x)) / W; // R
    double b = y / H; // B
    double l = x / W; // L
    double *bxb = new double[4 * sizeof(double)]
        {t, r, b, l}; // T - R - B - L
    for (int i = 0; i <4 ;i++) {
        bxb[i] = (double) ((int) (bxb[i] * 100)) / 100;
    }

    Rect cbx = this->mkbox(frame, bxb);
    delete bxb;
    return cbx;
}

Mat PathChaser::isolate(Mat frame) {
    Rect box = this->mkbox(frame, this->isobox);
    return frame(box);
}

Rect PathChaser::mkbox(Mat frame, double *frontier) {
    double w = frame.cols;
    double h = frame.rows;
    double x = w * frontier[3];
    double y = h * frontier[0];
    double w2 = w - w * (frontier[1] + frontier[3]);
    double h2 = h - h * (frontier[0] + frontier[2]);
    return Rect(x, y, w2, h2);
}

Mat PathChaser::bird(Mat source) {

    Mat destination;

    double focalLength, dist, alpha, beta, gamma;

    alpha =((double) bdAlpha -90) * CV_PI/180;
    beta =((double) bdBeta -90) * CV_PI/180;
    gamma =((double) bdGam -90) * CV_PI/180;
    focalLength = (double) bdF;
    dist = (double) bdDist;

    Size image_size = source.size();
    double w = (double)image_size.width, h = (double)image_size.height;


    // Projecion matrix 2D -> 3D
    Mat A1 = (Mat_<float>(4, 3)<<
        1, 0, -w/2,
        0, 1, -h/2,
        0, 0, 0,
        0, 0, 1 );

    // Rotation matrices Rx, Ry, Rz
    Mat RX = (Mat_<float>(4, 4) <<
        1, 0, 0, 0,
        0, cos(alpha), -sin(alpha), 0,
        0, sin(alpha), cos(alpha), 0,
        0, 0, 0, 1 );

    Mat RY = (Mat_<float>(4, 4) <<
        cos(beta), 0, -sin(beta), 0,
        0, 1, 0, 0,
        sin(beta), 0, cos(beta), 0,
        0, 0, 0, 1	);

    Mat RZ = (Mat_<float>(4, 4) <<
        cos(gamma), -sin(gamma), 0, 0,
        sin(gamma), cos(gamma), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1	);


    // R - rotation matrix
    Mat R = RX * RY * RZ;

    // T - translation matrix
    Mat T = (Mat_<float>(4, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, dist,
        0, 0, 0, 1);

    // K - intrinsic matrix
    Mat K = (Mat_<float>(3, 4) <<
        focalLength, 0, w/2, 0,
        0, focalLength, h/2, 0,
        0, 0, 1, 0
        );

    Mat transformationMat = K * (T * (R * A1));

    warpPerspective(source, destination, transformationMat, image_size, INTER_CUBIC | WARP_INVERSE_MAP);

    #ifdef debug_birdview
    show(this->dbwn, destination);
    #endif

    return destination;
}


Mat PathChaser::chunk(Mat frame) {
    Mat image = frame.clone();
    Mat eqal;


    Rect cbx = this->mkcbox(image);
    Mat reg = image(cbx);

    #ifdef debug
    frame = this->draw(image, cbx, "box");
    show("color region", reg);
    #endif

    Mat mask;

    this->minmax(reg);
    this->updsclr();


    inRange(image, this->lower, this->upper, mask);
    this->refineMask(mask);
    #ifdef debug
    show("mask", mask);
    #endif

    return frame;
}

void PathChaser::refineMask(Mat &frame) {
    Mat m(this->mc, this->mc, CV_8U, Scalar(1));
    cv::morphologyEx(frame, frame, cv::MORPH_CLOSE, m);
}

void PathChaser::minmax(Mat frame) {

    if (this->frc >= this->fcp) return;

    Mat hsv; cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    uint8_t* pixptr = (uint8_t*) hsv.data;

    int cn = hsv.channels();
    int cl = hsv.cols;
    int rw = hsv.rows;

    for(int i = 0; i < rw; i++) {
        for(int j = 0; j < cl; j++) {
            for (int k = 0; k < cn; k++) {
                int v = pixptr[i * cl * cn + j * cn + k];
                if (v < this->min[k]) this->min[k] = v;
                if (v > this->max[k]) this->max[k] = v;
            }
        }
    }

    #ifdef debug
        cout << "\nframe capture: " << this->frc << endl;
        cout << "lower " << this->print(this->min, 3) << endl;
        cout << "upper " << this->print(this->max, 3) << endl;
    #endif
}

void PathChaser::updsclr() {
    this->lower = Scalar(this->min[0], this->min[1], this->min[2]);
    this->upper = Scalar(this->max[0], this->max[1], this->max[2]);
}

string PathChaser::print(int *arr, int n) {
    string s = "";
    for (int i = 0; i < n; i++) {
        s += arr[i] + " ";
    }
    return s;
}


Mat PathChaser::preprocess(Mat frame) {
    Mat image = frame.clone();
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    if (this->ppBlurW == 0) this->ppBlurW = 1;
    if (this->ppBlurH == 0) this->ppBlurH = 1;
    cv::blur(image, image, Size(this->ppBlurW, this->ppBlurH));
    cv::threshold(image, image, this->ppLowThr, this->ppHighThr, cv::THRESH_BINARY);

    #ifdef debug_preprocess
    show(this->dbwn2, image);
    #endif

    return image;
}

Mat PathChaser::noiseremove(Mat frame) {
    Mat image = frame.clone();

    Size erosize = Size(this->ppEroW + 1, this->ppEroH + 1);
    Point eropnt = Point(this->ppEroW, this->ppEroH);
    Mat eroele = getStructuringElement(this->ppEroType, erosize, eropnt);
    cv::erode(image, image, eroele, Point(0, 0), this->ppEroIt);

    Size dltsize = Size(this->ppDltW + 1, this->ppDltH + 1);
    Point dltpnt = Point(this->ppDltW, this->ppDltH);
    Mat dltele = getStructuringElement(this->ppDltType, dltsize, dltpnt);
    cv::dilate(image, image, dltele, Point(0, 0), this->ppDltIt);

    // this->ctrclean(image);

    #ifdef debug_noiseremove
    show(this->dbwn3, image);
    #endif
    return image;
}

vector<vector<Point2f>> PathChaser::dbscan(vector<Point2f> points, int epsilon) {
    vector<vector<Point2f>> clusters;
    if (points.size() == 0) return clusters;

    vector<Point2f> cluster;
    vector<Point2f> remains;
    Point2f S = points.at(0);
    stack<Point2f> neighbors;
    while (true) {
        
        // find all S neighbors
        for (int i = 0; i < points.size(); i++) {
            Point2f T = points.at(i);
            double distance = this->calcDistanceP(S, T);
            if (distance < epsilon) {
                cluster.push_back(T);
                neighbors.push(T);
            } else {
                remains.push_back(T);
            }
        }

        if (!neighbors.empty()) {

            // if there is a neighbor found, S = new neightbor
            S = neighbors.top();
            neighbors.pop();

            // find neighbor that is not in cluster
            points = remains;
        } else {
            // if there is no neighbor left to find, new cluster is added
            clusters.push_back(cluster);
            cluster = vector<Point2f>();

            // start find neighbor from start
            S = points.at(0);
        }

        if (points.size() == 0) {
            while (!neighbors.empty()) {
                cluster.push_back(neighbors.top());
                neighbors.pop();
            }
            clusters.push_back(cluster);
            break;
        }

        remains = vector<Point2f>();
    }
    return clusters;
}

vector<Point2f> PathChaser::genpoints(vector<Mat> parts) {
    vector<Point2f> points;
    int segh = parts.at(0).rows;

    int k = 0;
    this->cpMinDt = 0;
    for (int i = parts.size() - 1; i >= 0; i--) {
        if (k++ > this->cpLimlv) break;
        Mat part = parts.at(i);
        vector<Point2f> centroits = this->converge(part);

        for (int j = 0; j < centroits.size(); j++) {
            Point2f p = centroits.at(j);
            p.y = p.y + segh * i;
            points.push_back(p);
        }
    }

    return points;
}

vector<Point2f> PathChaser::getlinepoint(Vec4f line, Size s, int hb, int lb) {
    /*
    * hb: high bound - number of pixel of upper line bound
    * lb: low bound - number of pixel of lower line bound
    */

    vector<Point2f> coords;
    Point a, b;
    double x0, y0, vx, vy;

    a.y = s.height + lb;
    b.y = s.height - hb;

    // vtpt x,y = line0, line1
    vx = line[1]; 
    vy = -line[0];
    
    x0 = line[2]; 
    y0 = line[3];

    // c1: x = x0 + t * vx, t = (y - y0) / vy, vx,vy = l0,l1
    // c2: x = (vy / vx) * (y0 - y) + x0, vx, vy = l1,-l0
   
    a.x = (vy / vx) * (y0 - a.y) + x0;
    b.x = (vy / vx) * (y0 - b.y) + x0;

    coords.push_back(a);
    coords.push_back(b);

    return coords;
}

Vec4f PathChaser::fitlines(vector<Point2f> points) {
    Vec4f line; cv::fitLine(points, line, 1, 10, 10, 10);
    return line;
}

void PathChaser::drawline(Vec4f line, Mat frame, Scalar color) {
    
    vector<Point2f> coords = this->getlinepoint(line, frame.size(), this->cpFlmHB, this->cpFlmLB);
    Point a = coords.at(0);
    Point b = coords.at(1);

    cv::line(frame, a, b, color);
}

void PathChaser::drawpoints(vector<Point2f> points, Mat frame, Scalar color) {
    for (int j = 0; j < points.size(); j++) {
        Point p = points.at(j);
        circle(frame, p, (int)3, color, 2, 8, 0);
    }
}

void PathChaser::drawpoint(Point2f point, Mat frame, Scalar color) {
    circle(frame, point, (int)3, color, 2, 8, 0);
}

Scalar PathChaser::randclr() {
    int x = this->rng.uniform(0,255);
    int y = this->rng.uniform(0,255);
    int z = this->rng.uniform(0,255);
    return Scalar(x, y, z);;
}

void PathChaser::drawclusters(vector<vector<Point2f>> clusters, Mat frame) {
    for (int i = 0; i < clusters.size(); i++) {
        vector<Point2f> cluster = clusters.at(i);
        this->drawpoints(cluster, frame, this->randclr());
    }
}

vector<Vec4f> PathChaser::fitlinesMult(vector<vector<Point2f>> clusters) {
    vector<Vec4f> lines;
    for (int i = 0; i < clusters.size(); i++) {
        vector<Point2f> cluster = clusters.at(i);
        Vec4f line = this->fitlines(cluster);
        lines.push_back(line);
    }
    return lines;
}

void PathChaser::drawlines(vector<Vec4f> lines, Mat frame) {
    for (int i = 0; i < lines.size(); i++) {
        this->drawline(lines.at(i), frame, this->clrgrn);
    }
}

vector<Mat> PathChaser::segment(Mat frame, int n) {
    if (n == 0) n = 1;
    vector<Mat> parts;
    int width = frame.cols;
    int height = frame.rows;
    int h = height / n;

    for (int i = 0; i < n; i++) {
        Rect roi;
        roi.x = 0;
        roi.y = h * i;
        roi.width = width;
        roi.height = h;
        Mat part = frame(roi);
        parts.push_back(part);
    }
    return parts;
}

void PathChaser::ctrclean(Mat frame) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    vector< vector<Point> > contours_poly(contours.size());
    vector<Rect> rects(contours.size());

    for( int i = 0; i < abs(contours.size()); i++ ) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true );
        rects[i] = boundingRect(Mat(contours_poly[i]));
        
        RotatedRect box = cv::minAreaRect(contours[i]);

        Point2f pt[4];
        box.points(pt);


        Point2f vt = this->fvect(pt[0], pt[3]);
        Point2f vt2 = this->fvect(pt[0], pt[1]);

        double w = this->lvect(vt);
        double h = this->lvect(vt2);
        double s = w * h;

        Scalar clr = this->clrwht;

        if (s > 8000) {
            Mat rpl(rects[i].height, rects[i].width, CV_8UC1, Scalar(0));
            rpl.copyTo(frame(rects[i]));

            for (int j = 0; j < 4; j++) {
                line(frame, pt[j], pt[(j + 1) % 4], clr);
            }

            rectangle(frame, rects[0], clr, 2, 8, 0);
            circle(frame, pt[0], 3, clr, 3, 8, 0);
            circle(frame, pt[1], 3, clr, 3, 8, 0);
            circle(frame, pt[3], 3, clr, 3, 8, 0);

        }

    }

}

vector<Point2f> PathChaser::converge(Mat frame) {
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    vector< vector<Point> > contours_poly(contours.size());
    vector<Point2f> center(contours.size());
    vector<float> radius(contours.size());

    for( int i = 0; i < abs(contours.size()); i++ ) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true );
        minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
    }

    return center;
}


// read file config

bool PathChaser::readconfig(string fn) {
    fstream fp;

    fp.open(fn, std::fstream::in | std::fstream::out | std::fstream::app);

    // If file does not exist, Create new file
    if (!fp)  {
        cout << "Cannot open file, file does not exist. Creating new file..";
        fp.open(fn,  fstream::in | fstream::out | fstream::trunc); fp.close();
        return false;
    }

    for(std::string line; getline(fp, line);) {
        // remove comment partials
        this->rmcmt(line);

        // skip empty line
        bool isEmpty = line.find_first_not_of (' ') == line.npos;
        if (line.size() == 0 || isEmpty) continue;

        // trim spaces
        this->trim(line);

        // split token
        vector<string> pair = this->getpair(line, " ");
        if (pair.size() < 2) continue;
        this->params[pair.at(0)] = pair.at(1);
    }
    fp.close();

    // fill params value to variable
    this->fillparam();
}

vector<string> PathChaser::getpair(string str, string dl) {
    vector<string> pair;
    size_t p  = str.find_first_of(dl);
    string left = str.substr(0, p);
    string right = "";
    if (p < str.size()) {
        right = str.substr(p, str.size() - 1);
    }

    this->trim(left);
    this->trim(right);
    
    if (!left.empty()) pair.push_back(left);
    if (!right.empty()) pair.push_back(right);
    return pair;
}

void PathChaser::rmcmt(string &line) {
    for (int i = 0; i < line.size(); i++) {
        char c = line.at(i);
        if (c == '#') {
            line = line.substr(0, i);
            break;
        }
    }
}

void PathChaser::trim(string &line) {
    if (line.size() == 0) return;
    while (isspace(line.at(0))) {
        line = line.substr(1).append(line.substr(0,1));
    }
    while (isspace(line.at(line.size() - 1))) {
        line = line.substr(0, line.size() - 1);
    }
}

float PathChaser::param(string key) {
    string val = this->params[key];
    if (val.empty()) {
        cout << "config error: \"" << key << "\" not found" << endl;
        return -1;
    }
    float r = stof(val);
    return r;
}


void PathChaser::fillparam() {
    // birdview params
    this->bdAlpha = this->param("view_alpha");
    this->bdBeta = this->param("view_beta");
    this->bdGam = this->param("view_gamma");
    this->bdF = this->param("view_f");
    this->bdDist = this->param("view_dist");

    // preprocess params
    this->ppLowThr = this->param("thresh_low");
    this->ppHighThr = this->param("thresh_high");
    this->ppBlurW = this->param("blur_w");
    this->ppBlurH = this->param("blur_h");

    // noise removal params
    this->ppEroW = this->param("erode_w");
    this->ppEroH = this->param("erode_h");
    this->ppEroIt = this->param("erode_iter");
    this->ppEroType = this->param("erode_type");

    this->ppDltW = this->param("dilate_w");
    this->ppDltH = this->param("dilate_h");
    this->ppDltIt = this->param("dilate_iter");
    this->ppDltType = this->param("dilate_type");

    // cloud points params
    this->cpMaxlv = this->param("seg_total"); // segments
    this->cpLimlv = this->param("seg_limit");
    this->cpLimFlH = this->param("seg_flim_h");
    this->cpLimFlL = this->param("seg_flim_l");
    this->cpLimDt = 2;
    this->cpMinDt = 0; // px
    this->cpDviDt = 0;

    // cluster grouping params
    this->clsAgliL = this->param("cls_agli_low");
    this->clsAgliH = this->param("cls_agli_high");
    this->clsAglFm = this->param("cls_regr_magl");

}

void PathChaser::settrackbar() {
    #ifdef debug

    #ifdef debug_birdview
    this->dbwn = "debug birdview";
    namedWindow(this->dbwn, 1);
    createTrackbar("Alpha", this->dbwn, &this->bdAlpha, 180, PathChaser::debugger, this);
    createTrackbar("Beta", this->dbwn, &this->bdBeta, 180, PathChaser::debugger, this);
    createTrackbar("Gamma", this->dbwn, &this->bdGam, 180, PathChaser::debugger, this);
    createTrackbar("f", this->dbwn, &this->bdF, 2000, PathChaser::debugger, this);
    createTrackbar("Distance", this->dbwn, &this->bdDist, 2000, PathChaser::debugger, this);
    #endif

    #ifdef debug_preprocess
    this->dbwn2 = "debug preprocessing";
    namedWindow(this->dbwn2, 1);
    createTrackbar("Thresh Lower", this->dbwn2, &this->ppLowThr, 255, PathChaser::debugger, this);
    createTrackbar("Thresh Upper", this->dbwn2, &this->ppHighThr, 255, PathChaser::debugger, this);
    createTrackbar("Blur K width", this->dbwn2, &this->ppBlurW, 25, PathChaser::debugger, this);
    createTrackbar("Blur K height", this->dbwn2, &this->ppBlurH, 25, PathChaser::debugger, this);
    #endif

    #ifdef debug_noiseremove
    this->dbwn3 = "debug noise remove";
    namedWindow(this->dbwn3, 1);
    createTrackbar("Erode Width", this->dbwn3, &this->ppEroW, 10, PathChaser::debugger, this);
    createTrackbar("Erode Height", this->dbwn3, &this->ppEroH, 10, PathChaser::debugger, this);
    createTrackbar("Erode Iter", this->dbwn3, &this->ppEroIt, 10, PathChaser::debugger, this);
    createTrackbar("Erode Type", this->dbwn3, &this->ppEroType, 2, PathChaser::debugger, this);

    createTrackbar("Dilate Width", this->dbwn3, &this->ppDltW, 10, PathChaser::debugger, this);
    createTrackbar("Dilate Height", this->dbwn3, &this->ppDltH, 10, PathChaser::debugger, this);
    createTrackbar("Dilate Iter", this->dbwn3, &this->ppDltIt, 10, PathChaser::debugger, this);
    createTrackbar("Dilate Type", this->dbwn3, &this->ppDltType, 2, PathChaser::debugger, this);
    #endif

    #ifdef debug_graph1
    this->dbwn4 = "debug segment and clustering";
    namedWindow(this->dbwn4, 1);

    #ifdef debug_segment
    this->mgAgl = round(this->clsAglFm * 100);
    createTrackbar("Total segments", this->dbwn4, &this->cpMaxlv, 50, PathChaser::debugger, this);
    createTrackbar("Used segments", this->dbwn4, &this->cpLimlv, 50, PathChaser::debugger, this);
    createTrackbar("High line segments", this->dbwn4, &this->cpLimFlH, 50, PathChaser::debugger, this);
    createTrackbar("Low line segments", this->dbwn4, &this->cpLimFlL, 50, PathChaser::debugger, this);
    createTrackbar("Min group angle", this->dbwn4, &this->mgAgl, 1000, PathChaser::debugger, this);
    #endif

    #endif // debug graph 1

    #ifdef debug_graph2
    
    #ifdef debug_cluster_grouping
    
    #endif
    
    this->dbwn5 = "debug grouping clustering";
    namedWindow(this->dbwn5, 1);
    
    
    #endif

    this->clrred = Scalar(0, 0, 255);
    this->clrgrn = Scalar(0, 255, 0);
    this->clrylw = Scalar(51, 255, 255);
    this->clrwht = Scalar(255, 255, 255);
    this->clrblu = Scalar(255, 0, 0);
    this->rng = RNG(12345);
    #endif
}

double PathChaser::calcDistanceP(Point2f a, Point2f b) {
    double m = pow(a.x - b.x, 2);
    double n = pow(a.y - b.y, 2);
    return sqrt(m + n);
}

void PathChaser::getMeanDistance(vector<Point2f> points) {
    double result = 0;
    if (this->cpLimDt == -1) this->cpLimDt = points.size();
    for (int i = 0; i < points.size(); i++) {
        if (i > this->cpLimDt || i + 1 >= points.size()) break;
        Point2f a = points.at(i);
        Point2f b = points.at(i + 1);
        result += this->calcDistanceP(a, b);
    }
    this->cpMinDt = result / this->cpLimDt + this->cpDviDt;
}


vector<vector<Point2f>> PathChaser::groupclusters(
        vector<vector<Point2f>> clusters,
        vector<vector<int>> couples,
        vector<double> vtangles,
        double minAngle)
{   
    vector<vector<Point2f>*> newclusters;
    vector<vector<int>*> clstId; // dim 1: groups, dim 2: id of clusters
    
    for (int i = 0; i < vtangles.size(); i++) {
        if (vtangles.at(i) <= minAngle) continue;
        // means two belong to a group
        vector<int> couple = couples.at(i);
        int i1 = couple.at(0);
        int i2 = couple.at(1);

        // check the cluster is already in group
        int gid = -1; // common group id
        bool i1g, i2g;
        i1g = false;
        i2g = false;
        /* group cases:
        * - neither cluster in any group => merge c1 and c2 = new group
        * - cluster i1 in a group => add c2 to group
        * - cluster i2 in a group => add c1 to group
        * - both cluster in a group => do nothing
        */
        
        for (int j = 0; j < clstId.size(); j++) {
            vector<int> *grpId = clstId.at(j);
            for (int k = 0; k < grpId->size(); k++) {
                int id = grpId->at(k);
                if (id == i1) i1g = true;
                if (id == i2) i2g = true;
                if (i1g || i2g) {
                    gid = j;
                    goto checkend;
                }
            }
        }
        checkend:

        if (gid == -1) {
            // create new group id
            vector<int> *grpId = new vector<int>();
            grpId->push_back(i1);
            grpId->push_back(i2);
            clstId.push_back(grpId);

            // clusters grouping
            vector<Point2f> *AB = new vector<Point2f>();
            vector<Point2f> A = clusters.at(i1);
            vector<Point2f> B = clusters.at(i2);
            AB->reserve(A.size() + B.size());
            AB->insert(AB->end(), A.begin(), A.end());
            AB->insert(AB->end(), B.begin(), B.end());

            newclusters.push_back(AB);
        } else {
            // make sure not both clusters already in same group
            if (!i1g || !i2g) {

                // add new group id
                vector<int> *grpId = clstId.at(gid);
                grpId->push_back(i1g ? i2 : i1);

                // clusters adding
                vector<Point2f> *AB = newclusters.at(gid);

                // if cluster1 already in group -> add cluster2 to group and vice versa
                vector<Point2f> A = clusters.at(i1g ? i2 : i1);
                AB->insert(AB->end(), A.begin(), A.end());
            }
        }
    }
    // convert from cluster pointer to normal cluster
    vector<vector<Point2f>> nclusters;
    for_each(newclusters.begin(), newclusters.end(), [&](vector<Point2f>* cluster){
        vector<Point2f> ncluster;
        for_each(cluster->begin(), cluster->end(), [&](Point2f point){
            ncluster.push_back(point);
        });
        nclusters.push_back(ncluster);
    });

    for (int i = 0; i < clusters.size(); i++) {
        vector<Point2f> cluster = clusters.at(i);
        bool exist = false;
        for_each(clstId.begin(), clstId.end(), [&](vector<int>* id){
            for_each(id->begin(),id->end(), [&](int x){
                if (x == i) exist = true;
            });
        });
        if (exist) continue;
        nclusters.push_back(cluster);
    }
    
    // free group id memory
    for_each(clstId.begin(), clstId.end(), [](vector<int>* id){
        delete id;
    });
    for_each(newclusters.begin(), newclusters.end(), [](vector<Point2f>* cluster){
        delete cluster;
    });

    // cout << newclusters.size() << " " << nclusters.size() << endl;

    return nclusters;
}

vector<double> PathChaser::findVectorsAngles(
        vector<vector<Point2f>> pointsline, 
        vector<Point2f> its,
        vector<vector<int>> couples)
{   
    vector<double> angles;
    for (int i = 0; i < couples.size(); i++) {
        vector<int> couple = couples.at(i);
        int i1 = couple.at(0);
        int i2 = couple.at(1);

        Point2f gp1 = pointsline.at(i1).at(0);
        Point2f gp2 = pointsline.at(i2).at(1);
        Point2f x = its.at(i);

        Point2f v1 = this->fvect(gp1, x);
        Point2f v2 = this->fvect(gp2, x);
        double agl = this->avect(v1, v2);
        angles.push_back(agl);
    }
    return angles;
}

double PathChaser::avect(Point2f v1, Point2f v2) {
    double dot = this->pvect(v1, v2);
    double dlw = abs(this->lvect(v1)) * abs(this->lvect(v2));
    return acos(dot / dlw);
}

double PathChaser::pvect(Point2f v1, Point2f v2) {
    return v1.x * v2.x + v1.y * v2.y;
}

double PathChaser::lvect(Point2f v) {
    return sqrt(pow(v.x, 2) + pow(v.y, 2));
}

Point2f PathChaser::fvect(Point2f a, Point2f b) {
    return Point2f(b.x - a.x, b.y - a.y);
}

void PathChaser::drawftlIntersect(
        vector<vector<Point2f>> pointsline, 
        vector<Point2f> its,
        vector<vector<int>> couples,
        Mat frame)
{
    for (int i = 0; i < couples.size(); i++) {
        vector<int> couple = couples.at(i);
        int i1 = couple.at(0);
        int i2 = couple.at(1);

        vector<Point2f> gp1 = pointsline.at(i1);
        vector<Point2f> gp2 = pointsline.at(i2);
        Point2f x = its.at(i);

        cv::line(frame, gp1.at(1), x, this->clrred, 2, 8, 0);
        cv::line(frame, gp2.at(1), x, this->clrred, 2, 8, 0);
    }
}

bool PathChaser::intersectLineSegment (
    vector<Point2f> p1, vector<Point2f> p2, Point2f &r)
{   
    float p0_x, p0_y, p1_x, p1_y;
    float p2_x, p2_y, p3_x, p3_y;
    float i_x, i_y;

    p0_x = p1.at(0).x;
    p0_y = p1.at(0).y;

    p1_x = p1.at(1).x;
    p1_y = p1.at(1).y;

    p2_x = p2.at(0).x;
    p2_y = p2.at(0).y;

    p3_x = p2.at(1).x;
    p3_y = p2.at(1).y;


    float s1_x, s1_y, s2_x, s2_y;
    s1_x = p1_x - p0_x;
    s1_y = p1_y - p0_y;
    
    s2_x = p3_x - p2_x;
    s2_y = p3_y - p2_y;

    float s, t;
    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y);
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y);

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
    {
        // Collision detected
        r.x = p0_x + (t * s1_x);
        r.y = p0_y + (t * s1_y);
        return true;
    }

    return false;
}

bool PathChaser::intersectLine(
    Point2f o1, Point2f p1, 
    Point2f o2, Point2f p2, 
    Point2f &r)
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

vector<vector<Point2f>> PathChaser::findIntersection (
    vector<vector<Point2f>> grp, 
    vector<vector<Point2f>> pvt,
    vector<Point2f> *its,
    vector<vector<int>> &couples)
{
    vector<vector<Point2f>> newclusters;

    for (int i = 0; i < pvt.size(); i++) {
        for (int j = 0; j < pvt.size() - 1; j++) {
            // skip same line
            if (i == j) continue;

            // check and find intersection point
            vector<Point2f> ga = pvt.at(i);
            vector<Point2f> gb = pvt.at(j);

            // filter group base on rules
            /*
            * base on angle ratio of two intersected line
            * 
            */
            Point2f v1 = this->fvect(ga.at(0), ga.at(1));
            Point2f v2 = this->fvect(gb.at(0), gb.at(1));
            Point2f v3(0,1); // Ox vector
            double a1 = this->avect(v1, v3);
            double a2 = this->avect(v2, v3);
            double kp = a1 / a2;
            // cout << kp << endl;
            if (kp <= this->clsAgliL || kp >= this->clsAgliH) {
                continue;
            }


            Point2f r;
            // bool hasJunction = this->intersectLine(ga.at(0), ga.at(1), gb.at(0), gb.at(1), r);
            bool hasJunction = this->intersectLineSegment(ga, gb, r);
            if (hasJunction) {
                its->push_back(r);

                vector<int> couple;
                couple.push_back(i);
                couple.push_back(j);

                couples.push_back(couple);
            }
        }
    }

    return newclusters;
}

vector<vector<Point2f>> PathChaser::linetopoint(vector<Vec4f> lines, Size s, int hb, int lb) {
    // from each line, find point start and points end
    vector<vector<Point2f>> pointsline;
    for (int i = 0; i < lines.size(); i++) {
        Vec4f line = lines.at(i);
        vector<Point2f> points = this->getlinepoint(line, s, hb, lb);
        pointsline.push_back(points);
    }
    return pointsline;
}

void PathChaser::show(string title, Mat frame) {
    Mat copy = frame.clone();
    PathChaser::showframe(title, copy, this);
}

void PathChaser::showframe(string title, Mat frame, void * x) {
    PathChaser *d = (PathChaser *) x;
    d->drawfc(frame);
    imshow(title, frame);
}

vector<vector<Point2f>> PathChaser::filterclusters(vector<vector<Point2f>> clusters, int minpoint) {
    vector<vector<Point2f>> newclusters;
    for (int i = 0; i < clusters.size(); i++) {
        vector<Point2f> cluster = clusters.at(i);
        int n = cluster.size();
        if (n >= minpoint) newclusters.push_back(cluster);
    }
    return newclusters;
}

void PathChaser::debugger(int, void* x) {
    // debugger with trackbar
    PathChaser *d = (PathChaser *) x;

    d->clsAglFm = (double) d->mgAgl / 100;
    cout << d->clsAglFm << endl;

    d->roadline(d->raw_mat);
}