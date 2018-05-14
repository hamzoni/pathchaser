#include "pathchaser.h"
// #define debug


#define dev

PathChaser::PathChaser()
{
    this->clingleft = true;

    this->isobox = new double[4 * sizeof(double)](); // T - R - B - L
    this->clrbox = new double[4 * sizeof(double)](); // X, Y, W, H

    #ifndef dev
    this->readconfig("pro.conf");

    #else
    this->readconfig("dev.conf");

    // this->min = new int[3 * sizeof(int)] {5, 78, 56};
    // this->max = new int[3 * sizeof(int)] {34, 255, 204};

    this->min = new int[3 * sizeof(int)] {255, 255, 255};
    this->max = new int[3 * sizeof(int)] {0, 0, 0};
    
    this->fcp = 0;
    this->frs = 1;

    this->pcp = Point2f(-1, -1); // previous center point
    this->plp = Point2f(-1, -1); // previous left point
    this->prp = Point2f(-1, -1); // previous right point


    #endif

    this->frc = 0;
    this->mc = 10;

    // default frame size
    this->fsize = Size(640, 480);

    this->settrackbar();
}

void PathChaser::laplacian(Mat src, Mat &sharp, Mat &lapla) {

    // #define show_sharp
    // #define show_lapla 

    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); 
    sharp = src;

    filter2D(sharp, lapla, CV_32F, kernel);
    
    src.convertTo(sharp, CV_32F);
    
    sharp = sharp - lapla;

    sharp.convertTo(sharp, CV_8UC3);
    lapla.convertTo(lapla, CV_8UC3);

    #ifdef show_sharp
    imshow("image sharped", sharp);
    #endif

    #ifdef show_lapla
    imshow("laplacian", lapla);
    #endif
}


Mat PathChaser::roadshape(Mat frame) {
    Mat morph = frame.clone();
    for (int r = 1; r < 4; r++)
    {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*r+1, 2*r+1));
        morphologyEx(morph, morph, CV_MOP_CLOSE, kernel);
        morphologyEx(morph, morph, CV_MOP_OPEN, kernel);
    }
    /* take morphological gradient */
    Mat mgrad;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(morph, mgrad, CV_MOP_GRADIENT, kernel);

    Mat ch[3], merged;
    /* split the gradient image into channels */
    split(mgrad, ch);
    /* apply Otsu threshold to each channel */
    threshold(ch[0], ch[0], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    threshold(ch[1], ch[1], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    threshold(ch[2], ch[2], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    /* merge the channels */
    merge(ch, 3, merged);

    Mat lapla;
    this->laplacian(merged, merged, lapla);

    return merged;
}

double PathChaser::roadlineOTSU(Mat frame) {

    double angle = 0;

    Mat image = frame.clone();

    this->raw_mat = image.clone();

    // set basic variables
    if (this->cpLimlv <= 0 || this->cpLimlv > this->cpMaxlv) {
        this->cpLimlv = this->cpMaxlv;
    }

    this->fsize = Size(image.cols, image.rows);

    // set segment height
    double seg_h = ((double) image.rows / this->cpMaxlv);
    this->cpSegmH = seg_h * (this->cpLimlv + 1);
    this->cpFlmHB = seg_h * (this->cpLimFlH + 1);
    this->cpFlmLB = seg_h * (this->cpLimFlL + 1);

    // new code
    frame = this->roadshape(frame);

    Mat rgb = frame.clone();

    Mat gray;
    cvtColor(rgb, gray, COLOR_BGR2GRAY);
    
    Mat thresh;
    threshold(gray, thresh, 125, 255, THRESH_BINARY);

    Mat bird = this->bird(thresh);

    bird = this->masknoisermv(bird);

    #ifdef debug
    show("rgb", rgb);
    show("thresh", thresh);
    show("bird", bird);
    #endif

    bird.convertTo(bird, CV_8UC3);

    // bird = this->noiseremove(bird);
    
    vector<Mat> parts = this->segment(bird, this->cpMaxlv);
    vector<vector<Point2f>> gclusters = this->gencluster(parts);

    angle = this->calcAngle(gclusters[0], gclusters[1]);

    angle = this->rtd(angle, 90) * -1;

    return angle;
}

double PathChaser::roadline(Mat frame) {
    
    double angle = this->roadlineOTSU(frame);

    return angle;

    Mat image = frame.clone();
    this->raw_mat = image.clone();

    // set basic variables
    if (this->cpLimlv <= 0 || this->cpLimlv > this->cpMaxlv) {
        this->cpLimlv = this->cpMaxlv;
    }

    this->fsize = Size(image.cols, image.rows);
    // set segment height
    double seg_h = ((double) image.rows / this->cpMaxlv);
    this->cpSegmH = seg_h * (this->cpLimlv + 1);
    this->cpFlmHB = seg_h * (this->cpLimFlH + 1);
    this->cpFlmLB = seg_h * (this->cpLimFlL + 1);

    Mat w1,w2,w3,w4,w5,w6;


    w1 = this->chunk(image);

    w2 = this->bird(w1);

    w3 = this->noiseremove(w2);

    #ifdef show_cloak

    show("Image root", image);
    show("Bird's view", w1);
    show("After chunk", w2);

    #endif

    vector<Mat> parts = this->segment(w3, this->cpMaxlv);

    // gen points and clustering same time
    vector<vector<Point2f>> gclusters = this->gencluster(parts);

    angle = this->calcAngle(gclusters[0], gclusters[1]);

    angle = this->rtd(angle, 90) * -1;

    return angle;

    vector<Point2f> points = this->genpoints(parts);
    
    // this->groupByLine(points);

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

    // clustering #2
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

    return angle;
}

double PathChaser::rtd(double rad, double phi = 0) {
    return (rad * 180) / CV_PI - phi;
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

    vector<vector<Point2f>> groups = this->dbscan(points, 30);
    vector<vector<Point2f>> twcrps; // 3-ways cross road points
    vector<vector<Point2f>> sdpths; // side path road points
    
    Mat cltgrp = Mat::zeros(this->fsize, CV_8UC3);
    int k = 0;
    for_each(groups.begin(), groups.end(), [&](vector<Point2f> group) {
        // display clusters
        this->drawpoints(group, cltgrp, this->randclr());
        
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
    
    clusters.push_back(leftside); // 0
    clusters.push_back(rightside); // 1

    #ifdef shop_clot
    Mat graph5 = Mat::zeros(fsize, CV_8UC1);
    this->drawpoints(points, graph5, this->clrwht);
    
    Mat graph6 = Mat::zeros(fsize, CV_8UC3);
    this->drawpoints(leftside, graph6, this->clrblu);
    this->drawpoints(rightside, graph6, this->clrylw);

    show("before find side", graph5);

    if (leftside.size() == 0 || rightside.size() == 0)
        return clusters;

    Vec4f ll = this->fitlines(leftside);
    Vec4f rl = this->fitlines(rightside);

    int lwb = this->cpFlmLB;
    int upb = this->cpFlmHB;
    Size sf = this->fsize; // after cropped mat
    Size sfs = this->raw_mat.size(); // before cropped mat

    vector<Point2f> lp = this->getlinepoint(ll, sf, lwb, upb);
    vector<Point2f> rp = this->getlinepoint(rl, sf, lwb, upb);

    // three top points
    Point2f alp = lp[1];
    Point2f arp = rp[1];
    Point2f acp = this->mpoint(alp, arp);

    // one center lower point
    Point2f lcp(sfs.width / 2, sfs.height);

    //left and right lower
    Point2f lbp(0, sfs.height);
    Point2f rbp(sfs.width, sfs.height);

    // calculate vectors
    Point2f mlv = this->fvect(acp, lcp); // mid center line vector
    Point2f blv = this->fvect(lcp, lbp); // lower left vector
    Point2f brv = this->fvect(lcp, rbp); // lower right vector

    // cling vector
    Point2f lvt = this->fvect(lcp, alp);
    Point2f rvt = this->fvect(lcp, arp);

    // draw top three points
    this->drawpoint(alp, graph6, this->clrylw);
    this->drawpoint(acp, graph6, this->clrgrn);
    this->drawpoint(arp, graph6, this->clrylw);

    // draw lower center point
    this->drawpoint(lcp, graph6, this->clrgrn);

    // draw two side line
    this->drawline(ll, graph6, this->clrblu);
    this->drawline(rl, graph6, this->clrred);

    // draw connect top and low center points
    line(graph6, acp, lcp, this->clrgrn, 3, 8, 0);

    show("after find side", graph6);
    show("points grouping", cltgrp);
    #endif

    return clusters;
}

double PathChaser::calcAngle(vector<Point2f> left, vector<Point2f> right) {
    double angle = -1;

    if (left.size() == 0 || right.size() == 0) {
        return 0;
    }

    Vec4f ll = this->fitlines(left);
    Vec4f rl = this->fitlines(right);

    int lwb = this->cpFlmLB;
    int upb = this->cpFlmHB;
    Size sf = this->fsize; // after cropped mat
    Size sfs = this->raw_mat.size(); // before cropped mat

    vector<Point2f> lp = this->getlinepoint(ll, sf, lwb, upb);
    vector<Point2f> rp = this->getlinepoint(rl, sf, lwb, upb);

    // three top points
    Point2f alp = lp[1];
    Point2f arp = rp[1];
    Point2f acp = this->mpoint(alp, arp);

    // one center lower point
    Point2f lcp(sfs.width / 2, sfs.height);

    //left and right lower
    Point2f lbp(0, sfs.height);
    Point2f rbp(sfs.width, sfs.height);

    // calculate vectors
    Point2f mlv = this->fvect(acp, lcp); // mid center line vector
    Point2f blv = this->fvect(lcp, lbp); // lower left vector
    Point2f brv = this->fvect(lcp, rbp); // lower right vector

    // cling vector
    Point2f lvt = this->fvect(lcp, alp);
    Point2f rvt = this->fvect(lcp, arp);

    // calculate angle
    // if (this->clingleft) {
    //     angle = this->avect(lvt, brv);
    // } else {
    //     angle = this->avect(rvt, brv);
    // }
    
    angle = this->avect(mlv, brv);

    #ifdef debug
    
    Mat aglmat = this->raw_mat.clone();

    // draw top three points
    this->drawpoint(alp, aglmat, this->clrylw);
    this->drawpoint(acp, aglmat, this->clrgrn);
    this->drawpoint(arp, aglmat, this->clrylw);

    // draw lower center point
    this->drawpoint(lcp, aglmat, this->clrgrn);

    // draw two side line
    this->drawline(ll, aglmat, this->clrblu);
    this->drawline(rl, aglmat, this->clrred);

    // draw connect top and low center points
    line(aglmat, acp, lcp, this->clrgrn, 3, 8, 0);
    // if (this->clingleft) {
    //     line(aglmat, alp, lcp, this->clrgrn, 3, 8, 0);
    // } else {
    //     line(aglmat, arp, lcp, this->clrgrn, 3, 8, 0);
    // }


    show("angle mat", aglmat);

    #endif

    return angle;
}

Point2f PathChaser::mpoint(Point2f a, Point2f b) {
    return Point2f((a.x + b.x) / 2, (a.y + b.y) / 2);
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

    #ifdef debug
    Mat graph = Mat::zeros(fsize, CV_8UC3);
    this->drawpoints(ctpoints, graph, this->clrylw);
    this->drawpoints(left, graph, this->clrwht);
    this->drawpoints(right, graph, this->clrwht);
    show("multiple points", graph);
    #endif
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

Mat PathChaser::skeletonization(Mat inputImage)
{
    Mat outputImage;
    cvtColor(inputImage, outputImage, CV_BGR2GRAY);

    threshold(outputImage, outputImage, 0, 255, THRESH_BINARY+THRESH_OTSU);

    this->thinning(outputImage);

    show("outputImage", outputImage);

    return inputImage;
}

void PathChaser::thinningIteration(Mat& im, int iter)
{
    Mat marker = Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

void PathChaser::thinning(Mat& im)
{
    im /= 255;

    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;

    do {
        this->thinningIteration(im, 0);
        this->thinningIteration(im, 1);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (countNonZero(diff) > 0);

    im *= 255;
}

void PathChaser::groupByLine(vector<Point2f> points) {
     Mat frame = Mat::zeros(this->fsize, CV_8UC3);

    Point2f A(this->fsize.width / 2, 0);
    Point2f B = this->aglpoint(this->fsize.height);

    this->drawpoint(A, frame, this->clrred);
    this->drawpoint(B, frame, this->clrred);

    
    for (int i = 0; i < points.size(); i++) {
        this->drawpoint(points.at(i), frame, this->clrblu);
    }

    line(frame, A, B, this->clrylw);

    // calculate distance
    vector<int> dists;
    vector<vector<int>> indices;
    for (int i = 0; i < points.size(); i++) {
        Point2f p = points.at(i);
        int d = this->calcPointDist(p);
        dists.push_back(d);
    }
    
    vector<vector<int>> groups = this->kmeand(dists, indices, this->cpbDist);


    for (int i = 0; i < groups.size(); i++) {
        Scalar clr = this->randclr();
        for (int j = 0; j < groups.at(i).size(); j++) {
            this->drawpoint(points.at(indices.at(i).at(j)), frame, clr);
        }
    }

    show(this->dbwn9, frame);
}

void PathChaser::testAlgorithm() {

    vector<Point2f> points;
    for (int i = 0; i < 100; i++) {
        double x = this->rng.uniform(0, 640);
        double y = this->rng.uniform(0, 480);
        Point2f p(x, y);
        points.push_back(p);
    }

    this->groupByLine(points);
   
    waitKey();

    // vector<int> items;
    // items.push_back(5);
    // items.push_back(1);
    // items.push_back(4);
    // items.push_back(3);
    
    // items.push_back(10);

    // items.push_back(15);
    // items.push_back(20);
    // items.push_back(17);
    

    // /* 
    // sample inputs: 
    // 1,3,2,0  4   5  7  6
    // 1,3,4,5, 10, 15,17,20

    // d = 3: 1,3,4,5 10 15,17,20
    // d = 5: 1,3,4,5,10,15,17,20;

    // */
    // vector<vector<int>> indices;
    // vector<vector<int>> groups = this->kmeand(items, indices, 5);


    // for (int i = 0; i < groups.size(); i++) {
    //     cout << "Group " << i + 1 << endl;
    //     for (int j = 0; j < groups.at(i).size(); j++) {
    //         cout << groups.at(i).at(j) << " " << indices.at(i).at(j) << endl;
    //     }
    // }

}

void PathChaser::video(string v, int wk) {

    #ifdef test_algorithm
    this->testAlgorithm();
    return;
    #endif 

    VideoCapture cap(v);

    cap.set(CV_CAP_PROP_POS_FRAMES, this->frs);

    Mat frame;
    this->frc = this->frs;
    bool start = false;
    bool pause = false;

    this->frc --;
    while (1) {
        cap >> frame;
        this->current_ticks = clock();

        this->frc++;
        if (this->frc >= cap.get(cv::CAP_PROP_FRAME_COUNT)) {
            this->frc = 1;
            cap.set(CV_CAP_PROP_POS_FRAMES, 1);
        }

        if (frame.empty()) continue;
        
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

        double angle = this->roadline(frame);
        cout << angle << endl;

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

        this->delta_ticks = clock() - this->current_ticks;
        if(delta_ticks > 0) this->fps = CLOCKS_PER_SEC / this->delta_ticks;
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
    String val2 = "FPS: " + to_string(this->fps);
    Point pos = Point(10, frame.rows - 10);
    Point pos2 = Point(10, frame.rows - 28);
    cv::putText(frame, val, pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, this->clrwht, 1, 8);
    cv::putText(frame, val2, pos2, cv::FONT_HERSHEY_SIMPLEX, 0.6, this->clrwht, 1, 8);
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

Mat PathChaser::trapeziumroi(Mat &frame) {
    int rows = frame.rows;
    int cols = frame.cols;
    
    
    Mat raw = frame.clone();
    Mat img = Mat::zeros(frame.size(), CV_8UC3);
    Mat mask(frame.size(), CV_8UC1, Scalar::all(0));
    Mat spl = frame.clone();

    Size roisize(this->tbxLW, this->tbxH);

    Point a, b, c, d;
    if (this->tbxX == 0) this->tbxX = rows / 2 + this->tbxLW / 2;
    if (this->tbxX >= frame.cols) this->tbxX = this->tbxX % frame.cols;
    if (this->tbxY >= frame.rows) this->tbxY = this->tbxY % frame.rows;

    a = Point(this->tbxX - this->tbxLW / 2, rows - this->tbxY);
    b = Point(a.x + (this->tbxLW - this->tbxHW) / 2, a.y - this->tbxH);
    c = Point(b.x + this->tbxHW, b.y);
    d = Point(a.x + this->tbxLW, a.y);

    line(spl, a, b, this->clrwht);
    line(spl, b, c, this->clrwht);
    line(spl, c, d, this->clrwht);
    line(spl, d, a, this->clrwht);

    vector<Point> points;
    
    points.push_back(a);
    points.push_back(b);
    points.push_back(c);
    points.push_back(d);

    vector<Point> roipoly;

    approxPolyDP(points, roipoly, 1.0, true);

    fillConvexPoly(mask, &roipoly[0], roipoly.size(), 255, 8, 0);

    raw.copyTo(img, mask);
    
    // crop black edges
    Mat bin;
    inRange(img, Scalar(0,0,0), Scalar(0,0,0), bin);
    bitwise_not(bin, bin);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(bin, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Point> poly;

    if (contours.size() > 0) {
        vector<Point> contour = contours[0];
        approxPolyDP(Mat(contour), poly, 10, true);

        Rect rect = boundingRect(poly);

        img = img(rect);

        #ifdef debug
        #ifdef debug_chunk_trapazoid
        string roiname = "Trapazoid ROI";
        imshow(roiname, img);
        show(this->dbwn8, spl);
        #endif
        #endif
    }

    return img;

}

void PathChaser::conv2(const cv::Mat &img, const cv::Mat& kernel, PathChaser::ConvolutionType type, cv::Mat& dest) {
	cv::Mat source = img;
	if(PathChaser::CONVOLUTION_FULL == type) {
		source = Mat();
		const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
		copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
	}

	cv::Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
	int borderMode = BORDER_CONSTANT;
	cv::Mat fkernel;
	flip(kernel, fkernel, -1);
	cv::filter2D(source, dest, CV_64F, fkernel, anchor, 0, borderMode);

	if(PathChaser::CONVOLUTION_VALID == type) {
		dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
           .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
	}
}

double PathChaser::sqr(double x) {
	return x * x;
}

bool PathChaser::isInRange(double val, double l, double r) {
	return (l <= val && val <= r);
}

void PathChaser::waveletTransform(const cv::Mat& img, cv::Mat& edge, double threshold = 0.15) {
    // https://github.com/fpt-corp/DriverlessCarChallenge_2017-2018/blob/master/example/lane_detection/api_lane_detection.cpp
    
	Mat src = img;
	if (img.type() == 16)
		cvtColor(img, src, CV_BGR2GRAY);
	double pi = M_PI;
	int SIZE = src.rows;
	int SIZE1 = src.cols;
	double m = 1.0;
	double dlt = pow(2.0, m);
	int N = 20;
	double A = -1 / sqrt(2 * pi);//M_PI = acos(-1.0)
	cv::Mat phi_x = cv::Mat(N, N, CV_64F);
	cv::Mat phi_y = cv::Mat(N, N, CV_64F);
	for(int idx = 1; idx <= N; ++idx) {
        for(int idy = 1; idy <= N; ++idy) {
			double x = idx - (N + 1) / 2.0;
			double y = idy - (N + 1) / 2.0;
			double coff = A / sqr(dlt) * exp(-(sqr(x) + sqr(y)) / (2 * sqr(dlt)));
			phi_x.at<double>(idx - 1, idy - 1) = (coff * x);
			phi_y.at<double>(idx - 1, idy - 1) = (coff * y);
		}
    }
		
	normalize(phi_x, phi_x);
	normalize(phi_y, phi_y);
	cv::Mat Gx, Gy;
	conv2(src, phi_x, CONVOLUTION_SAME, Gx);
	conv2(src, phi_y, CONVOLUTION_SAME, Gy);
	cv::Mat Grads = cv::Mat(src.rows, src.cols, CV_64F);
	for(int i = 0; i < Gx.rows; ++i)
		for(int j = 0; j < Gx.cols; ++j) {
			double x = Gx.at<double>(i, j);
			double y = Gy.at<double>(i, j);
			double sqx = sqr(x);
			double sqy = sqr(y);
			Grads.at<double>(i, j) = sqrt(sqx + sqy);
		}
	double mEPS = 100.0 / (1LL << 52);//matlab eps = 2 ^ -52
	cv::Mat angle_array = cv::Mat::zeros(SIZE, SIZE1, CV_64F);
	for(int i = 0; i < SIZE; ++i)
		for(int j = 0; j < SIZE1; ++j) {
			double p = 90;
			if (fabs(Gx.at<double>(i, j)) > mEPS) {
				p = atan(Gy.at<double>(i, j) / Gx.at<double>(i, j)) * 180 / pi;
				if (p < 0) p += 360;
				if (Gx.at<double>(i, j) < 0 && p > 180)
					p -= 180;
				else if (Gx.at<double>(i, j) < 0 && p < 180)
					p += 180;
			}
			angle_array.at<double>(i, j) = p;
		}
	Mat edge_array = cv::Mat::zeros(SIZE, SIZE1, CV_64F);
	for(int i = 1; i < SIZE - 1; ++i) {
        for(int j = 1; j < SIZE1 - 1; ++j) {
			double aval = angle_array.at<double>(i, j);
			double gval = Grads.at<double>(i, j);
			if (this->isInRange(aval,-22.5,22.5) || this->isInRange(aval, 180-22.5,180+22.5)) {
				if (gval > Grads.at<double>(i+1,j) && gval > Grads.at<double>(i-1,j))
					edge_array.at<double>(i, j) = gval;
			}
			else
			if (this->isInRange(aval,90-22.5,90+22.5) || this->isInRange(aval,270-22.5,270+22.5)) {
				if (gval > Grads.at<double>(i, j+1) && gval > Grads.at<double>(i, j-1))
					edge_array.at<double>(i, j) = gval;
			}
			else
			if(this->isInRange(aval,45-22.5,45+22.5) || this->isInRange(aval,225-22.5,225+22.5)) {
				if (gval > Grads.at<double>(i+1,j+1) && gval > Grads.at<double>(i-1,j-1))
					edge_array.at<double>(i,j) = gval;
			}
			else
				if (gval > Grads.at<double>(i+1,j-1) && gval > Grads.at<double>(i-1,j+1))
					edge_array.at<double>(i, j) = gval;
		}
    }
		
	double MAX_E = edge_array.at<double>(0, 0);
	for(int i = 0; i < edge_array.rows; ++i)
		for(int j = 0; j < edge_array.cols; ++j)
			if (MAX_E < edge_array.at<double>(i, j))
				MAX_E = edge_array.at<double>(i, j);
	edge = Mat::zeros(src.rows, src.cols, CV_8U);
	for(int i = 0; i < edge_array.rows; ++i)
		for(int j = 0; j < edge_array.cols; ++j) {
			edge_array.at<double>(i, j) /= MAX_E;
			if (edge_array.at<double>(i, j) > threshold)
				edge.at<uchar>(i, j) = 255;
			else
				edge.at<uchar>(i, j) = 0;
		}
}

Mat PathChaser::chunk(Mat frame) {
    Mat image = frame.clone();

    image = this->isolate(image);

    Mat roiframe = image.clone();
    Mat roiimg = image.clone();
    roiimg = this->trapeziumroi(roiframe);

    // find min max value of color
    this->minmax(roiimg, true);

    // update scalar value
    this->updsclr();

    // get mask from new color range
    Mat blur = image.clone();
    cv::blur(image, blur, Size(5, 5));

    // get road color and mask
    Mat mask = Mat::zeros(image.size(), CV_8UC1), hsv;
    cvtColor(blur, hsv, COLOR_BGR2HSV);
    inRange(hsv, this->lower, this->upper, mask);

    for (int r = 1; r < 6; r++) {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*r+1, 2*r+1));
        morphologyEx(mask, mask, CV_MOP_CLOSE, kernel);
        morphologyEx(mask, mask, CV_MOP_OPEN, kernel);
    }

    // the largest contour is often the main road area
    vector<vector<Point>> contours;
    findContours(mask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // find largest contour
    double maxarea = -1;
    for_each(contours.begin(), contours.end(), [&](vector<Point> contour) {
        double area = contourArea(contour);
        if (maxarea == -1) {
            maxarea = area;
        } else {
            if (area > maxarea) {
                maxarea = area;
                contours[0] = contour;
            }
        }
    });

    Mat fnlmask = Mat::zeros(image.size(), CV_8UC1);
    drawContours(fnlmask, contours, 0, this->clrwht, CV_FILLED);

    Mat cropmask = Mat::zeros(image.size(), CV_8UC3);
    image.copyTo(cropmask, fnlmask);

    drawContours(cropmask, contours, 0, this->clrwht, 3, 8);

    #ifdef show_cloak
    show("before chunk", cropmask);
    #endif

    cropmask = this->preprocess(cropmask);
    equalizeHist(cropmask, cropmask);

    return cropmask;
}

vector<Vec4i> PathChaser::findHoughLines(Mat mask, Mat &output) {
    vector<Vec4i> lines;
    HoughLinesP(mask, lines, 1, CV_PI/180, 80, 30, 10 );
    for (int i = 0; i < lines.size(); i++) {
         line(output, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }
    return lines;
}

double PathChaser::calcPointDist(Point2f M) {
    double d = 0;

    Point2f O = Point2f(this->fsize.width / 2, 0);
    Point2f B = this->aglpoint(this->fsize.height);

    Point2f dv = this->fvect(O, B); // directive vector
    Point2f rv = Point2f(dv.y, - dv.x); // relative vector

    /*
        a(x - x0) + b(y - y0) = 0;
        <=> ax - ax0 + by - by0 = 0
        <=> ax + by - ax0 - by0 = 0
        => c = - ax0 - by0
    */

    double a, b, c;
    a = rv.x;
    b = rv.y;
    c = - (a * O.x + b * O.y);
    
    d = abs(a * M.x + b * M.y + c) / sqrt(a * a + b * b);

    // double algd = this->cpblAgl;
    // double aglr = algd * CV_PI / 180;
    // double a, b, c, d; // a,b in y = ax + b

    // Point2f R;

    // if (algd > 90) a = tan(aglr) > 0 ? - tan(aglr) : tan(aglr);
    // if (algd < 90) a = tan(aglr) < 0 ? - tan(aglr) : tan(aglr);

    // b = O.y - a * O.x;

    // c = b, b = -1,

    // d = abs(a * M.x + b * M.y + c) / sqrt(a * a + b * b);

    return d;
}

Point2f PathChaser::aglpoint(int y = 0) {
    double algd = this->cpblAgl;
    double aglr = algd * CV_PI / 180;
    double a, b; // a,b in y = ax + b

    Point2f O(this->fsize.width / 2, 0);
    Point2f R;

    if (algd > 90) a = tan(aglr) > 0 ? - tan(aglr) : tan(aglr);
    if (algd < 90) a = tan(aglr) < 0 ? - tan(aglr) : tan(aglr);

    if (algd != 90) {
        b = O.y - a * O.x;
        R.x = (y - b) / a;
    } else {
        R.x = 0;
    }

    R.y = y;

    return R;

}

vector<vector<int>> PathChaser::kmeand(vector<int> items, vector<vector<int>> &indices, int md) {
    vector<vector<int>> groups;

    // build indices dictionary
    map<int, int> idict;
    for (int i = 0; i < items.size(); i++)
        idict[items.at(i)] = i;

    this->countsort(items);

    stack<int> s;

    int i = 0;
    s.push(items.at(i++));

    while (!s.empty()) {
        int a = s.top();
        int b = items.at(i++);

        if (abs(a - b) <= md) {
            s.push(b);
        } else {
            vector<int> group;
            vector<int> gidxs;
            while (!s.empty()) {
                int v = s.top();
                group.push_back(v);
                gidxs.push_back(idict.at(v));
                s.pop();
            }
            groups.push_back(group);
            indices.push_back(gidxs);
            s.push(b);
        }

        if (i == items.size()) break;
    }

    vector<int> group;
    vector<int> gidxs;
    while (!s.empty()) {
        int v = s.top();
        group.push_back(v);
        gidxs.push_back(idict.at(v));
        s.pop();
    }
    groups.push_back(group);
    indices.push_back(gidxs);

    return groups;
}

void PathChaser::countsort(vector<int> &v) {
    std::map<int, int> freq;
    for (int i = 0; i < v.size(); i++) freq[v.at(i)]++;
    int i = 0;
    for (auto it: freq) {
        while (it.second--) v[i++] = it.first;
    }
}

void PathChaser::magicwand(Mat image) {
    Rect cbx = this->mkcbox(image);
    Mat reg = image(cbx);

    #ifdef debug
    // frame = this->draw(image, cbx, "box");
    // show("origin", frame);
    #endif

    Mat hsv, mask;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    inRange(hsv, this->lower, this->upper, mask);

    // show("ctr0", mask);

    vector<Rect> roirects;

    Mat ctr1 = this->findCtrMask(mask, roirects);
    show("ctr1", ctr1);

    rectangle(image, cbx, this->clrwht, 2);

    string a = format("%d %d %d", (int)this->lower[0], (int)this->lower[1], (int)this->lower[2]);
    string b = format("%d %d %d", (int)this->upper[0], (int)this->upper[1], (int)this->upper[2]);
    putText(image, a, Point(cbx.x, cbx.y), FONT_HERSHEY_COMPLEX, 0.5, this->clrwht, 1, 8, 0);
    putText(image, b, Point(cbx.x, cbx.y + 10), FONT_HERSHEY_COMPLEX, 0.5, this->clrwht, 1, 8, 0);

    for (int i = 0; i < roirects.size(); i++) {
        Rect rect = roirects.at(i);
        Mat roi = image(rect);
        Scalar min, max;
        this->minmaxroi(roi, min, max);

        Scalar color = this->randclr();

        string s1 = format("%d %d %d", (int)min[0], (int)min[1], (int)min[2]);
        string s2 = format("%d %d %d", (int)max[0], (int)max[1], (int)max[2]);
        putText(image, s1, Point(rect.x, rect.y), FONT_HERSHEY_COMPLEX, 0.5, color, 1, 8, 0);
        putText(image, s2, Point(rect.x, rect.y + 10), FONT_HERSHEY_COMPLEX, 0.5, color, 1, 8, 0);

        rectangle(image, rect, color, 1);
    }

    show("magic wand result", image);
}

Mat PathChaser::findCtrMask(Mat bin, vector<Rect> &roirects) {
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(bin, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect(contours.size());
    vector<Point2f>center(contours.size());
    vector<float>radius(contours.size());

    for(int i = 0; i < contours.size(); i++) {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 10, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }   

    Mat morph = bin.clone();

    for (int r = 1; r < 4; r++) {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*r+1, 2*r+1));
        morphologyEx(morph, morph, CV_MOP_CLOSE, kernel);
        morphologyEx(morph, morph, CV_MOP_OPEN, kernel);
    }

    Mat drawing = Mat::zeros(morph.size(), CV_8UC3 );
    for(int i = 0; i < contours.size(); i++ ) {
        double area = contourArea(contours.at(i));

        if (area <= 50) continue;
    
        Scalar color = this->randclr();
        drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        
        vector<Point> ctrpoly = contours_poly[i];
        for (int j = 0; j < ctrpoly.size(); j++) {
            if (j % 3 == 0) {
                color = this->randclr();
                int r = 20;
                Rect rect(ctrpoly[j].x - r, ctrpoly[j].y - r, r * 2, r * 2);
                if (rect.x >= 0 && rect.y >= 0 && rect.x + rect.width < bin.cols && rect.y + rect.height < bin.rows) {
                    roirects.push_back(rect);
                    rectangle(drawing, rect, color, 1);
                }
            }
        }

        // rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        // circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
    }
    
    show("drawing", drawing);

    return morph;

}

void PathChaser::minmaxroi(Mat roi, Scalar &min, Scalar &max) {
    /*
    * Feature: pass lower and upper color of rame to min, max variable of class
    */

   min = Scalar(255, 255, 255);
   max = Scalar(0, 0, 0);

    Mat hsv; cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    int cn = hsv.channels();
    int cl = hsv.cols;
    int rw = hsv.rows;

    for(int y = 0; y < rw; y++) {
        for(int x = 0; x < cl; x++) {
            /* B0-G1-R2 */
            int b = hsv.at<cv::Vec3b>(y,x)[0];
            int g = hsv.at<cv::Vec3b>(y,x)[1];
            int r = hsv.at<cv::Vec3b>(y,x)[2];

            min[0] = b < min[0] ? b : min[0];
            min[1] = g < min[1] ? g : min[1];
            min[2] = r < min[2] ? r : min[2];

            max[0] = b > max[0] ? b : max[0];
            max[1] = g > max[1] ? g : max[1];
            max[2] = r > max[2] ? r : max[2];
        }
    }

    #ifdef debug
        // cout << min << " " << max << endl;
    #endif
}

void PathChaser::minmax(Mat frame, bool e) {
    /*
    * Feature: pass lower and upper color of rame to min, max variable of class
    */

    // stop get color range when it's over limit
    if (this->fcp > 0) { // if seed value is 0, means get color from all frame
        if (this->frc >= this->fcp) return;
    }

    // if param 'e' is set to true, means every frame is a new color range
    #ifndef debug_colorsearch 
    if (e) {
        for (int i = 0; i < 3; i++) {
            this->min[i] = 255;
            this->max[i] = 0;
        }
    }
    #endif

    Mat hsv; cvtColor(frame, hsv, cv::COLOR_BGR2HSV);


    int cn = hsv.channels();
    int cl = hsv.cols;
    int rw = hsv.rows;

    for(int y = 0; y < rw; y++) {
        for(int x = 0; x < cl; x++) {
            /* B0-G1-R2 */
            int b = hsv.at<cv::Vec3b>(y,x)[0];
            int g = hsv.at<cv::Vec3b>(y,x)[1];
            int r = hsv.at<cv::Vec3b>(y,x)[2];

            if (b == 0 && g == 0 && r == 0) continue;

            #ifndef debug_colorsearch 
            this->min[0] = b < this->min[0] ? b : this->min[0];
            this->min[1] = g < this->min[1] ? g : this->min[1];
            this->min[2] = r < this->min[2] ? r : this->min[2];

            // this->max[0] = b > this->max[0] ? b : this->max[0];
            // this->max[1] = g > this->max[1] ? g : this->max[1];
            // this->max[2] = r > this->max[2] ? r : this->max[2];
            
            this->max[0] = 255;
            this->max[1] = 255;
            this->max[2] = 255;

            #endif
        }
    }
}

void PathChaser::updsclr() {
    this->lower = Scalar(this->min[0], this->min[1], this->min[2]);
    this->upper = Scalar(this->max[0], this->max[1], this->max[2]);
}

string PathChaser::print(int *arr, int n) {
    string s = "";
    for (int i = 0; i < n; i++) {
        s += to_string(arr[i]) + " ";
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

Mat PathChaser::masknoisermv(Mat mask) {
    Size erosize = Size(this->ppEroW + 1, this->ppEroH + 1);
    Point eropnt = Point(this->ppEroW, this->ppEroH);
    Mat eroele = getStructuringElement(this->ppEroType, erosize, eropnt);
    cv::erode(mask, mask, eroele, Point(0, 0), this->ppEroIt);

    Size dltsize = Size(this->ppDltW + 1, this->ppDltH + 1);
    Point dltpnt = Point(this->ppDltW, this->ppDltH);
    Mat dltele = getStructuringElement(this->ppDltType, dltsize, dltpnt);
    cv::dilate(mask, mask, dltele, Point(0, 0), this->ppDltIt);

    // this->ctrclean(image);

    #ifdef debug_noiseremove
    show(this->dbwn3, mask);
    #endif
    return mask;
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

    // color box trapazoid
    this->tbxLW = this->param("clrtrp_lw");
    this->tbxHW = this->param("clrtrp_hw");
    this->tbxH = this->param("clrtrp_h");
    this->tbxX = this->param("clrtrp_x");
    this->tbxY = this->param("clrtrp_y");

    // isolate box
    this->isobox[0] = this->param("crpb_top");
    this->isobox[1] = this->param("crpb_right");
    this->isobox[2] = this->param("crpb_bottom");
    this->isobox[3] = this->param("crpb_left");

    // color box
    this->clrbox[0] = this->param("clrbox_x");
    this->clrbox[1] = this->param("clrbox_y");
    this->clrbox[2] = this->param("clrbox_w");
    this->clrbox[3] = this->param("clrbox_h");

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
    // this->dbwn9 = "Clustering points by line";
    // namedWindow(this->dbwn9, 1);
    // createTrackbar("Angle", this->dbwn9, &this->cpblAgl, 180, PathChaser::debugger, this);
    // createTrackbar("Distance", this->dbwn9, &this->cpbDist, 1000, PathChaser::debugger, this);

    #ifdef debug_chunk_trapazoid
    this->dbwn8 = "Trapazoid ROI Area";
    namedWindow(this->dbwn8, 1);
    createTrackbar("X", this->dbwn8, &this->tbxX, 1000, PathChaser::debugger, this);
    createTrackbar("Y", this->dbwn8, &this->tbxY, 1000, PathChaser::debugger, this);
    createTrackbar("LW", this->dbwn8, &this->tbxLW, 1000, PathChaser::debugger, this);
    createTrackbar("HW", this->dbwn8, &this->tbxHW, 1000, PathChaser::debugger, this);
    createTrackbar("H", this->dbwn8, &this->tbxH, 1000, PathChaser::debugger, this);
    #endif

    #ifdef debug_rmnoise_mask
    this->dbwn7 = "remove noise mask";
    namedWindow(this->dbwn7, 1);
    createTrackbar("L B", this->dbwn7, &this->min[0], 255, PathChaser::debugger, this);

    #endif

    #ifdef debug_colorsearch
    this->dbwn6 = "color search mask";
    namedWindow(this->dbwn6, 1);

    createTrackbar("L B", this->dbwn6, &this->min[0], 255, PathChaser::debugger, this);
    createTrackbar("L G", this->dbwn6, &this->min[1], 255, PathChaser::debugger, this);
    createTrackbar("L R", this->dbwn6, &this->min[2], 255, PathChaser::debugger, this);

    createTrackbar("H B", this->dbwn6, &this->max[0], 255, PathChaser::debugger, this);
    createTrackbar("H G", this->dbwn6, &this->max[1], 255, PathChaser::debugger, this);
    createTrackbar("H R", this->dbwn6, &this->max[2], 255, PathChaser::debugger, this);
    #endif 

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

    d->roadline(d->raw_mat);
}
