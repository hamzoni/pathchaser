#include <iostream>
#include "pathchaser.cpp"
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main() {
    PathChaser *p = new PathChaser();
    // p->frs = 362;
    // p->frs = 462;
    // p->frs = 614;
    p->frs = 688;
    p->video("/home/taquy/Desktop/vid/18.avi", 200);
    
    return -1;
}