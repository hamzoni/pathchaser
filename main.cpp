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
    // p->frs = 525; // nga 3
    // p->frs = 170; // nga 3 lan 1
    p->frs = 1;
    // p->frs = 189; // nga 3 lan 2
    p->video("/home/taquy/Desktop/vid/18.avi", 400);
    // p->video("/home/taquy/Desktop/vid/23.avi", 200);
    

    return -1;
}
// 
