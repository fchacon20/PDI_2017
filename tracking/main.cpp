#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <string>
#include <math.h>

using namespace cv;
using namespace std;

int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS= 100;

//minimum and maximum object area
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;

void on_trackbar( int, void* )
{//This function gets called whenever a
    // trackbar position is changed
}

string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void createTrackbars(){
    //create window for trackbars
    namedWindow(trackbarWindowName,0);
    //create memory to store trackbar name on window
    //char TrackbarName[50];
    /*sprintf( TrackbarName, "H_MIN", H_MIN);
    sprintf( TrackbarName, "H_MAX", H_MAX);
    sprintf( TrackbarName, "S_MIN", S_MIN);
    sprintf( TrackbarName, "S_MAX", S_MAX);
    sprintf( TrackbarName, "V_MIN", V_MIN);
    sprintf( TrackbarName, "V_MAX", V_MAX);*/
    //create trackbars and insert them into window
    //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    //the max value the trackbar can move (eg. H_HIGH),
    //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
    //                                  ---->    ---->     ---->
    createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );
}

void drawObject(int x, int y,Mat &frame){

    //use some of the openCV drawing functions to draw crosshairs
    //on your tracked image!

    circle(frame,Point(x,y),20,Scalar(0,255,0),2);
    if(y-25>0)
        line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);

    if(y+25<FRAME_HEIGHT)
        line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),2);

    if(x-25>0)
        line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);

    if(x+25<FRAME_WIDTH)
        line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),2);

    putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);
}

void morphOps(Mat &thresh){

    //create structuring element that will be used to "dilate" and "erode" image.
    //the element chosen here is a 3px by 3px rectangle

    Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
    //dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));

    erode(thresh,thresh,erodeElement);
    erode(thresh,thresh,erodeElement);

    dilate(thresh,thresh,dilateElement);
    dilate(thresh,thresh,dilateElement);

}

void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed){

    Mat temp;
    threshold.copyTo(temp);
    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
    //use moments method to find our filtered object
    double refArea = 0;
    bool objectFound = false;
    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        //if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
        if(numObjects<MAX_NUM_OBJECTS){
            for (int index = 0; index >= 0; index = hierarchy[index][0]) {

                Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;

                //if the area is less than 20 px by 20px then it is probably just noise
                //if the area is the same as the 3/2 of the image size, probably just a bad filter
                //we only want the object with the largest area so we safe a reference area each
                //iteration and compare it to the area in the next iteration.
                if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea){
                    x = moment.m10/area;
                    y = moment.m01/area;
                    //drawObject(x,y,cameraFeed);
                    objectFound = true;
                    refArea = area;
                }else objectFound = false;


            }
            //let user know you found an object
            if(objectFound ==true){
                putText(cameraFeed,"Siguiendo un objeto",Point(0,50),2,1,Scalar(0,255,0),2);
                //draw object location on screen
                drawObject(x,y,cameraFeed);
            }

        }else putText(cameraFeed,"Demasiado ruido. Ajustar el filtro",Point(0,50),1,2,Scalar(0,0,255),2);
    }
}

float initialVelocity(vector<Point2f> V){
    return(sqrt(pow(V[1].x,2) + pow(V[1].y,2)));
}

Point2f getVelocity(vector<Point2f> P, float x0, float y0){
    return(Point2f((x0 - P.back().x)*30, (y0 - P.back().y)*30));
}


Point2f getAcceleration(vector<Point2f> V, float vx0, float vy0){
    return(Point2f((vx0 - V.back().x)*30, (vy0 - V.back().y)*30));
}

vector<Point2f> getAngularVelocity(vector<Point2f> tVel, float r){
    vector<Point2f> angularVelocity;
    for(vector<Point2f>::iterator it = tVel.begin(); it != tVel.end(); ++it){
        angularVelocity.push_back(*it/r);
    }
    return(angularVelocity);
}

float getAngularAcceleration(float a ,float r){
    float alfa = a/r;
    return(alfa);
}

float getCentripetalAcceleration(float v, float r){
    float acc = pow(v,2)/r;
    return acc;
}

bool calibration(int &x, int &y, Mat threshold,  Mat &cameraFeed, vector<Point2f> &pos){
    Mat temp;
    threshold.copyTo(temp);
    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
    //use moments method to find our filtered object
    double refArea = 0;
    bool objectFound = false;

    //vector<Point2f> pos;

    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        //if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
        if(numObjects == 4){
            for (int index = 0; index >= 0; index = hierarchy[index][0]) {

                Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;

                //if the area is less than 20 px by 20px then it is probably just noise
                //if the area is the same as the 3/2 of the image size, probably just a bad filter
                //we only want the object with the largest area so we safe a reference area each
                //iteration and compare it to the area in the next iteration.
                //if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea){
                    pos.push_back(Point2f(moment.m10/area, moment.m01/area));
                    //cout << moment.m10/area << " " << moment.m01/area << endl;
                    //objectFound = true;
                    //refArea = area;
                //}else objectFound = false;

            }

            return true;
            //let user know you found an object
            if(objectFound == true){
                putText(cameraFeed,"Siguiendo un objeto",Point(0,50),2,1,Scalar(0,255,0),2);
                //draw object location on screen
                drawObject(x,y,cameraFeed);
            }

        }else putText(cameraFeed,"Demasiado ruido. Ajustar el filtro",Point(0,50),1,2,Scalar(0,0,255),2);
        return false;
    }
}


int main(){

    bool trackObjects = true;
    bool useMorphOps = true;
    Mat HSV;
    Mat threshold;
    Mat videoFeed;
    Mat PH = Mat(3,1, CV_32FC1);
    Mat realPoints = Mat(3,1,CV_32FC1);
    Mat H;

    //Parametro debe ser entregado por el usuario a través de la interfaz, este es en Kg.
    float masa = 10.0;
    float gravedad = 9.81;

    //Elegido por el usuario
    int mov = 5;
    /*
     * 0 -> Movimiento Lineal
     * 1 -> Plano Inclinado
     * 2 -> Lanzamiento proyectiles
     * 3 -> Movimiento Circular
     * 4 -> Movimiento Pendular
     */

    //x and y values for the location of the object
    int x=0, y=0;
    float X=0, Y=0;

    //create slider bars for HSV filtering
    //Descomentar para calibracion
    createTrackbars();

    //video capture object to acquire webcam feed
    cv::VideoCapture vid;

    switch (mov) {
        case 0:
            vid.open("MLineal.mp4");
            break;
        case 1:
            vid.open("Inclinado.mp4");
            break;
        case 2:
            vid.open("Proyectiles.mp4");
            break;
        case 3:
            vid.open("MCU.mp4");
            break;
        case 4:
            vid.open("Pendular.mp4");
            break;
        default:
            vid.open(0);
            break;
    }

    //Frecuencia para movimento pendular
    int hz = 0;
    int direction = 1; //parte con velocidad positiva

    bool isCalibrated = false;

    //Variables de movimiento
    float x0 = 0.0, y0 = 0.0;
    float vx0 = 0.0, vy0 = 0.0;
    vector<Point2f> P; //Position
    vector<Point2f> V; //Velocity
    vector<Point2f> A; //Aceleration
    vector<Point2f> pos;

    vector<Point2f> dst;

    dst.push_back(Point2f(0,0));
    dst.push_back(Point2f(0,10));
    dst.push_back(Point2f(10,0));
    dst.push_back(Point2f(10,10));

    //start an infinite loop where webcam feed is copied to videoFeed matrix
    //all of our operations will be performed within this loop
    while(1){
        while(!isCalibrated){
            vid >> videoFeed;

            if (videoFeed.empty())
                    break;

            cvtColor(videoFeed,HSV,COLOR_BGR2HSV);

            //filter HSV image between values and store filtered image to
            //threshold matrix

            //Parametros para el punto rojo de los ejemplos, si se utiliza otro objeto,
            //es necesario "perillarlos" de nuevo
            H_MIN = 139;
            H_MAX = 256;
            S_MIN = 103;
            S_MAX = 256;
            V_MIN = 1;
            V_MAX = 256;

            inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);

            //perform morphological operations on thresholded image to eliminate noise
            //and emphasize the filtered object(s)
            if(useMorphOps)
                morphOps(threshold);

            //pass in thresholded frame to our object tracking function
            //this function will return the x and y coordinates of the
            //filtered object

            isCalibrated = calibration(x,y,threshold,videoFeed, pos);

            while(1){
                //show frames
            imshow("Threshold",threshold);
            imshow("Raw video",videoFeed);
                //imshow("HSV",HSV);
                if( (waitKey(10)) != -1)
                    break;
            }
            if(pos.size() != 0){
                vector<float> posX;
                vector<Point2f> newPos;

                for (std::vector<Point2f>::iterator it = pos.begin() ; it != pos.end(); ++it){
                    posX.push_back(it->x);
                }

                sort(posX.begin(), posX.end());

                //Orden por x
                for (std::vector<float>::iterator it2 = posX.begin() ; it2 != posX.end(); ++it2){
                    for (std::vector<Point2f>::iterator it = pos.begin() ; it != pos.end(); ++it){
                        if(*it2 == it->x){
                            newPos.push_back(*it);
                            break;
                        }
                    }
                }

                //Orden por y
                if(newPos[0].y > newPos[1].y)
                    iter_swap(newPos.begin(), newPos.begin()+1);
                if(newPos[2].y > newPos[3].y)
                    iter_swap(newPos.begin()+2, newPos.begin()+3);

                pos = newPos;
                /*for (std::vector<Point2f>::iterator it = pos.begin() ; it != pos.end(); ++it){
                    cout << *it << endl;
                }*/

                H = findHomography(pos, dst);
            }
        }

        /*PH.at<float>(0,0) = x;
        PH.at<float>(1,0) = y;
        PH.at<float>(2,0) = 1;*/


        //X = realPoints.at<float>(0,0)/realPoints.at<float>(3,0);
        //Y = realPoints.at<float>(1,0)/realPoints.at<float>(3,0);

        /*cout << H.at<float>(0,0) << endl;
        cout << H.at<float>(1,0) << endl;
        cout << H.at<float>(2,0) << endl;*/

        /*realPoints.at<float>(0,0) = 1;
        realPoints.at<float>(1,0) = 2;
        realPoints.at<float>(2,0) = 3;*/

        //cout << realPoints.at<float>(1,0) << endl;
        //cout << realPoints << endl;

        vid >> videoFeed;

        if (videoFeed.empty())
                break;

        cvtColor(videoFeed,HSV,COLOR_BGR2HSV);

        //filter HSV image between values and store filtered image to
        //threshold matrix

        //Parametros para el punto rojo de los ejemplos, si se utiliza otro objeto,
        //es necesario "perillarlos" de nuevo
        //Detectar negro
        H_MIN = 81;
        H_MAX = 190;
        S_MIN = 0;
        S_MAX = 78;
        V_MIN = 33;
        V_MAX = 113;

        inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);

        //perform morphological operations on thresholded image to eliminate noise
        //and emphasize the filtered object(s)
        if(useMorphOps)
            morphOps(threshold);

        //pass in thresholded frame to our object tracking function
        //this function will return the x and y coordinates of the
        //filtered object
        if(trackObjects)
            trackFilteredObject(x,y,threshold,videoFeed);

        X = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2))/(H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2));
        Y = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2))/(H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2));

        cout << "X: " << X << endl << "Y: " << Y << endl;
        //cout << "H: " << endl << H << endl << "(0,0): " << H.at<double>(0,2) << endl;


        //Vectores de movimiento
        P.push_back(Point2f(x,y));
        V.push_back(getVelocity(P, x0, y0));
        A.push_back(getAcceleration(V, vx0, vy0));

        //cout << "Posición del objeto es: " << P.back() << endl;
        //cout << "La velocidad es: " << V.back() << endl;
        //cout << "La aceleración es: " << A.back() << endl;

        //Frecuencia para movimento pendular
        if(mov == 4){
            if((direction == 1) && (V.back().x < 0)){
                direction = -1; //vel negativa
            }else if((direction == -1) && (V.back().x > 0)){
                direction = 1; //vel positiva
                hz++;
            }
        }

        //while(1){
            //show frames
        imshow("Threshold",threshold);
        imshow("Raw video",videoFeed);
            //imshow("HSV",HSV);
        //    if( (waitKey(10)) != -1)
         //       break;
        //}

        //Valores guardados para siguiente iteracion
        x0 = (float)x;
        y0 = (float)y;
        vx0 = V.back().x;
        vy0 = V.back().y;

        waitKey(30);
    }

    if(mov == 0 || mov == 1 || mov == 2){
        //A[0] y A[1] contienen valores basura
        cout << "La magnitud de la fuerza inicial aplicada fue de " << sqrt(pow(masa*A[2].x,2)+pow(masa*A[2].y,2)) << endl;

        //La aceleración debería ser constante una vez que se aplica la fuerza si es que fuera
        //un objeto real. Solo aplicable a movimiento lineal por el momento.
        cout << "El coeficiente de roce es " << (sqrt(pow(A[4].x,2)+pow(A[4].y,2)))/gravedad << endl;
    }

    if(mov == 4)
        cout << "Dio " << hz << " vueltas " << endl;

    return 0;

}
