#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <string>

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
const int MAX_NUM_OBJECTS=20;

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

int main(){

    bool trackObjects = true;
    bool useMorphOps = true;
    Mat HSV;
    Mat threshold;
    Mat videoFeed;

    //parámetro debe ser entregado por el usuario a través de la interfaz, este es en Kg.
    float masa = 10.0;
    float gravedad = 9.81;

    ofstream myfile;
    myfile.open("velocity.txt");

    int mov = 0;
    /*
     * 0 -> Movimiento Lineal
     * 1 -> Plano Inclinado
     * 2 -> Lanzamiento proyectiles
     * 3 -> Movimiento Circular
     * 4 -> Movimiento Pendular
     */

    //x and y values for the location of the object
    int x=0, y=0;

    //create slider bars for HSV filtering
    //Descomentar para calibracion
    //createTrackbars();

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

    //vid.open(0);

    float x0 = 0.0, y0 = 0.0;
    float vx0 = 0.0, vy0 = 0.0;
    vector<Point2f> V;
    vector<Point2f> A;

    //start an infinite loop where webcam feed is copied to videoFeed matrix
    //all of our operations will be performed within this loop
    while(1){
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
        if(trackObjects)
            trackFilteredObject(x,y,threshold,videoFeed);

        //Obtener posicion del objeto
        cout << "Posición del objeto es (" << x << ","<< y << ")" << endl;

        V.push_back(Point2f(abs(x0 - x)*30, abs(y0 - y)*30));

        //cout << "La velocidad es: (" << V.back().x << ", " << V.back().y << ")" << endl;

        A.push_back(Point2f((vx0 - V.back().x)*30, (vy0 - V.back().y)*30));
        //cout << "La aceleración es: (" << A.back().x << ", " << A.back().y << ")" << endl;

        //cout << "Velocidad del objeto es (" << abs(x0 - x)/30 << ","<< abs(y0 - y)/30 << ")" << endl;
        //myfile << "(" << abs(x0 - x)/30 << ","<< abs(y0 - y)/30 << ")\n";

        //cout << "Aceleración del objeto es (" << abs(x0 - x)/30 << ","<< abs(y0 - y)/30 << ")" << endl;


        //while(1){
            //show frames
        imshow("Threshold",threshold);
        imshow("Raw video",videoFeed);
            //imshow("HSV",HSV);
          //  if( (waitKey(10)) != -1)
            //    break;
        //}

        x0 = (float)x;
        y0 = (float)y;
        vx0 = V.back().x;
        vy0 = V.back().y;

        //delay 30ms so that screen can refresh.
        //image will not appear without this waitKey() command
        waitKey(30);
    }


    //cout << A[0] << " " << A[1] << " " << A[2] << endl;
    cout << "La magnitud de la fuerza inicial aplicada fue de " << masa*A[2] << endl;

    //La aceleración debería ser constante una vez que se aplica la fuerza si es que fuera
    //un objeto real.
    cout << "El coeficiente de roce es " << (A[4])/gravedad << endl;

    myfile.close();
    return 0;

}
