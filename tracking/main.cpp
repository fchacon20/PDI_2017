#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <string>
#include <math.h>
#include <fstream>

using namespace cv;
using namespace std;

//Parámetros para aislar el objeto según sus canales HSV
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

//Tamaño del frame de captura
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

//Numero máximo de objetos que puede detectar
const int MAX_NUM_OBJECTS= 100;

//Area mínima y máxima del objeto
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;

void on_trackbar( int, void* )
{//Esta función es llamada cuando la
    // la posición de un trackbar es cambiada
}

string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void createTrackbars(){
    //Crear ventana para los trackbars
    namedWindow(trackbarWindowName,0);
    createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );
}

void drawObject(int x, int y,Mat &frame){

    //Dibujar cruces en el objeto a seguir

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

    //Crear elementos estructurantes que serán usados para dilatar y erosionar la imagen
    Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
    //Se dilata con un elemento más grande para que el objeto sea más visible
    Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));

    erode(thresh,thresh,erodeElement);
    erode(thresh,thresh,erodeElement);

    dilate(thresh,thresh,dilateElement);
    dilate(thresh,thresh,dilateElement);

}

void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed){

    Mat temp;
    threshold.copyTo(temp);

    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );

    //Se usan Moments para obtener el objeto filtrado
    double refArea = 0;
    bool objectFound = false;
    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        //Si el número de objetos detectado es demasiado, hay demasiado ruido con el filtro elegido
        if(numObjects<MAX_NUM_OBJECTS){
            for (int index = 0; index >= 0; index = hierarchy[index][0]) {

                Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;

                if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea){
                    x = moment.m10/area;
                    y = moment.m01/area;
                    objectFound = true;
                    refArea = area;
                }else objectFound = false;
            }

            if(objectFound ==true){
                putText(cameraFeed,"Siguiendo un objeto",Point(0,50),2,1,Scalar(0,255,0),2);
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

float gravityValue(vector<Point2f> V){
    //FPS: 30
    //Cantidad de frames: 10-5+1 = 6
    return ((V[10].y - V[5].y)*30*6);
}

//Esta función encuentra los cuatro puntos iniciales para saber la distancia real
bool calibration(Mat threshold, vector<Point2f> &pos){
    Mat temp;
    threshold.copyTo(temp);

    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );

    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        if(numObjects == 4){
            for (int index = 0; index >= 0; index = hierarchy[index][0]) {
                Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;
                pos.push_back(Point2f(moment.m10/area, moment.m01/area));
            }
            return true;
        }
    }
    return false;
}


int main(){

    bool trackObjects = true;
    bool useMorphOps = true;
    Mat HSV;
    Mat threshold;
    Mat videoFeed;
    Mat H;

    ofstream file;
    file.open("data.csv");
    file << "x1,x2,v1,v2,a1,a2\n";

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
     * Otro valor -> Video
     */

    //x e y son los valores en pixeles
    //X e Y son los valores reales
    int x=0, y=0;
    float X=0, Y=0;

    //La siguiente función se usa para filtrar la imagen y aislar el objeto de interés
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

    //Distancias reales entre puntos de calibración
    dst.push_back(Point2f(0,0));
    dst.push_back(Point2f(0,10));
    dst.push_back(Point2f(10,0));
    dst.push_back(Point2f(10,10));

    //Loop infinito donde se hace seguimiento del objeto de interés
    while(1){

        //Calibración inicial
        while(!isCalibrated){
            vid >> videoFeed;

            if (videoFeed.empty())
                    break;

            cvtColor(videoFeed,HSV,COLOR_BGR2HSV);

            //Parámetros para los 4 puntos rojos de la calibración
            //Es posible cambiar el color, per se debe encontrar nuevamente el filtro
            H_MIN = 139;
            H_MAX = 256;
            S_MIN = 103;
            S_MAX = 256;
            V_MIN = 1;
            V_MAX = 256;

            inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);

            //Operaciones morfológicas para enfatizar el objeto de interés
            if(useMorphOps)
                morphOps(threshold);

            isCalibrated = calibration(threshold, pos);

            //Presionar alguna tecla hasta encontrar los puntos de calibración
            while(1){
                imshow("Threshold",threshold);
                imshow("Raw video",videoFeed);
                if( (waitKey(10)) != -1)
                    break;
            }

            //Orden de los puntos de calibración según el input que se le entregará
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
                H = findHomography(pos, dst);
            }
        }

        vid >> videoFeed;

        if (videoFeed.empty())
                break;

        cvtColor(videoFeed,HSV,COLOR_BGR2HSV);

        //Negro
        /*H_MIN = 81;
        H_MAX = 190;
        S_MIN = 0;
        S_MAX = 78;
        V_MIN = 33;
        V_MAX = 113;*/

        //Verde
        H_MIN = 24;
        H_MAX = 63;
        S_MIN = 68;
        S_MAX = 138;
        V_MIN = 45;
        V_MAX = 242;

        //Rojo
        /*H_MIN = 139;
        H_MAX = 256;
        S_MIN = 103;
        S_MAX = 256;
        V_MIN = 1;
        V_MAX = 256;*/

        inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);

        if(useMorphOps)
            morphOps(threshold);

        //Esta función obtiene los valores de x e y
        if(trackObjects)
            trackFilteredObject(x,y,threshold,videoFeed);

        //Transformación a valores reales
        X = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2))/(H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2));
        Y = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2))/(H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2));

        //Vectores de movimiento
        P.push_back(Point2f(X,Y));
        V.push_back(getVelocity(P, x0, y0));
        A.push_back(getAcceleration(V, vx0, vy0));

        //Datos guardados en archivo para su procesamiento
        file << P.back().x << "," << P.back().y << ","
             << V.back().x << "," << V.back().x << ","
             << A.back().x << "," << A.back().y << "\n";

        //Frecuencia para movimento pendular
        if(mov == 4){
            if((direction == 1) && (V.back().x < 0)){
                direction = -1; //vel negativa
            }else if((direction == -1) && (V.back().x > 0)){
                direction = 1; //vel positiva
                hz++;
            }
        }

        imshow("Threshold",threshold);
        imshow("Raw video",videoFeed);

        //Valores guardados para siguiente iteracion
        x0 = X;
        y0 = Y;
        vx0 = V.back().x;
        vy0 = V.back().y;

        waitKey(30);
    }

    float ang = 0;

    //Obtener ángulo de inclinación
    if(mov == 1 || mov == 2){
        //La elección del frame 20 es arbitraria
        ang = atan(abs(P[0].y-P[20].y)/abs(P[0].x-P[20].x));
        cout << "El ángulo formado es " << ang << endl;
    }

    //Comprobar valor de la aceleración de gravedad
    if (mov == 2){
        cout << "La estimación de la aceleración de gravedad es: " << gravityValue(V) << endl;
    }

    if(mov == 0 || mov == 1 || mov == 2){
        //A[0] y A[1] contienen valores basura
        cout << "La magnitud de la fuerza inicial aplicada fue de " << sqrt(pow(masa*A[2].x,2)+pow(masa*A[2].y,2)) << endl;

        if(mov == 0 || mov == 1){
            //La aceleración debería ser constante una vez que se aplica la fuerza si es que fuera
            //un objeto real. Si es movimiento lineal, el ángulo será 0.
            cout << "El coeficiente de roce es " <<
                    ((sqrt(pow(A[4].x,2)+pow(A[4].y,2))/gravedad)+sin(ang))/cos(ang) << endl;
        }
    }

    if(mov == 4)
        cout << "Dio " << hz << " vueltas " << endl;

    file.close();
    return 0;

}
