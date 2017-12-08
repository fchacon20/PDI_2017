#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <string>
#include <QLabel>
#include <QFileDialog>
#include <QLineEdit>
#include <QDebug>
#include <iostream>

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    isRecorded = false;
    ui->lineEdit->setText("video.avi");
    trigger = new QTimer(this);
    connect(trigger, &QTimer::timeout, this, &MainWindow::processGui);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_start_clicked()
{
    initCamera();
    trigger->start(100);
}

void MainWindow::on_stop_clicked()
{
    if(cap.isOpened())
    {
        // Disconnect camera
        cap.release();
        qDebug() << "Camera successfully disconnected.";
    }
    trigger->stop();
}

void MainWindow::on_record_clicked()
{
    isRecorded = !isRecorded;
    if(isRecorded)
        initVideo();
}

void MainWindow::initCamera()
{
    cap.open(0);
    if(cap.isOpened())
    {
        qDebug() << "Camera successfully connected.";
    }
}

void MainWindow::initVideo()
{
    std::string filename =  ui->lineEdit->text().toStdString();
    int fcc =   CV_FOURCC('D','I','V','X');
    int fps =   20;
    cv::Size frameSize(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    writer = cv::VideoWriter(filename,fcc,fps,frameSize);
}

void MainWindow::processGui()
{
    if(cap.read(frame)){
        cv::cvtColor(frame,frame,CV_BGR2RGB);
        if(isRecorded)
        {
            writer.write(frame);
            cv::putText(frame,"[REC]",cv::Point(0,30),5,1,cv::Scalar(0,0,225));
        }

        QImage qframe((uchar*)frame.data,frame.cols,frame.rows,frame.step,QImage::Format_RGB888);

        ui->camera1->setPixmap(QPixmap::fromImage(qframe));
    }
}

void MainWindow::on_toolButton_clicked()
{

    QString filename = QFileDialog::getSaveFileName(0, "Save file", QDir::currentPath(), "Video files (*.avi);;All files (*.*)");

    ui->lineEdit->setText(filename);

    initVideo();
}
