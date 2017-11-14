#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCamera>
#include <QMediaRecorder>
#include <QMultimedia>

#include <opencv2/opencv.hpp>
#include <QTimer>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    void initCamera();
    void initVideo();
    void processGui();

    cv::VideoWriter writer;
    cv::VideoCapture cap;
    cv::Mat frame;

    QTimer *trigger;
    bool isRecorded;

private slots:
    void on_start_clicked();
    void on_stop_clicked();
    void on_record_clicked();
    void on_toolButton_clicked();

};

#endif // MAINWINDOW_H
