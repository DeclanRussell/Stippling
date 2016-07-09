#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QGroupBox>
#include <QGridLayout>
#include <QSpacerItem>
#include <iostream>
#include <QGridLayout>
#include <QProgressBar>
#include <QDoubleSpinBox>

#include "NGLScene.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow(){}

public slots:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to save our samples to a file
    //----------------------------------------------------------------------------------------------------------------------
    void save();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the image for us to ray trace
    //----------------------------------------------------------------------------------------------------------------------
    void setSampleImage();

private:
    NGLScene *m_openGLWidget;
    QGridLayout *m_gridLayout;




};

#endif // MAINWINDOW_H
