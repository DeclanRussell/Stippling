#include "MainWindow.h"
#include <QFileDialog>
#include <QPushButton>
#include <QDesktopServices>
#include <QGLFormat>
#include <QGroupBox>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent){
    QGroupBox *gb = new QGroupBox(this);
    setCentralWidget(gb);
    m_gridLayout = new QGridLayout(gb);
    gb->setLayout(m_gridLayout);

    QGLFormat format;
    format.setVersion(4,1);
    format.setProfile(QGLFormat::CoreProfile);

    //add our openGL context to our scene
    m_openGLWidget = new NGLScene(format,this);
    m_gridLayout->addWidget(m_openGLWidget,0,0,4,4);
    m_openGLWidget->show();

    // Group box to hold our general controls
    QGroupBox *genGb = new QGroupBox("General Controls",gb);
    m_gridLayout->addWidget(genGb,4,0,1,4);
    // Grid layout for this groupbox
    QGridLayout *genGbLyt = new QGridLayout(genGb);
    genGb->setLayout(genGbLyt);

    // Add some controls to interface with the application
    // Toggle play button
    QPushButton *tglPlayBtn = new QPushButton("Play/Pause",genGb);
    genGbLyt->addWidget(tglPlayBtn,1,0,1,1);
    connect(tglPlayBtn,SIGNAL(pressed()),m_openGLWidget,SLOT(tglUpdate()));

    // Reset button
    QPushButton *rstBtn = new QPushButton("Add new active particles",genGb);
    genGbLyt->addWidget(rstBtn,2,0,1,1);
    connect(rstBtn,SIGNAL(pressed()),m_openGLWidget,SLOT(resetSim()));

    // Save samples button
    QPushButton *saveBtn = new QPushButton("Save samples to file",genGb);
    genGbLyt->addWidget(saveBtn,3,0,1,1);
    connect(saveBtn,SIGNAL(pressed()),this,SLOT(save()));

    // Select sample image button
    QPushButton *selSampImgBtn = new QPushButton("Select sample image",genGb);
    genGbLyt->addWidget(selSampImgBtn,5,0,1,1);
    connect(selSampImgBtn,SIGNAL(pressed()),this,SLOT(setSampleImage()));

}

void MainWindow::save()
{
    QString dir = QFileDialog::getSaveFileName(this,"Save Samples to File");
    if(!dir.isEmpty())
    {
        m_openGLWidget->exportSamplesToFile(dir);
    }
}

void MainWindow::setSampleImage()
{
    QString dir = QFileDialog::getOpenFileName(this,"Select Samples Iamge");
    if(!dir.isEmpty())
    {
        m_openGLWidget->setSampleImage(dir);
    }
}
