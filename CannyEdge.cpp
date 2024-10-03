#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//void Conv(Mat M,)

void disp(Mat image,string name){
    namedWindow(name, WINDOW_AUTOSIZE );
    imshow(name, image);
    waitKey(0);
}

int main(int argc,char** argv){
    if (argc!=2){
        cout<<"usage: ./CannyEdge <Image_Path>"<<endl;
    }

    Mat image;
    image = imread( argv[1], IMREAD_COLOR );
    Mat original=image;
 
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    disp(image,"Original");


    //Greyscale, Apply Gaussian filter (5x5)
    cvtColor(image, image, cv::COLOR_BGR2GRAY);
    Mat Gaussian=getGaussianKernel(5,1.4);
    sepFilter2D(image,image,-1,Gaussian,Gaussian);
    
    disp(image,"Greyscale, Gaussian filter");
    
    //Edge Detection: Sobel operator

    Mat Sobelx,Sobely;
    Sobel(image,Sobelx,-1,1,0); 
    Sobel(image,Sobely,-1,0,1);    

    Mat tot=Sobelx.mul(Sobelx)+Sobely.mul(Sobely);
    disp(tot,"Sobel Operator");

    tot.convertTo(tot,CV_32F);
    Mat sqt;
    pow(tot,0.5,tot);

    
    //Calculate grad direction
    //dirmap: {0:e-w,1:ne-sw,2:n-s,3:nw-se}
    vector<vector<double>> dir(tot.rows,vector<double>(tot.cols,0));
    vector<vector<int>> dirmap(tot.rows,vector<int>(tot.cols,0));
    for (int i=0;i<tot.rows;i++){
        const double* rowx=Sobelx.ptr<double>(i);
        const double* rowy=Sobely.ptr<double>(i);
        for (int j=0;j<tot.cols;j++){
            dir[i][j]=atan(2*rowx[j]/rowy[j]);
            dirmap[i][j]=((int)floor(dir[i][j]+180.0/8.0)%180)/45;

        }
    }

    //Minimum cut-off suppression
    vector<vector<int>> kept(tot.rows,vector<int>(tot.cols,0));
    for (int i=0;i<tot.rows;i++){
        const double* row_n1=tot.ptr<double>(i);
        if (i>0) row_n1=tot.ptr<double>(i-1);
        const double* row_0=tot.ptr<double>(i);
        const double* row_p1=row_0;
        if (i<tot.rows-1) row_p1=tot.ptr<double>(i+1);

        for (int j=0;j<tot.cols;j++){

            double chk=row_0[j],cmp1=chk,cmp2=chk;

            switch (dirmap[i][j]){
                case 0:
                    if (j>0) cmp1=row_0[j-1];
                    if (j<(tot.cols-1)) cmp2=row_0[j+1];
                break;

                case 1:
                    if (i>0 && j<(tot.cols-1)) cmp1=row_n1[j+1];
                    if (i<(tot.rows-1) && j>0) cmp1=row_p1[j-1];
                break;

                case 2:
                    if (i>0) cmp1=row_n1[j];
                    if (i<(tot.rows-1)) cmp2=row_p1[j];
                break;

                case 3:
                    if (i>0 && j>0) cmp1=row_n1[j-1];
                    if (i<(tot.rows-1) && j<(tot.cols-1)) cmp1=row_p1[j+1];
                break;
            }

            if (chk<cmp1 || chk>=cmp2) kept[i][j]++;

        }
    }
    cout<<"test"<<endl;

    Mat keptm(tot.rows,tot.cols,DataType<int>::type);
    for (int i=0;i<tot.rows;i++){
        for (int j=0;j<tot.rows;j++){
            keptm.at<int>(i,j)=kept[i][j];
        }
        
    }

    
    tot.mul(keptm);
    disp(tot,"Min cut-off direction");
    

    //double threshold
    double maxv,minv;
    Point maxloc,minloc;
    minMaxLoc(tot,&minv,&maxv,&minloc,&maxloc);
    double high_thresh=0.5*maxv,low_thresh=0.5*high_thresh;

    vector<vector<int>> edge(tot.rows,vector<int>(tot.cols,0));
    for (int i=0;i<tot.rows;i++){
        const double* row_0=tot.ptr<double>(i);
        for (int j=0;j<tot.rows;j++){
            if (row_0[j]>high_thresh) edge[i][j]=2;
            else if (row_0[j]>low_thresh) edge[i][j]=1;
        }
        
    }

    //Edge tracking by hysterics
    vector<vector<int>> edgecopy=edge;
    for (int i=0;i<tot.rows;i++){
        for (int j=0;j<tot.rows;j++){
            if (edge[i][j]==1){
                if ((i>0 && j>0 && edge[i-1][j-1]==2) || (i>0 && edge[i-1][j]==2) || (i>0 && j<tot.cols-1 && edge[i-1][j+1]==2) ||
                ( j>0 && edge[i][j-1]==2) ||  ( j<tot.cols-1 && edge[i-1][j+1]==2) ||
                (i<(tot.rows-1) && j>0 && edge[i-1][j-1]==2) || (i<(tot.rows-1) && edge[i-1][j]==2) || (i<(tot.rows-1)&& j<tot.cols-1 && edge[i-1][j+1]==2) )
                edgecopy[i][j]=2;
            }
        }
    }

    for (int i=0;i<tot.rows;i++){
        for (int j=0;j<tot.rows;j++){
            keptm.at<int>(i,j)=(edgecopy[i][j]==2?1:0);
        }
        
    }
    tot.mul(keptm);
    disp(tot,"Double threshold with edge checking hysterics");


 
    return 0;

}