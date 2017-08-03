#include <superpixels.h>

Superpixels::Superpixels(Eigen::MatrixXf im_r, Eigen::MatrixXf im_g, Eigen::MatrixXf im_b, Eigen::MatrixXf xx, Eigen::MatrixXf yy, Eigen::MatrixXf depth, unsigned int superpixelsNo, float m) {

    rows = im_r.rows(); // / 2 / 1.6;
    cols = im_r.cols(); // / 2 / 1.6;

    Eigen::MatrixXf r_resize = Eigen::MatrixXf::Zero(rows, cols);
    Eigen::MatrixXf g_resize = Eigen::MatrixXf::Zero(rows, cols);
    Eigen::MatrixXf b_resize = Eigen::MatrixXf::Zero(rows, cols);

    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++) {
            r_resize(i,j) = im_r(i, j);
            g_resize(i,j) = im_g(i, j);
            b_resize(i,j) = im_b(i, j);
        }
    }



    noOfPixels = rows * cols;
    this->m = m;

    segmentationT0 = std::chrono::high_resolution_clock::now();

    lmat = Eigen::MatrixXd::Zero(rows, cols);
    amat = Eigen::MatrixXd::Zero(rows, cols);
    bmat = Eigen::MatrixXd::Zero(rows, cols);

    xmat = xx;
    ymat = yy;
    zmat = depth;

    this->superpixelsNo = superpixelsNo;

    labelsImage = Eigen::MatrixXi::Ones(rows, cols) * superpixelsNo;
    reachedByQueue = Eigen::MatrixXi::Zero(rows, cols);

    int numklabels = 0;

    rgbToLab(r_resize, g_resize, b_resize);

    runSNIC(cols, rows, &numklabels, superpixelsNo, m);

    segmentationT1 = std::chrono::high_resolution_clock::now();

    segmentationDuration = segmentationT1 - segmentationT0;

        std::cout<<"\n Run-time superpixels seg: "<<segmentationDuration.count()<<"\n";
}


void Superpixels::rgbToLab(Eigen::MatrixXf im_r, Eigen::MatrixXf im_g, Eigen::MatrixXf im_b)
{
    int i; int sR, sG, sB;
    double R,G,B;
    double X,Y,Z;
    double r, g, b;
    const double epsilon = 0.008856;	//actual CIE standard
    const double kappa   = 903.3;		//actual CIE standard

    const double Xr = 0.950456;	//reference white
    const double Yr = 1.0;		//reference white
    const double Zr = 1.088754;	//reference white
    double xr,yr,zr;
    double fx, fy, fz;
    double lval,aval,bval;


    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {

            //sR = im_r(i,j); sG = im_g(i,j); sB = im_b(i,j);
            R = im_r(i,j); ; // /255.0;
            G = im_g(i,j); ; // /255.0;
            B = im_b(i,j); // /255.0;

            if(R <= 0.04045)	r = R/12.92;
            else				r = pow((R+0.055)/1.055,2.4);
            if(G <= 0.04045)	g = G/12.92;
            else				g = pow((G+0.055)/1.055,2.4);
            if(B <= 0.04045)	b = B/12.92;
            else				b = pow((B+0.055)/1.055,2.4);

            X = r*0.4124564 + g*0.3575761 + b*0.1804375;
            Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
            Z = r*0.0193339 + g*0.1191920 + b*0.9503041;

            //------------------------
            // XYZ to LAB conversion
            //------------------------
            xr = X/Xr;
            yr = Y/Yr;
            zr = Z/Zr;

            if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
            else				fx = (kappa*xr + 16.0)/116.0;
            if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
            else				fy = (kappa*yr + 16.0)/116.0;
            if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
            else				fz = (kappa*zr + 16.0)/116.0;

            lval = 116.0*fy-16.0;
            aval = 500.0*(fx-fy);
            bval = 200.0*(fy-fz);

            this->lmat(i, j) = lval; this->amat(i, j) = aval; this->bmat(i, j) = bval;
        }
    }

}

void Superpixels::findSeeds(const int width, const int height, int &numk, vector<int> &kx, vector<int> &ky, vector<float> &kx_metric, vector<float> &ky_metric, vector<float> &kz_metric) {
    const int sz = width*height;
    int gridstep = sqrt(double(sz)/double(numk)) + 0.5;
    int halfstep = gridstep/2;
    double h = height; double w = width;

    int xsteps = int(width/gridstep);
    int ysteps = int(height/gridstep);
    int err1 = abs(xsteps*ysteps - numk);
    int err2 = abs(int(width/(gridstep-1))*int(height/(gridstep-1)) - numk);
    if(err2 < err1)
    {
        gridstep -= 1.0;
        xsteps = width /(gridstep);
        ysteps = height/(gridstep);
    }

    numk = (xsteps*ysteps);
    kx.resize(numk); ky.resize(numk); kx_metric.resize(numk); ky_metric.resize(numk); kz_metric.resize(numk);

    int n = 0;
    for(int y = halfstep, rowstep = 0; y < height && n < numk; y += gridstep, rowstep++)
    {
        for(int x = halfstep; x < width && n < numk; x += gridstep)
        {
            if( y <= h-halfstep && x <= w-halfstep)
            {
                kx[n] = x;
                ky[n] = y;
                kx_metric[n] = xmat(y, x);
                ky_metric[n] = ymat(y, x);
                kz_metric[n] = zmat(y, x);
                n++;
            }
        }
    }
}

void Superpixels::runSNIC(const int width, const int height, int *outnumk, const int innumk, const double compactness) {
    const int w = width;
    const int h = height;
    const int sz = w*h;
    const int dx8[8] = {-1,  0, 1, 0, -1,  1, 1, -1};//for 4 or 8 connectivity
    const int dy8[8] = { 0, -1, 0, 1, -1, -1, 1,  1};//for 4 or 8 connectivity
    const int dn8[8] = {-1, -w, 1, w, -1-w,1-w,1+w,-1+w};

    struct NODE
    {
        unsigned int i; // the x and y values packed into one
        float z_metric;
        float x_metric;
        float y_metric;
        unsigned int k; // the label
        double d;       // the distance
    };
    struct compare
    {
        bool operator()(const NODE& one, const NODE& two)
        {
            return one.d > two.d;//for increasing order of distances
        }
    };
    //-------------
    // Find seeds
    //-------------
    vector<int> cx(0),cy(0); vector<float> cz_metric(0), cx_metric(0), cy_metric(0);
    int numk = innumk;
    findSeeds(width,height,numk,cx,cy,cx_metric, cy_metric, cz_metric);//the function may modify numk from its initial value


    //-------------
    // Initialize
    //-------------
    NODE tempnode;
    priority_queue<NODE, vector<NODE>, compare> pq;
 //   memset(labels,-1,sz*sizeof(int));
    for(int k = 0; k < numk; k++)
    {
        NODE tempnode;
        tempnode.i = cx[k] << 16 | cy[k];
        tempnode.z_metric = cz_metric[k];
        tempnode.x_metric = cx_metric[k];
        tempnode.y_metric = cy_metric[k];

        tempnode.k = k;
        tempnode.d = 0;
        pq.push(tempnode);

        reachedByQueue(cy[k], cx[k]) = 1;
    }
    vector<double> kl(numk, 0), ka(numk, 0), kb(numk, 0);
    vector<double> kx(numk,0),ky(numk,0), kz_metric(numk, 0), kx_metric(numk, 0), ky_metric(numk, 0);
    vector<double> ksize(numk,0);

    const int CONNECTIVITY = 4; //values can be 4 or 8
    const double M = compactness;//10.0;
    const double invwt = (M*M*numk)/double(sz);

    int qlength = pq.size();
    int pixelcount = 0;
    int xx(0),yy(0),ii(0);
    double ldiff(0),adiff(0),bdiff(0),xdiff(0),ydiff(0),zdiff(0),colordist(0),xydist(0),slicdist(0);

    //-------------
    // Run main loop
    //-------------
    while(qlength > 0) //while(nodevec.size() > 0)
    {
        NODE node = pq.top(); pq.pop(); qlength--;
        const int k = node.k;
        const int x = node.i >> 16 & 0xffff;
        const int y = node.i & 0xffff;
        const int i = y*width+x;

        int row = y;
        int col = x;

        if(labelsImage(row, col) == superpixelsNo)
        {
            labelsImage(row, col) = k;
            pixelcount++;

            kl[k] += lmat(row, col);
            ka[k] += amat(row, col);
            kb[k] += bmat(row, col);

            kx[k] += x;
            ky[k] += y;

            kx_metric[k] += xmat(row, col);
            ky_metric[k] += ymat(row, col);
            kz_metric[k] += zmat(row, col);

            ksize[k] += 1.0;

            for(int p = 0; p < CONNECTIVITY; p++)
            {
                xx = x + dx8[p];
                yy = y + dy8[p];
                if(!(xx < 0 || xx >= w || yy < 0 || yy >= h))
                {
                    ii = i + dn8[p];

                    if(/* reachedByQueue(yy, xx) == 0 && */ labelsImage(yy, xx) == superpixelsNo)//create new nodes
                    {

                        ldiff = kl[k] - lmat(yy, xx)*ksize[k];
                        adiff = ka[k] - amat(yy, xx)*ksize[k];
                        bdiff = kb[k] - bmat(yy, xx)*ksize[k];

                        xdiff = kx[k] - xx*ksize[k];   //pixel distances
                        ydiff = ky[k] - yy*ksize[k];

//                        xdiff = (kx_metric[k] - xmat(yy, xx) * ksize[k]) ;
//                        ydiff = (ky_metric[k] - ymat(yy, xx) * ksize[k]) ;
                        zdiff = (kz_metric[k] - zmat(yy, xx) * ksize[k]); //* 100.0;

                      //  std::cout<<xdiff<<" "<<ydiff<<" "<<zdiff<<"\n";


                        colordist   = ldiff*ldiff + adiff*adiff + bdiff*bdiff;
                        xydist      = xdiff*xdiff + ydiff*ydiff; //+ zdiff*zdiff;

                        slicdist    = (colordist + xydist*invwt /*+ zdiff*zdiff*400.0 */)/(ksize[k]*ksize[k]);//late normalization by ksize[k], to have only one division operation

                        tempnode.i = xx << 16 | yy;
                        tempnode.k = k;
                        tempnode.d = slicdist;
                        pq.push(tempnode); qlength++;

                        reachedByQueue(yy, xx) = 1;

                    }
                }
            }
        }

    }
    *outnumk = numk;
    //---------------------------------------------
    // Label the rarely occuring unlabelled pixels
    //---------------------------------------------
//    if(labelsImage(0,0) == superpixelsNo) labelsImage(0,0) = 0;
//    for(int y = 1; y < height; y++)
//    {
//        for(int x = 1; x < width; x++)
//        {
//            int i = y*width+x;
//            if(labelsImage(y, x) == superPixelsNo)//find an adjacent label
//            {
//                if(labelsImage(y, x-1) != superPixelsNo) labelsImage(y, x) = labelsImage(y, x-1);
//                else if(labelsImage(y-1, x) != superPixelsNo) labelsImage(y, x) = labelsImage(y-1, x);
//            }//if labels[i] < 0 ends
//        }
//    }

}



Eigen::MatrixXi Superpixels::getLabelsImage() {
    return labelsImage;
}

//std::vector<Eigen::MatrixXi> Superpixels::getLabelsPyramid(int levels) {

//    std::vector<Eigen::MatrixXi> labelsPyramid;

//    Eigen::MatrixXi level1 = Eigen::MatrixXi::Ones(rows / 2, cols / 2) * this->superpixelsNo;
//    Eigen::MatrixXi level2 = Eigen::MatrixXi::Ones(rows / 4, cols / 4) * this->superpixelsNo;
//    Eigen::MatrixXi level3 = Eigen::MatrixXi::Ones(rows / 8, cols / 8) * this->superpixelsNo;
//    Eigen::MatrixXi level4 = Eigen::MatrixXi::Ones(rows / 16, cols / 16) * this->superpixelsNo;

//    for (int i=0; i<rows; i++) {
//        for (int j=0; j<cols; j++) {
//            level1(i/2, j/2) = labelsImage(i, j);
//            level2(i/4, j/4) = labelsImage(i, j);
//            level3(i/8, j/8) = labelsImage(i, j);
//            level4(i/16, j/16) = labelsImage(i, j);

//        }
//    }

//    labelsPyramid.push_back(labelsImage);
//    labelsPyramid.push_back(level1);
//    labelsPyramid.push_back(level2);
//    labelsPyramid.push_back(level3);
//    labelsPyramid.push_back(level4);

//    return labelsPyramid;
//}

