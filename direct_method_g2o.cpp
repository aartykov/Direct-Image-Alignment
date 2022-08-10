#include <iostream>

#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include<opencv2/core/persistence.hpp>
//#include <opencv2/video/tracking.hpp>
//#include<chrono>

#include<Eigen/Core>
#include<Eigen/Dense>
//#include<opencv2/core/eigen.hpp>
#include <boost/format.hpp>

#define SOPHUS_USE_BASIC_LOGGING
#include"sophus/se3.hpp"

#include<g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>



using namespace std;
typedef vector<Eigen::Vector2d> VecVector;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;


// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}


// G2O Optimization
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>
{   
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override
        {
            _estimate = Sophus::SE3d();
        }

        // left multiplication on SE3
        virtual void oplusImpl(const double* update) override
        {
            Eigen::Matrix<double, 6, 1> update_eigen;
            update_eigen << update[0], update[1], update[2], update[3],
                            update[4], update[5];
            
            _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
        }

        virtual bool read(istream& in) override {}
        virtual bool write(ostream& out) const override {}
 
};


class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeProjection(cv::Mat& img1, cv::Mat& img2, Eigen::Matrix3d& K, double depth_ref):
                       img1_(img1), img2_(img2), K_(K), depth_ref_(depth_ref) { }


        virtual void computeError() override
        {   
            double fx = K_(0,0), fy = K_(1,1), cx = K_(0,2), cy = K_(1,2);
            const VertexPose* V = static_cast<VertexPose*>(_vertices[0]);

            Eigen::Vector2d pix_ref_ = _measurement;    

            Eigen::Vector3d p_ref;
            p_ref << (pix_ref_[0] - cx)/fx, (pix_ref_[1] - cy)/fy, 1.0;
            p_ref *= depth_ref_;
            Sophus::SE3d T21 = V->estimate();
            Eigen::Vector3d p_curr = T21 * p_ref;
            p_curr /= p_curr[2];
            float u = p_curr[0]*fx + cx;
            float v = p_curr[1]*fy + cy;

            int half_patch_size = 1;
            if(p_curr[2] < 0 || u<half_patch_size || u>img2_.cols-half_patch_size || v<
            half_patch_size || v>img2_.rows-half_patch_size) // depth invalid
                status_ = false;
            else
                status_ = true;
            
            if(status_) // BU HATALI OLABILIR !!!!
                _error(0,0) = GetPixelValue(img1_, pix_ref_[0], pix_ref_[1]) - GetPixelValue(img2_, u, v);
        }


        // define analytical jacobian funcrion Ji
        virtual void linearizeOplus() override
        {
            const VertexPose* V = static_cast<VertexPose*>(_vertices[0]);
            double fx = K_(0,0), fy = K_(1,1), cx = K_(0,2), cy = K_(1,2);

            Eigen::Vector2d pix_ref_ = _measurement;

            // backproject pixel
            Eigen::Vector3d p_ref((pix_ref_[0]-cx)/fx, (pix_ref_[1]-cy)/fy, 1.0);
            p_ref *= depth_ref_;
            Sophus::SE3d T = V->estimate();
            Eigen::Vector3d p_curr = T * p_ref;

            // project the point 
            float u=fx*p_curr[0]/p_curr[2] + cx, v=fy*p_curr[1]/p_curr[2]+cy;

            int half_patch_size = 1;
            if(p_curr[2] < 0 || u<half_patch_size || u>img2_.cols-half_patch_size || v<
            half_patch_size || v>img2_.rows-half_patch_size) // invalid depth
                status_ = false;
            else
                status_ = true;

            if(status_) 
            {   
                double X = p_curr[0], Y = p_curr[1], Z = p_curr[2], Z2 = Z*Z, Z_inv=1.0/Z,
                Z2_inv = Z_inv * Z_inv;

                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0,0) = fx * Z_inv; 
                J_pixel_xi(0,1) = 0.0;
                J_pixel_xi(0,2) = -fx * X * Z2_inv;
                J_pixel_xi(0,3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0,4) = fx + fx*X*X*Z2_inv;
                J_pixel_xi(0,5) = -fx*Y*Z_inv;

                J_pixel_xi(1,0) = 0.0;
                J_pixel_xi(1,1) = fy * Z_inv;
                J_pixel_xi(1,2) = -fy * Y * Z2_inv;
                J_pixel_xi(1,3) = -fy  - fy*Y*Y*Z2_inv;
                J_pixel_xi(1,4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1,5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2_, u+1, v) - GetPixelValue(img2_, u
                    -1, v)),
                    0.5 * (GetPixelValue(img2_, u, v+1) - GetPixelValue(img2_, u
                    , v-1))
                );

            _jacobianOplusXi.block<1,6>(0,0) = -1.0 * (J_img_pixel.transpose() * J_pixel_xi);
            
            }
        
        }

        virtual bool read(istream& in) override { }
        virtual bool write(ostream& out) const override { }
    
    private:
        cv::Mat& img1_;
        cv::Mat& img2_;
        Eigen::Matrix3d& K_;
        double depth_ref_;
        bool status_ = true;

};


void MotionOnlyBAG2O(vector<Eigen::Vector2d>&, cv::Mat&, cv::Mat&, vector<double>&, Sophus::SE3d&);

int main(int argc, char** argv)
{
    // Camera intrinsics
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // baseline
    double baseline = 0.573;
    // paths
    string left_file = "../left.png";
    string disparity_file = "../disparity.png";

    boost::format fmt_others("./%06d.png");    // other files

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    vector<Eigen::Vector2d> pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for(int i=0; i<nPoints; i++)
    {
        int x = rng.uniform(boarder, left_img.cols-boarder); // don't pick pixels closer to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity;
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x,y));
    }

    // estimates 01~05.png's pose using  this information
    Sophus::SE3d T_cur_ref;
    for(int i=1; i<6; i++)
    {
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        MotionOnlyBAG2O(pixels_ref, left_img, img, depth_ref, T_cur_ref);
    }





    return 0;
}


void MotionOnlyBAG2O(vector<Eigen::Vector2d>& ref_points2d, cv::Mat& img1, cv::Mat& img2, vector<double>& ref_depth, Sophus::SE3d& pose)
{
    
    // First, let's define g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> BlockSolverType;

    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

   
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true); 

    // Vertex
    VertexPose* vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(pose); // set initial pose
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;

    // Edges
    int index = 1;
    for(size_t i=0; i<ref_points2d.size(); i++)
    {
        auto p2d = ref_points2d[i];
        auto depth = ref_depth[i];
        EdgeProjection* edge = new EdgeProjection(img1, img2, K_eigen, depth);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }


    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    cout << "pose etimated by g2o = \n" << vertex_pose->estimate().matrix() << endl;
    
    pose = vertex_pose->estimate();

}

