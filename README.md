# Description

This is an implementation of a single layer direct image alignment with g2o graph optimization library. Direct image alignment is a technique to adjust the given relative pose estimation by minimizing photometric cost function. The main assumption, here, is the photometric consistency assumption, which means that intensity value of a pixel seen at the current image is same as the one seen at the previous image. Actually, this is a strong assumption, since any change in intensity, lighting and exposure time may easily affect the performance of the algorithm. Nevertheless, the algorithm is relatively faster compared to the optical flow or feature-based methods. In this project, I demonstrated how one can turn the non-linear optimization problem  into a graph optimization problem and solve it with well-known pose graph optimization library g2o.    

### Dependencies

 - Opencv
 - Eigen
 - g2o
 - Sophus


