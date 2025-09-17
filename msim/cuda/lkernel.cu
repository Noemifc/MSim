// volops_templated.cu
// CUDA volume operations with templated type support

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// Host+Device helper: build 3×3 rotation matrix from quaternion (w,x,y,z)
//------------------------------------------------------------------------------
__host__ __device__
void quaternion_to_matrix(
    double w, double x, double y, double z,
    double R[9]
) {
    double n = sqrt(w*w + x*x + y*y + z*z);
    w /= n; x /= n; y /= n; z /= n;
    R[0] = 1 - 2*(y*y + z*z);
    R[1] =     2*(x*y - z*w);
    R[2] =     2*(x*z + y*w);
    R[3] =     2*(x*y + z*w);
    R[4] = 1 - 2*(x*x + z*z);
    R[5] =     2*(y*z - x*w);
    R[6] =     2*(x*z - y*w);
    R[7] =     2*(y*z + x*w);
    R[8] = 1 - 2*(x*x + y*y);
}

//------------------------------------------------------------------------------
// Template: Rotate 3D volume by quaternion about its center (nearest-neighbor)
//------------------------------------------------------------------------------
template <typename T>
__global__ void rotate_volume_kernel(
    const T* __restrict__ in_vol,
    T*       __restrict__ out_vol,
    int nx, int ny, int nz,
    double qw, double qx, double qy, double qz
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    __shared__ double R[9];
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        quaternion_to_matrix(qw, qx, qy, qz, R);
    }
    __syncthreads();

    double cx = 0.5 * (nx - 1);
    double cy = 0.5 * (ny - 1);
    double cz = 0.5 * (nz - 1);
    double x = ix - cx, y = iy - cy, z = iz - cz;

    double xs = R[0]*x + R[3]*y + R[6]*z + cx;
    double ys = R[1]*x + R[4]*y + R[7]*z + cy;
    double zs = R[2]*x + R[5]*y + R[8]*z + cz;

    int rx = int(round(xs));
    int ry = int(round(ys));
    int rz = int(round(zs));

    T val = 0;
    if (rx >= 0 && rx < nx && ry >= 0 && ry < ny && rz >= 0 && rz < nz) {
        val = in_vol[rz * ny * nx + ry * nx + rx];
    }
    out_vol[iz * ny * nx + iy * nx + ix] = val;
}

//------------------------------------------------------------------------------
// Dispatcher: rotate_volume
//------------------------------------------------------------------------------
extern "C"
void rotate_volume(
    const void* in_h,
    void* out_h,
    int nx, int ny, int nz,
    double qw, double qx, double qy, double qz,
    int dtype  // 0=uint8, 1=uint16, 2=float32, 3=float64
) {
    size_t N = size_t(nx) * ny * nz;
    dim3 t(8,8,8), b((nx+7)/8, (ny+7)/8, (nz+7)/8);

    if (dtype == 0) {
        const uint8_t* in_u8 = static_cast<const uint8_t*>(in_h);
        uint8_t* out_u8 = static_cast<uint8_t*>(out_h);
        uint8_t *d_in, *d_out;
        cudaMalloc(&d_in, N);
        cudaMalloc(&d_out, N);
        cudaMemcpy(d_in, in_u8, N, cudaMemcpyHostToDevice);
        rotate_volume_kernel<uint8_t><<<b, t>>>(d_in, d_out, nx, ny, nz, qw, qx, qy, qz);
        cudaMemcpy(out_u8, d_out, N, cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out);
    }
    else if (dtype == 1) {
        const uint16_t* in_u16 = static_cast<const uint16_t*>(in_h);
        uint16_t* out_u16 = static_cast<uint16_t*>(out_h);
        uint16_t *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(uint16_t));
        cudaMalloc(&d_out, N * sizeof(uint16_t));
        cudaMemcpy(d_in, in_u16, N * sizeof(uint16_t), cudaMemcpyHostToDevice);
        rotate_volume_kernel<uint16_t><<<b, t>>>(d_in, d_out, nx, ny, nz, qw, qx, qy, qz);
        cudaMemcpy(out_u16, d_out, N * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out);
    }
    else if (dtype == 2) {
        const float* in_f32 = static_cast<const float*>(in_h);
        float* out_f32 = static_cast<float*>(out_h);
        float *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_out, N * sizeof(float));
        cudaMemcpy(d_in, in_f32, N * sizeof(float), cudaMemcpyHostToDevice);
        rotate_volume_kernel<float><<<b, t>>>(d_in, d_out, nx, ny, nz, qw, qx, qy, qz);
        cudaMemcpy(out_f32, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out);
    }
    else if (dtype == 3) {
        const double* in_f64 = static_cast<const double*>(in_h);
        double* out_f64 = static_cast<double*>(out_h);
        double *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(double));
        cudaMalloc(&d_out, N * sizeof(double));
        cudaMemcpy(d_in, in_f64, N * sizeof(double), cudaMemcpyHostToDevice);
        rotate_volume_kernel<double><<<b, t>>>(d_in, d_out, nx, ny, nz, qw, qx, qy, qz);
        cudaMemcpy(out_f64, d_out, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out);
    }
}
