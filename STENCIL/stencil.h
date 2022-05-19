enum ImageType {HD, UHD};
#define HDROWS 1920
#define HDCOLS 1080
#define UHDROWS 3840
#define UHDCOLS 2160

void Stencil(float *X, enum ImageType typ, int k, float *S, float *Y);
