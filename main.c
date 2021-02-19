#include <stdint.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <stdio.h>

struct __attribute__((__packed__)) BITMAPFILEHEADER  {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct __attribute__((__packed__)) INFOHEADER
{   uint32_t biSize;  //specifies the number of bytes required by the struct
    uint32_t biWidth;  //specifies width in pixels
    uint32_t biHeight;  //species height in pixels
    uint16_t biPlanes; //specifies the number of color planes, must be 1
    uint16_t biBitCount; //specifies the number of bit per pixel
    uint32_t biCompression;//spcifies the type of compression
    uint32_t biSizeImage;  //size of image in bytes
    uint32_t biXPelsPerMeter;  //number of pixels per meter in x axis
    uint32_t biYPelsPerMeter;  //number of pixels per meter in y axis
    uint32_t biClrUsed;  //number of colors used by th ebitmap
    uint32_t biClrImportant;  //number of colors that are important
};

struct __attribute__((__packed__)) DATA {
    uint16_t * A;
    uint16_t * R;
    uint16_t * G;
    uint16_t * B;
    uint16_t * OneMinusA;
    int size;
};

struct DATA ReadImage(char* filename, int x) {
    struct DATA res;
    FILE* filePtr = fopen(filename,"rb");
    
    struct BITMAPFILEHEADER info1;
    fread(&info1, sizeof(struct BITMAPFILEHEADER),1,filePtr);
    
    struct INFOHEADER info2;
    fread(&info2, sizeof(struct INFOHEADER),1,filePtr);
    fseek(filePtr, 0, 0);
    
    for (int i = 0; i < info1.bfOffBits; ++i) {
        uint8_t c;
        fread(&c, 1, 1, filePtr);
    }

    int size = info2.biSizeImage/4;
    int actualSize = ((size + 3)/4)* 4;
    res.size = size;
    uint16_t* Gdata =(uint16_t*) calloc(actualSize, 2);
    uint16_t* Rdata =(uint16_t*) calloc(actualSize, 2);
    uint16_t* Bdata =(uint16_t*) calloc(actualSize, 2);
    uint16_t* Adata = (uint16_t*) calloc(actualSize, 2);
    uint16_t* A1data = (uint16_t*) calloc(actualSize, 2);
    int count = 0;
    
    for (size_t i = 0; i < size; ++i) {
        uint8_t cur;
        fread(&cur, 1,1,filePtr);
        
        if (cur != 0 && x > 0) {
            count++;
        }
        
        Adata[i] = (uint16_t)cur;
        A1data[i] = 255 - Adata[i];

        fread(&cur, 1,1,filePtr);
        Rdata[i] = (uint16_t)cur;

        fread(&cur, 1,1,filePtr);
        Gdata[i] = (uint16_t)cur;

        fread(&cur, 1,1,filePtr);
        Bdata[i] = (uint16_t)cur;
    }
    
    res.A = Adata;
    res.B = Bdata;
    res.R = Rdata;
    res.G = Gdata;
    res.OneMinusA = A1data;
    
    fclose(filePtr);
    return res;
}

__m128i DivideBy255(__m128i x) {
    uint16_t a[8];
    _mm_store_si128((__m128i*)a, x);
    
    for(int i = 0; i < 8; ++i) {
        a[i] /= 255;
    }
    
    return _mm_load_si128((__m128i*)a);
}

void countSum(uint16_t* first,uint16_t * second, uint16_t* alpha, uint16_t*minusAlpha, uint16_t* res, int actSize) {//counts (second - first) * alpha + first
    for (int i = 0; i + 7 < actSize; i += 8) {
            __m128i s = _mm_slli_epi16(_mm_loadu_si128((__m128i*) &second[i]), 8);
            __m128i a = _mm_slli_epi16(_mm_loadu_si128((__m128i*) &alpha[i]), 8);
            __m128i f = _mm_slli_epi16(_mm_loadu_si128((__m128i*) &first[i]), 8);
            __m128i mA = _mm_slli_epi16(_mm_loadu_si128((__m128i*) &minusAlpha[i]), 8);
            __m128i x1 = _mm_mulhi_epu16(f,  mA);
            __m128i x2 = _mm_mulhi_epu16(s, a);
            __m128i result = DivideBy255(_mm_add_epi16(x1, x2));
            _mm_storeu_si128((__m128i*) &res[i],result);
        }
}

void Overlay(char* f1, char* f2, char* res) {
    FILE *resPtr = fopen(res, "wb");
    struct DATA im1 = ReadImage(f1, 0);
    struct DATA im2 = ReadImage(f2, 1);
    int size = im1.size;
    int actSize = ((im1.size + 3)/4)*4;

    countSum(im1.R, im2.R, im2.A, im2.OneMinusA, im1.R, actSize);
    countSum(im1.G, im2.G, im2.A, im2.OneMinusA, im1.G, actSize);
    countSum(im1.B, im2.B, im2.A, im2.OneMinusA, im1.B, actSize);

    FILE *filePtr = fopen(f1, "rb");
    struct BITMAPFILEHEADER info1;
    fread(&info1, sizeof(struct BITMAPFILEHEADER), 1, filePtr);
    struct INFOHEADER info2;
    fread(&info2, sizeof(struct INFOHEADER), 1, filePtr);
    fseek(filePtr, 0, 0);

    for (int i = 0; i < info1.bfOffBits; ++i) {
        uint8_t c;
        fread(&c, 1, 1, filePtr);
        fwrite(&c, 1, 1, resPtr);
    }


    int actualSize = ((size + 3) / 4) * 4;
    fclose(filePtr);

        for (int i = 0; i < size; ++i) {
            uint8_t x = (uint8_t) im1.A[i];
            uint8_t RValue = (uint8_t) im1.R[i];
            uint8_t GValue = (uint8_t) im1.G[i];
            uint8_t BValue = (uint8_t) im1.B[i];
            fwrite(&x, sizeof(uint8_t), 1, resPtr);
            fwrite(&RValue, sizeof(uint8_t), 1, resPtr);
            fwrite(&GValue, sizeof(uint8_t), 1, resPtr);
            fwrite(&BValue, sizeof(uint8_t), 1, resPtr);
        }

    fclose(resPtr);
    free(im1.A);
    free(im1.B);
    free(im1.R);
    free(im1.G);
    free(im1.OneMinusA);
    free(im2.OneMinusA);
    free(im2.A);
    free(im2.B);
    free(im2.R);
    free(im2.G);
}

int main(int argc, char *argv[]) {
    Overlay(argv[1],
            argv[2],
            argv[3]);
    return 0;
}
