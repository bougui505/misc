#include <stdlib.h>
#include <math.h>

#define ARRAYSIZE(a) (sizeof(a) / sizeof(a[0]))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

double sep_weight(double sep){if(sep <= 4){return 0.50;}else if(sep == 5){return 0.75;}else{return 1.00;}}
double gaussian(double mean, double stdev, double x){return exp(-pow((x - mean),2)/(2*(pow(stdev,2))));}

double * initialize_matrix(int na, int nb, double cmap_a[na][na], double cmap_b[nb][nb], double sep_x, double sep_y){
    double * outmat = (double *)malloc(sizeof(double) * na * nb);
    int ptr = 0;
    for (int ai=0; ai< na; ai++){
        for (int bi=0; bi< nb; bi++){
            for (int aj=0; aj< na; aj++){
                for (int bj=0; bj< nb; bj++){
                    ptr++;
                    int sa = ai - aj;
                    int sb = bi - bj;
                    if ((sa>0 && sb>0) || (sa<0 && sb <0)){
                        int s_dif = abs(abs(sa)-abs(sb));
                        int s_min = MIN(abs(sa), abs(sb));
                        double s_std = sep_y * (1+ pow(s_min - 2, sep_x));
                        double w = sep_weight(s_min) * gaussian(0, s_std, s_dif);
                        outmat[ptr] = cmap_a[ai][aj] * cmap_b[bi][bj] * w;
                    }
                    else {
                        outmat[ptr] = -1;
                    }
                }
            }
        }
    }
    return outmat;
}
