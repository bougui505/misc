#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

double sep_weight(double sep){if(sep <= 4){return 0.50;}else if(sep == 5){return 0.75;}else{return 1.00;}}
double gaussian(double mean, double stdev, double x){return exp(-pow((x - mean),2)/(2*(pow(stdev,2))));}

void initialize_matrix(int na, int nb, double * cmap_a, double * cmap_b, double sep_x, double sep_y){
    double * M = (double *)malloc(sizeof(double) * na*nb);
    double Mval = 0;
    double contact_a = 0.;
    double contact_b = 0.;
    double s_dif = 0;
    double s_min = 0;
    double s_std = 0;
    double w = 0;
    double sa = 0;
    double sb = 0;
    int aptr = 0;
    int bptr = 0;
    for (int ai=0; ai< na; ai++){
        for (int bi=0; bi< nb; bi++){
            int ptr = 0;
            for (int aj=0; aj< na; aj++){
                for (int bj=0; bj< nb; bj++){
                    ptr++;
                    aptr = ai * na + aj;
                    bptr = bi * nb + bj;
                    contact_a = cmap_a[aptr];
                    contact_b = cmap_b[bptr];
                    if (contact_a != 0. && contact_b != 0.){
                        sa = ai - aj;
                        sb = bi - bj;
                        if ((sa>0 && sb>0) || (sa<0 && sb <0)){
                            s_dif = abs(abs(sa)-abs(sb));
                            s_min = MIN(abs(sa), abs(sb));
                            s_std = sep_y * (1+ pow(s_min - 2, sep_x));
                            w = sep_weight(s_min) * gaussian(0, s_std, s_dif);
                            if (s_dif/s_std <6){
                                Mval = contact_a * contact_b * w;
                            }
                            else{
                                Mval=0.;
                            }
                        }
                        else {
                             Mval = -1;
                        }
                    }
                    else{
                        Mval = 0.;
                    }
                    M[ptr] = Mval;
                    // if (Mval != 0 && Mval != -1){
                    //     printf("contact_a: %f\n", contact_a);
                    //     printf("contact_b: %f\n", contact_b);
                    //     printf("w: %f\n", w);
                    //     printf("Mval: %f\n", Mval);
                    // }
                }
            }
        }
    }
    // return M;
}
