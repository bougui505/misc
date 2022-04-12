#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double * traceback(int rows, int cols, double * sco_mtx, double gap_open, double gap_extension){
    // LOCAL_ALIGN
    // Start	0
    // [A]lign	1
    // [D]own	2
    // [R]ight	3
    double max_sco = 0;
    int * label = (int *)malloc(sizeof(int) * ((rows+1) * (cols+1)));
    for (int i = 0; i < (rows+1) * (cols+1) ; i++){
        label[i] = 0;
    }

    double sco[rows+1][cols+1];
    memset(sco, 0, sizeof(sco));
    int labelpt = 0;
    int max_i = 0;
    int max_j = 0;
    for (int i = 1; i <= rows; i++){
        for (int j = 1; j <= cols; j++){
            double A = sco[i-1][j-1] + sco_mtx[(i-1)*cols+(j-1)];
            double D = sco[i-1][j];
            double R = sco[i][j-1];
            if(label[(i-1)*cols+j] == 1){D += gap_open;}else{D += gap_extension;}
            if(label[i*cols+(j-1)] == 1){R += gap_open;}else{R += gap_extension;}
            labelpt = i * cols + j;
            if(A >= R){
                if(A >= D){
                    sco[i][j] = A;
                    label[labelpt] = 1;
                }
                else{
                    sco[i][j] = D;
                    label[labelpt] = 2;
                }
            }
            else{
                if(R >= D){
                    sco[i][j] = R;
                    label[labelpt] = 3;
                }
                else{
                    sco[i][j] = D;
                    label[labelpt] = 2;
                }
            }
            if(sco[i][j] > max_sco){max_i = i;max_j = j;max_sco = sco[i][j];}
        }
    }

    int i = max_i;int j = max_j;
    // Store the score in the last element of aln
    double * aln = (double *)malloc(sizeof(double) * (rows + 1));
    for (int i = 0; i < rows ; i++){
        aln[i] = -1;
    }
    aln[rows] = max_sco;
    while(1){
        labelpt = i * cols + j;
        if(label[labelpt] == 0){break;}
        else if(label[labelpt] == 1){aln[i-1] = j-1;i--;j--;}
        else if(label[labelpt] == 2){i--;}
        else if(label[labelpt] == 3){j--;}
    }
    return aln;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
double sep_weight(double sep){if(sep <= 4){return 0.50;}else if(sep == 5){return 0.75;}else{return 1.00;}}

double * update_mtx(int na, int nb, int * aln, double * sco_mtx, double * cmap_a, double * cmap_b, int iter){
    int bj = 0;
    int sa = 0;
    int sb = 0;
    int s_min = 0;
    double sco = 0.;
    double w = 0.;
    int aptr = 0;
    int bptr = 0;
    int mtxptr = 0;
    for (int ai=0; ai< na; ai++){
        for (int bi=0; bi< nb; bi++){
            sco = 0.;
            for (int aj=0; aj< na; aj++){
                aptr = ai * na + aj;
                bj = aln[aj];
                bptr = bi * nb + bj;
                sa = ai - aj;
                sb = bi - bj;
                if ((sa>0 && sb>0) || (sa<0 && sb <0)){
                    s_min = MIN(abs(sa), abs(sb));
                    w = sep_weight(s_min);
                    sco += cmap_a[aptr] * cmap_b[bptr] * w;
                }
            mtxptr = ai * nb + bi;
            sco_mtx[mtxptr] = iter/(iter+1) * sco_mtx[mtxptr] + sco/(iter+1);
            }
        }
    }
    return sco_mtx;
}
