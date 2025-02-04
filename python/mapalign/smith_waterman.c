#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double * traceback(int rows, int cols, const double * sco_mtx, double gap_open, double gap_extension){
    // LOCAL_ALIGN
    // Start	0
    // [A]lign	1
    // [D]own	2
    // [R]ight	3
    double max_sco = 0;
    int label[rows+1][cols+1];
    memset(label, 0, sizeof label);
    //for (int i = 0; i < (rows+1) * (cols+1) ; i++){
    //    label[i] = 0;
    //}

    double sco[rows+1][cols+1];
    memset(sco, 0, sizeof(sco));
    int max_i = 0;
    int max_j = 0;
    int i = 1;
    int j = 1;
    for (i = 1; i <= rows; i++){
        for (j = 1; j <= cols; j++){
            double A = sco[i-1][j-1] + sco_mtx[(i-1)*cols+(j-1)];
            double D = sco[i-1][j];
            double R = sco[i][j-1];
            if(label[i-1][j] == 1){D += gap_open;}else{D += gap_extension;}
            if(label[i][j-1] == 1){R += gap_open;}else{R += gap_extension;}
            if(A <= 0 && D <= 0 && R <= 0){label[i][j] = 0;sco[i][j] = 0;}
            else{
                if(A >= R){
                    if(A >= D){
                        sco[i][j] = A;
                        label[i][j] = 1;
                    }
                    else{
                        sco[i][j] = D;
                        label[i][j] = 2;
                    }
                }
                else{
                    if(R >= D){
                        sco[i][j] = R;
                        label[i][j] = 3;
                    }
                    else{
                        sco[i][j] = D;
                        label[i][j] = 2;
                    }
                }
            }
            if(sco[i][j] > max_sco){max_i = i;max_j = j;max_sco = sco[i][j];}
        }
    }

    // Store the score in the last element of aln
    double * aln = (double *)malloc(sizeof(double) * (rows + 1));
    for (i = 0; i < rows ; i++){
        aln[i] = -1;
    }
    aln[rows] = max_sco;

    i = max_i; j = max_j;
    while(1){
        // printf("%d, %d, %d\n", i, j, label[labelpt]);
        if(label[i][j] == 0){break;}
        else if(label[i][j] == 1){aln[i-1] = j-1;i--;j--;}
        else if(label[i][j] == 2){i--;}
        else if(label[i][j] == 3){j--;}
    }
    return aln;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
double sep_weight(double sep){if(sep <= 4){return 0.50;}else if(sep == 5){return 0.75;}else{return 1.00;}}

double * update_mtx(int na, int nb, const int * aln_in, const double * sco_mtx, double * cmap_a, double * cmap_b, int iter){
    double * sco_mtx_out = malloc(sizeof(double) * na*nb);
    int i = 0;
    for (i = 0; i < na*nb ; i++){
        sco_mtx_out[i] = sco_mtx[i];
    }
    // for (int i=0; i<na; i++){
    //     printf("%d ", aln_in[i]);
    // }
    int bj = 0;
    int sa = 0;
    int sb = 0;
    double s_min = 0;
    double sco = 0.;
    double w = 0.;
    int aptr = 0;
    int bptr = 0;
    int mtxptr = 0;
    double IT = (double)iter + 1;
    double s1 = (IT/(IT+1)); double s2 = (1/(IT+1));
    int ai = 0;
    int bi = 0;
    int aj = 0;
    for (ai=0; ai< na; ai++){
        for (bi=0; bi< nb; bi++){
            sco = 0.;
            for (aj=0; aj< na; aj++){
                aptr = ai * na + aj;
                bj = aln_in[aj];
                // printf("%d %d %d %d\n", ai, aj, bi, bj);
                if (bj != -1){ // if mapping exists
                    bptr = bi * nb + bj;
                    sa = ai - aj;
                    sb = bi - bj;
                    if ((sa>0 && sb>0) || (sa<0 && sb <0)){
                        s_min = MIN(abs(sa), abs(sb));
                        w = sep_weight(s_min);
                        sco += cmap_a[aptr] * cmap_b[bptr] * w;
                    }
                }
            mtxptr = ai * nb + bi;
            sco_mtx_out[mtxptr] = s1 * sco_mtx_out[mtxptr] + s2 * sco;
            }
        }
    }
    return sco_mtx_out;
}
