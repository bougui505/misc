#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int * traceback(int rows, int cols, double * sco_mtx, double gap_open, double gap_extension){
    // LOCAL_ALIGN
    // Start	0
    // [A]lign	1
    // [D]own	2
    // [R]ight	3
    double max_sco = 0;
    int * aln = (int *)malloc(sizeof(int) * rows);
    for (int i = 0; i < rows ; i++){
        aln[i] = -1.;
    }

    double sco[rows+1][cols+1];
    memset(sco, 0, sizeof(sco));
    for (int i = 1; i <= rows; i++){
        for (int j = 1; j <= cols; j++){
            double A = sco[i-1][j-1] + sco_mtx[(i-1)*cols+(j-1)];
            double D = sco[i-1][j];
            double R = sco[i][j-1];
            if(A >= R){
                if(A >= D){
                    sco[i][j] = A;
                    aln[i - 1] = j - 1;
                }
                else{
                    sco[i][j] = D;
                    aln[i-1] = j;
                }
            }
            else{
                if(R >= D){
                    sco[i][j] = R;
                    aln[i] = j - 1;
                }
                else{
                    sco[i][j] = D;
                    aln[i-1] = j;
                }
            }
            if(sco[i][j] > max_sco){max_sco = sco[i][j];}
        }
    }
    return aln;
}
