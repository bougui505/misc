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
    int * aln = (int *)malloc(sizeof(int) * rows);
    for (int i = 0; i < rows ; i++){
        aln[i] = -1;
    }
    while(1){
        labelpt = i * cols + j;
        if(label[labelpt] == 0){break;}
        else if(label[labelpt] == 1){aln[i-1] = j-1;i--;j--;}
        else if(label[labelpt] == 2){i--;}
        else if(label[labelpt] == 3){j--;}
    }
    return aln;
}
