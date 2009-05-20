#include <math.h>
#include "mex.h"

/* See lbfgsUpdate.m for details! */

void lbfgsUpdate(const int mrowsY, const int numCorrections, const double* y, const double* s, const int corrections, const int debug, const double* Hdiag, double* old_dirs_out, double* old_stps_out, double* Hdiag_out){
    /* Variable Declarations */
    int i, c, num_columns;
    double y_times_s, y_times_y;

    if( numCorrections == corrections ){
        num_columns = numCorrections;
    } else {
        num_columns = numCorrections+1;
    }
    
    y_times_s=0;
    y_times_y=0;
    for (i=0; i<mrowsY; i++){
        y_times_s += y[i]*s[i];
        y_times_y += y[i]*y[i];
    }
    
    if (y_times_s > 1e-10){
        if( numCorrections == corrections ){
            /* Limited-Memory Update: first shift elements over by 1 */
            for( c=0; c<numCorrections-1; c++ ){
                for( i=0; i<mrowsY; i++ ){
                    old_dirs_out[c*mrowsY + i] = old_dirs_out[(c+1)*mrowsY + i];
                    old_stps_out[c*mrowsY + i] = old_stps_out[(c+1)*mrowsY + i];
                }
            }
            num_columns = numCorrections;
        } else {
            /* Full Update */
            num_columns = numCorrections+1;
        }
        
        /* Fill last column with new elements */
        for( i=0; i<mrowsY; i++ ){
            old_dirs_out[(num_columns-1)*mrowsY + i] = s[i];
            old_stps_out[(num_columns-1)*mrowsY + i] = y[i];
        }
        
        /* Update scale of initial Hessian approximation */
        Hdiag_out[0] = y_times_s/y_times_y;
        
    } else {
        if (debug){
            printf("Skipping Update\n");
        }
        Hdiag_out[0] = Hdiag[0];
    }
}
    

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Variable Declarations */
    double *old_dirs, *old_stps, *y, *s;
    double *old_dirs_out, *old_stps_out, *Hdiag_out, *Hdiag;
    int corrections, debug, mrowsY, mrows, ncols, numCorrections, i, c;
    int *tmp_int_ptr, dims[2];
    bool skip_update;
    double y_times_s;

    /* Check for proper number of arguments. */
    if(nrhs > 8) {
        mexErrMsgTxt("Usage: [old_dirs,old_stps,Hdiag] = lbfgsUpdate(y,s,corrections,debug,old_dirs,old_stps,Hdiag)");
    } else if(nlhs>3) {
        mexErrMsgTxt("Too many output arguments: [old_dirs,old_stps,Hdiag]");
    }
    
    /* Get Input Pointers */
    mrowsY = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[2]) || !(ncols==1)) {
        mexErrMsgTxt("y must be a noncomplex double column vector.");
    }
	y = mxGetPr(prhs[0]);

    mrows = mxGetM(prhs[1]);
    ncols = mxGetN(prhs[1]);
    if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[2]) || !(ncols==1) || !(mrows==mrowsY)) {
        mexErrMsgTxt("s must be a noncomplex double column vector of the same length as y.");
    }
    s = mxGetPr(prhs[1]);

    mrows = mxGetM(prhs[2]);
    ncols = mxGetN(prhs[2]);
    if( !mxIsInt32(prhs[2]) || mxIsComplex(prhs[2]) || !(ncols==1) || !(mrows==1)) {
        mexErrMsgTxt("corrections must be a noncomplex int scalar.");
    }
    tmp_int_ptr = (int*) mxGetData(prhs[2]);
    corrections = tmp_int_ptr[0];
    
    mrows = mxGetM(prhs[3]);
    ncols = mxGetN(prhs[3]);
    if( !mxIsInt32(prhs[3]) || mxIsComplex(prhs[3]) || !(ncols==1) || !(mrows==1)) {
        mexErrMsgTxt("debug must be a noncomplex int scalar.");
    }
    tmp_int_ptr = (int*) mxGetData(prhs[3]);
    debug = tmp_int_ptr[0];
    
    mrows = mxGetM(prhs[4]);
    numCorrections = mxGetN(prhs[4]);
    if( !mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || !(mrows==mrowsY)) {
        mexErrMsgTxt("old_dirs must be a noncomplex double N by C matrix where C=length(y).");
    }
    if( numCorrections > corrections ){
        mexErrMsgTxt("number of columns in old_dirs must be <= corrections.");
    }
    old_dirs = mxGetPr(prhs[4]);
    
    mrows = mxGetM(prhs[5]);
    numCorrections = mxGetN(prhs[5]);
    if( !mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || !(mrows==mrowsY)) {
        mexErrMsgTxt("old_stps must be a noncomplex double N by C matrix where C=length(y).");
    }
    if( numCorrections > corrections ){
        mexErrMsgTxt("number of columns in old_dirs must be <= corrections.");
    }
    old_stps = mxGetPr(prhs[5]);

    mrows = mxGetM(prhs[6]);
    ncols = mxGetN(prhs[6]);
    if( !mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || !(mrows==1) || !(ncols==1)) {
        mexErrMsgTxt("Hdiag must be a noncomplex double scalar.");
    }
    Hdiag = mxGetPr(prhs[6]);
        
    /* The size of the output matrices is max(numCorretions+1, corrections) */
    dims[0] = mrowsY;
    
    y_times_s=0;
    for (i=0; i<mrowsY; i++){
        y_times_s += y[i]*s[i];
    }
    skip_update = (y_times_s <= 1e-10);
    
    if( (numCorrections == corrections) || skip_update ){
        dims[1] = numCorrections;
    } else {
        dims[1] = numCorrections+1;
    }
    plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    old_dirs_out = mxGetPr(plhs[0]);

    plhs[1] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    old_stps_out = mxGetPr(plhs[1]);
    
    dims[0]=1;
    dims[1]=1;
    plhs[2] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    Hdiag_out = mxGetPr(plhs[2]);

    /* Copy input to avoid badness. */
    for( c=0; c<numCorrections; c++ ){
        for( i=0; i<mrowsY; i++ ){
            /* Matlab indexing is: column_idx * total_num_rows + row_idx */
            old_dirs_out[c*mrowsY + i] = old_dirs[c*mrowsY + i];
            old_stps_out[c*mrowsY + i] = old_stps[c*mrowsY + i];
        }
    }    
    
    /* Doing the actual work. */
    if( !skip_update ){
        lbfgsUpdate(mrowsY,numCorrections,y,s,corrections,debug,Hdiag,old_dirs_out,old_stps_out,Hdiag_out);
    }
}