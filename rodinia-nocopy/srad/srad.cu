// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <srad.h>

// includes, project
#include <cuda.h>

// includes, kernels
#include <srad_kernel.cu>


#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
void m5_dump_stats(uint64_t ns_delay, uint64_t ns_period);
}
#endif

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");

	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{

    // BEGIN ADARSH DUMMY LOOP
    int blosum62[24][24] = {
	{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
	{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
	{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
	{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
	{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
	{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
	{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
	{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
	{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
	{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
	{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
	{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
	{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
	{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
	{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
	{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
	{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
	{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
	{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
	{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
    };

    int dummyArray[48][48];
    int i,j,k,ctr;
    for ( i=0; i<24; i++)
        for ( j=0; j<24; j++){
            dummyArray[i][j] = blosum62[i][j];
            dummyArray[i+24][j] = blosum62[i][j];
            dummyArray[i][j+24] = blosum62[i][j];
            dummyArray[i+24][j+24] = blosum62[i][j];
         }

    for ( k=1; k<6000; k++) {
        for ( i=1; i<47; i++)
            for ( j=1; j<47; j++)
                dummyArray[i][j] += (dummyArray[i-1][j] + dummyArray[i+1][j]) * (dummyArray[i][j+1] + dummyArray[i][j-1]);

        for ( i=1; i<47; i++)
            for ( j=1; j<47; j++)
                dummyArray[i][j] += (dummyArray[i-1][j] * dummyArray[i+1][j]) + (dummyArray[i][j+1] * dummyArray[i][j-1]);

        for (i=0;i<48; i++) {
            ctr = dummyArray[0][i];
            for ( j=1;j<48; j++) {
                ctr += dummyArray[j][i];
                dummyArray[j][i] = ctr;
            }
        }

		for (i=0;i<48; i++) {
		    ctr = dummyArray[i][0];
		    for ( j=1;j<48; j++) {
		        ctr += dummyArray[i][j];
		        dummyArray[i][j] = ctr;
		    }
        }
    }

    fprintf(stdout, "Begin dummy output\n");
    for ( i=1; i<48; i++)
        fprintf(stdout, "%d ", dummyArray[23][i]);
    fprintf(stdout, "\nEnd of dummy output\n");

    // END ADARSH DUMMY LOOP

    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv)
{
    int rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

    float *dN,*dS,*dW,*dE;
#ifdef CPU
	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW, k;
	float cN,cS,cW,cE,D;
#endif

#ifdef GPU

	//float *J_cuda;
  //  float *C_cuda;
	//float *E_C, *W_C, *N_C, *S_C;

#endif

	unsigned int r1, r2, c1, c2;
	float *c;



	if (argc == 9)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
		fprintf(stderr, "rows and cols must be multiples of 16\n");
		exit(1);
		}
		r1   = atoi(argv[3]);  //y1 position of the speckle
		r2   = atoi(argv[4]);  //y2 position of the speckle
		c1   = atoi(argv[5]);  //x1 position of the speckle
		c2   = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]); //Lambda value
		niter = atoi(argv[8]); //number of iterations

	}
    else{
	usage(argc, argv);
    }



	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;



	dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;

#ifdef CPU

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;


    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

#endif

#ifdef GPU

	//Allocate device memory
    //cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    //cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
	//cudaMalloc((void**)& E_C, sizeof(float)* size_I);
	//cudaMalloc((void**)& W_C, sizeof(float)* size_I);
	//cudaMalloc((void**)& S_C, sizeof(float)* size_I);
	//cudaMalloc((void**)& N_C, sizeof(float)* size_I);


#endif

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }
	printf("Start the SRAD main loop\n");
#ifdef GEM5_FUSION
    m5_dump_stats(0, 0);
    m5_work_begin(0, 0);
#endif
 for (iter=0; iter< niter; iter++){
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);



#ifdef CPU

		for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) {

				k = i * cols + j;
				Jc = J[k];

				// directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;

                G2 = (dN[k]*dN[k] + dS[k]*dS[k]
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);

                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;

                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
		}
	}
         for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                // current index
                k = i * cols + j;

                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
	}

#endif // CPU


#ifdef GPU

	//Currently the input size must be divided by 16 - the block size
	int block_x = cols/BLOCK_SIZE ;
    int block_y = rows/BLOCK_SIZE ;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);


	//Copy data from main memory to device memory
	//cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

	//Run kernels
	srad_cuda_1<<<dimGrid, dimBlock>>>(dE, dW, dN, dS, J, c, cols, rows, q0sqr);
  srad_cuda_2<<<dimGrid, dimBlock>>>(dE, dW, dN, dS, J, c, cols, rows, lambda, q0sqr);
  cudaThreadSynchronize();

	//Copy data from device memory to main memory
    //cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);

#endif
}

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    cudaThreadSynchronize();

#ifdef OUTPUT
    //Printing output
		printf("Printing Output:\n");
    for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         printf("%.5f ", J[i * cols + j]);
		}
     printf("\n");
   }
#endif

	printf("Computation Done\n");

	free(I);
  free(J);
  free(dN); free(dS); free(dW); free(dE);
#ifdef CPU
	free(iN); free(iS); free(jW); free(jE);
#endif
#ifdef GPU
  //  cudaFree(C_cuda);
	//cudaFree(J_cuda);
	//cudaFree(E_C);
	//cudaFree(W_C);
	//cudaFree(N_C);
	//cudaFree(S_C);
#endif
	free(c);

}


void random_matrix(float *I, int rows, int cols){

	srand(7);

	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}

