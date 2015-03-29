/*

Reconstructs a positive object from diffraction intensities using the difference map. Only non-negative
intensity values in the input file, object_intensity.dat, are used as constraints. The reconstructed
object values are written to the output file finish_object.dat. Iterations begin with a random object
unless the optional input file start_object.dat is placed in the directory. Reconstructions are averaged
after a transient period that is specified in the command line. Residual phase fluctuations during the
averaging period are used to compute a modulation transfer function which is written to the file mtf.dat.
The difference map error metric is written to the output file object.log.

Written by V. Elser; last modified April 2009.


compile:
gcc -O3 object_recon.c -lm -lfftw3 -o object_recon

usage:
object_recon iter start_ave (iter = number of iterations, start_ave = iterations before start of averaging)

needs:
support.dat, object_intensity.dat [start_object.dat]
fftw3 library

makes:
finish_object.dat, mtf.dat, object.log

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <assert.h>

#define MTF 20

int (*supp)[3] ;
int *supp_flag ;
int *ord, *beg, *end ;
double *tempx, ***x, ***p1, ***r1, ***p2, ***mag, ***ave, ***min_state ;
double *fftw_array_r ;
fftw_complex *fftw_array_c ;
fftw_plan forward_plan, backward_plan ; 

int size, qmax, size_supp, num_supp, shrink_interval, ave_iter = 0 ;

void print_recon() ;
void print_mtf() ;
void ave_recon( double*** ) ;
int setup() ;
void free_mem() ;
double diff() ;
void proj1( double***, double*** ) ;
void proj2( double***, double*** ) ;


int main(int argc, char* argv[])
	{
	int num_trials, iter, start_ave, i, t;
	double error, min_error ;
	FILE *fp ;
	
	if ( argc == 5 )
		{
        num_trials      = atoi(argv[1]) ;
		iter            = atoi(argv[2]) ;
		start_ave       = atoi(argv[3]) ;
        shrink_interval = atoi(argv[4]) ;
		}
	else
		{
		printf("expected four arguments: num_trials, num_iter start_ave shrink_interval\n") ;
		return 0 ;
		}
		
	if (!setup())
		return 0 ;
	
	fp = fopen("object.log", "w") ;
	fprintf(fp, "size = %d    size_supp = %d    num_supp = %d\n\n", size, size_supp, num_supp) ;
	fclose(fp) ;

    // TODO: Loop over a number of N_trials
    // TODO: Loop over a number of N_iter
    // TODO: Once "averaging begins, then we seek the N smallest error reconstruction(s)
    // TODO: Compute translation and rotation between different constructions before merging
    //          We'll do this in a complete search over orientations and translations.
    // TODO: Save the lowest error file out to disk
    // TODO: Do not compute MTF here. Do this outside of each attempt.
    for (t = 1 ; t <= num_trials ; ++t)
        {
	    min_error = 100. ;
        randomize_state() ;
        fp = fopen("object.log", "a") ;
        fprintf(fp, "\nStarting trial %02d\n", t) ;
        fclose(fp) ;
        for (i = 1 ; i <= iter ; ++i)
            {
            error = diff() ;
        
            if (i % shrink_interval && t == 1) //Only shrinkwrap on the first trial
                shrink_support(p2) ;

            if (i > start_ave)
                {
                ave_recon(p2) ;
                if (error < min_error)
                    {
                    min_error = error ;
                    replace_min_recon(p2) ; 
                    }
                }
            fp = fopen("object.log", "a") ;
            fprintf(fp, "iter = %d    error = %f\n", i, error) ;
            fclose(fp) ;
            }
        
        print_recon() ;
        print_min_recon(t) ; 	
        print_mtf() ;
        }
	free_mem() ;
	
	return 0 ;
	}
	

void randomize_state()
    {
    int s, is, js, ks, i, j, k ;
    srand( time(0) ) ;
    
    for (s = 0 ; s < num_supp ; ++s)
        {
        is = supp[s][0] ;
        js = supp[s][1] ;
        ks = supp[s][2] ;
        
        x[is][js][ks] = ((double) rand()) / RAND_MAX ;
        }
    
    for (i = 0 ; i < size_supp ; ++i)
    for (j = 0 ; j < size_supp ; ++j)
    for (k = 0 ; k < size_supp ; ++k)
        x[i][j][k] = ((double) rand()) / RAND_MAX ;
    }

void print_recon()
	{
	FILE *fp ;
	int i, j, k ;
	
	fp = fopen("finish_object.dat", "w") ;
	
	for (i = 0 ; i < size_supp ; ++i)
	for (j = 0 ; j < size_supp ; ++j)
		{
		for (k = 0 ; k < size_supp ; ++k)
			fprintf(fp, "%f ", ave[i][j][k] / ave_iter) ;
			
		fprintf(fp, "\n") ;
		}
		
	fclose(fp) ;
	}
	
void print_min_recon(int num)
	{
	FILE *fp ;
	int i, j, k ;
    char buffer [100] ;
    sprintf(buffer, "finish_min_object%03d.dat", num) ;
	fp = fopen(buffer, "w") ;
	
	for (i = 0 ; i < size_supp ; ++i)
	for (j = 0 ; j < size_supp ; ++j)
		{
		for (k = 0 ; k < size_supp ; ++k)
			fprintf(fp, "%f ", min_state[i][j][k]) ;
			
		fprintf(fp, "\n") ;
		}
		
	fclose(fp) ;
	}

int setup()
	{
	int i, j, k, is, js, ks, s, it, jt ;
	double intens ;
	FILE *fp ;

	fp = fopen("support.dat", "r") ;
	if (!fp)
		{
		printf("cannot open support.dat\n") ;
		return 0 ;
		}
		
	fscanf(fp, "%d %d", &qmax, &num_supp) ;
	size = 2 * qmax + 1 ;

	supp = malloc(num_supp * sizeof(*supp)) ;
    supp_flag = malloc(num_supp * sizeof(*supp_flag)) ;	
	size_supp = 0 ;
	for (s = 0 ; s < num_supp ; ++s)
        {
        supp_flag[s] = 1 ;
        for (i = 0 ; i < 3 ; ++i)
            {
            fscanf(fp, "%d", &supp[s][i]) ;
            
            if (supp[s][i] > size_supp)
                size_supp = supp[s][i] ;
            }
        }
	++size_supp ;

	fclose(fp) ;
	
	fp = fopen("object_intensity.dat", "r") ;
	if (!fp)
		{
		printf("cannot open object_intensity.dat\n") ;
		return 0 ;
		}
	
    ord = malloc(size_supp * size_supp * size_supp * sizeof(int));
    beg = malloc(size_supp * size_supp * size_supp * sizeof(int));
    end = malloc(size_supp * size_supp * size_supp * sizeof(int));
    tempx = malloc(size_supp * size_supp * size_supp * sizeof(double));
    
	mag = malloc(size * sizeof(double**)) ;
	x = malloc(size * sizeof(double**)) ;
	p1 = malloc(size * sizeof(double**)) ;
	r1 = malloc(size * sizeof(double**)) ;
	p2 = malloc(size * sizeof(double**)) ;
	ave = malloc(size * sizeof(double**)) ;
    min_state = malloc(size * sizeof(double **)) ;

	for (i = 0 ; i < size ; ++i)
		{
		mag[i] = malloc(size * sizeof(double*)) ;
		x[i] = malloc(size * sizeof(double*)) ;
		p1[i] = malloc(size * sizeof(double*)) ;
		r1[i] = malloc(size * sizeof(double*)) ;
		p2[i] = malloc(size * sizeof(double*)) ;
		ave[i] = malloc(size * sizeof(double*)) ;
		min_state[i] = malloc(size * sizeof(double*)) ;
		
		for (j = 0 ; j < size ; ++j)
			{
			mag[i][j] = malloc((qmax + 1) * sizeof(double)) ;
			x[i][j] = malloc(size * sizeof(double)) ;
			p1[i][j] = malloc(size * sizeof(double)) ;
			r1[i][j] = malloc(size * sizeof(double)) ;
			p2[i][j] = malloc(size * sizeof(double)) ;
			ave[i][j] = malloc(size * sizeof(double)) ;
			min_state[i][j] = malloc(size * sizeof(double)) ;
			}
		}
		
	for (i = 0 ; i < size ; ++i)
		{
		it = (i < qmax) ? i + qmax + 1 : i - qmax ;
		
		for (j = 0 ; j < size ; ++j)
			{
			jt = (j < qmax) ? j + qmax + 1 : j - qmax ;
			
			for (k = 0 ; k < size ; ++k)
				{
				fscanf(fp, "%lf", &intens) ;
				if (k >= qmax)
					mag[it][jt][k - qmax] = sqrt(intens) ;
				}
			}
		}
							
	fclose(fp) ;
	
	fftw_array_r = (double*) fftw_malloc(size*size*size * sizeof(double)) ;
	fftw_array_c = (fftw_complex*) fftw_malloc(size*size*(qmax + 1) * sizeof(fftw_complex)) ;
	
	forward_plan = fftw_plan_dft_r2c_3d(size, size, size, fftw_array_r, fftw_array_c, FFTW_MEASURE) ;
	backward_plan = fftw_plan_dft_c2r_3d(size, size, size, fftw_array_c, fftw_array_r, FFTW_MEASURE) ;
	
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
		{
		x[i][j][k] = 0. ;
		ave[i][j][k] = 0. ;
		}
		
	fp = fopen("start_object.dat", "r") ;
	if (!fp)
		{
		srand( time(0) ) ;
		
		for (s = 0 ; s < num_supp ; ++s)
			{
			is = supp[s][0] ;
			js = supp[s][1] ;
			ks = supp[s][2] ;
			
			x[is][js][ks] = ((double) rand()) / RAND_MAX ;
			}
        
		for (i = 0 ; i < size_supp ; ++i)
		for (j = 0 ; j < size_supp ; ++j)
		for (k = 0 ; k < size_supp ; ++k)
            x[i][j][k] = ((double) rand()) / RAND_MAX ;
            
		}
	else
		{
		for (i = 0 ; i < size_supp ; ++i)
		for (j = 0 ; j < size_supp ; ++j)
		for (k = 0 ; k < size_supp ; ++k)
			fscanf(fp, "%lf", &x[i][j][k]) ;
				
		fclose(fp) ;
		}
			
	return 1 ;
	}
		
	
void free_mem()
	{
	int i, j ;
	
	free(supp) ;
    free(supp_flag) ;
    free(ord) ;
    free(beg) ;
    free(end) ;
    free(tempx) ;
	
	for (i = 0 ; i < size ; ++i)
		{
		for (j = 0 ; j < size ; ++j)
			{
            free(min_state[i][j]) ;
			free(mag[i][j]) ;
			free(x[i][j]) ;
			free(p1[i][j]) ;
			free(r1[i][j]) ;
			free(p2[i][j]) ;
			free(ave[i][j]) ;
			}
        free(min_state[i]) ;			
		free(mag[i]) ;
		free(x[i]) ;
		free(p1[i]) ;
		free(r1[i]) ;
		free(p2[i]) ;
		free(ave[i]) ;
		}
	
    free(min_state) ;
	free(mag) ;
	free(x) ;
	free(p1) ;
	free(r1) ;
	free(p2) ;
	free(ave) ;
	
	fftw_free(fftw_array_r) ;
	fftw_free(fftw_array_c) ;
	
	fftw_destroy_plan(forward_plan) ;
	fftw_destroy_plan(backward_plan) ;
	}
	
	
void ave_recon( double ***in )
	{
	int i, j, k ;
	
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
		ave[i][j][k] += in[i][j][k] ;
		
	++ave_iter ;
	}
	

void replace_min_recon( double ***in )
	{
	int i, j, k ;
	
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
		min_state[i][j][k] += in[i][j][k] ;
	}


void print_mtf()
	{
	double rel_contrast[MTF] ;
	int bin_count[MTF] ;
	int i, ir, j, jr, k, r ;
	int qmax1 ;
	double fftw_norm , tempVal ;
	FILE *fp ;
	
	qmax1 = qmax + 1 ;
	fftw_norm = sqrt( (double) size * size * size ) ;
	
	for (r = 0 ; r < MTF ; ++r)
		{
		rel_contrast[r] = 0. ;
		bin_count[r] = 0 ;
		}
		
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
		fftw_array_r[(size * i + j) * size + k] = ave[i][j][k] / ave_iter ;
			
	fftw_execute( forward_plan ) ;
	
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < qmax1 ; ++k)
		{
		ir = (i < qmax + 1) ? i : i - size ;
		jr = (j < qmax + 1) ? j : j - size ;
		r = .5 + MTF * sqrt( ((double) ir*ir + jr*jr + k*k) / (qmax*qmax) ) ;
		
		if (r < MTF && mag[i][j][k] > 0.)
			{
            tempVal = cabs(fftw_array_c[(size * i + j) * qmax1 + k]) / (fftw_norm * mag[i][j][k]) ;
			rel_contrast[r] += tempVal ;
			++bin_count[r] ;
			}
		}
		
	fp = fopen("mtf.dat", "w") ;
	for (r = 0 ; r < MTF ; ++r)
		if (bin_count[r] == 0)
			fprintf(fp, "%5.3f  %8.6f\n", (r + 1.) / MTF, 0.) ;
		else
			fprintf(fp, "%5.3f  %8.6f\n", (r + 1.) / MTF, rel_contrast[r] / bin_count[r]) ;
		
	fclose(fp) ;
	}
	
	
double diff()
	{
	int i, j, k ;
	double change, error = 0. ;
	

    proj1(x, p1) ;
    
    double leash = 0.2;
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
        {
        x[i][j][k] = (1.0-leash)*x[i][j][k] + leash*p1[i][j][k] ;
        }

	proj2(x, p1) ;
	
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
		r1[i][j][k] = 2. * p1[i][j][k] - x[i][j][k] ;
		
	proj1(r1, p2) ;
	
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
			{
			change = p2[i][j][k] - p1[i][j][k] ;
			x[i][j][k] += change ;
			error += change * change ;
			}
		
	return sqrt( error / (size * size * size) ) ;
	}
	

void proj1( double ***in, double ***out )
	{
	int i, j, k, qmax1 ;
	double vol ;
	
	vol = size * size * size ;
	qmax1 = qmax + 1 ;
	
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
		fftw_array_r[(size * i + j) * size + k] = in[i][j][k] ;
			
	fftw_execute( forward_plan ) ;
	
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < qmax + 1 ; ++k)
		if (mag[i][j][k] > 0.)
			fftw_array_c[(size * i + j) * qmax1 + k] *= mag[i][j][k] / cabs(fftw_array_c[(size * i + j) * qmax1 + k]) ;
		else if (mag[i][j][k] == 0.)
			fftw_array_c[(size * i + j) * qmax1 + k] = 0. ;
		else
			fftw_array_c[(size * i + j) * qmax1 + k] /= sqrt(vol) ;
	
	fftw_execute( backward_plan ) ;
		
	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
		out[i][j][k] = fftw_array_r[(size * i + j) * size + k] / sqrt(vol) ;
	}
	
	
void proj2( double ***in, double ***out )
	{
	int i, j, k, is, js, ks, s, pivPos, L, R ;
	double val, cutoff, piv, b, c ;

	for (i = 0 ; i < size ; ++i)
	for (j = 0 ; j < size ; ++j)
	for (k = 0 ; k < size ; ++k)
        out[i][j][k] = 0. ;

	/*
	for (i = 0 ; i < size_supp ; ++i)
	for (j = 0 ; j < size_supp ; ++j)
	for (k = 0 ; k < size_supp ; ++k)
        {
        ord[(size_supp * i + j) * size_supp + k] = (size_supp * i + j) * size_supp + k;
        tempx[(size_supp * i + j) * size_supp + k] = in[i][j][k] ;
        }
	*/
     
	for (s = 0 ; s < num_supp ; ++s)
		{
		is = supp[s][0] ;
		js = supp[s][1] ;
		ks = supp[s][2] ;
		
		val = in[is][js][ks] ;
		
		if (val < 0. || supp_flag[s] == 0)
			continue ;
			
		out[is][js][ks] = val ;
		}
    
    /*
    i = 0 ; beg[0] = 0 ; end[0] = size_supp*size_supp*size_supp ;
	while (i >= 0) 
		{
		L = beg[i] ; R = end[i] - 1 ;
		
		if (L < R) 
			{
			piv = tempx[L] ; pivPos = ord[L] ; 
			while (L < R) 
				{
				while (tempx[R] >= piv && L < R) R-- ; 
				if (L < R) 
					{ ord[L] = ord[R] ; tempx[L++] = tempx[R] ; } 

				while (tempx[L] <= piv && L < R) L++ ; 
				if (L < R) 
					{ ord[R] = ord[L] ; tempx[R--] = tempx[L] ; } 
				}
			tempx[L] = piv ; ord[L] = pivPos ; beg[i+1] = L+1 ; end[i+1] = end[i] ; end[i++] = L; 
			}
		else 
			i--; 
		}

    cutoff = tempx[size_supp*size_supp*size_supp - num_supp] ;
     
	for (i = 0 ; i < size_supp ; ++i)
	for (j = 0 ; j < size_supp ; ++j)
	for (k = 0 ; k < size_supp ; ++k)
        {
        out[i][j][k] = (in[i][j][k] >= cutoff) ? in[i][j][k] : 0. ;
        }
    */
	}


void shrink_support(double ***in)
    {
    int s, is, js, ks ;
    double mean_1, mean_2, c_mean_1, c_mean_2, t_mean_1, t_mean_2 ;
    double update_err ;
    mean_1 = 1. ;
    mean_2 = 0. ;
    update_err = 10. ;
    while (update_err > 1.E-4)
        {
        c_mean_1 = c_mean_2 = 0. ;
        t_mean_1 = t_mean_2 = 0. ;
        for (s = 0 ; s < num_supp ; s++)
            {
            is = supp[s][0] ;
            js = supp[s][1] ;
            ks = supp[s][2] ;
            val = in[is][js][ks] ;
            if (fabs(val - mean_1) < fabs(val - mean_2))
                  {
                  t_mean_1 += val ;
                  c_mean_1 += 1. ;
                  }
            else
                  {
                  t_mean_2 += val ;
                  c_mean_2 += 1. ;
                  }
            }
        t_mean_1 /= c_mean_1 ;
        t_mean_2 /= c_mean_2 ;
        update_err = fabs(t_mean_1 - mean_1) + fabs(t_mean_2 - mean_2) ;
        mean_1 = t_mean_1 ;
        mean_2 = t_mean_2 ;
        }
    //Mean_1 should be > mean_2
    if (t_mean_1 > t_mean_2) 
        {
        mean_1 = t_mean_1 ; 
        mean_2 = t_mean_2 ;
        }
    else 
        {
        mean_2 = t_mean_1 ; 
        mean_1 = t_mean_2 ;
        }

    //Set support flag based on mean partitions
    for (s = 0 ; s < num_supp ; s++)
        {
        is = supp[s][0] ;
        js = supp[s][1] ;
        ks = supp[s][2] ;
        val = in[is][js][ks] ;
        if (fabs(val - mean_1) < fabs(val - mean_2))
            {
            supp_flag[i] = 1 ;
            }
        else
            supp_flag[i] = 0 ;
        }
    //Extend the padding to enforce continuity?
    }
    

