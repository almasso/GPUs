#define LIMIT -999
//#define TRACE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#ifdef _OPENACC
#include <openacc.h>
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
#define MAXOF2(a,b) (((a)>(b))?(a):(b))
#define MAXIMUM(a,b,c) (MAXOF2(MAXOF2(a,b),c))

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

char get_amino_symbol(int x)
{
	switch(x)
	{
		case 0:
			return('R');break;
		case 1:
			return('A');break;
		case 2:
			return('N');break;
		case 3:
			return('D');break;
		case 4:
			return('C');break;
		case 5:
			return('Q');break;
		case 6:
			return('E');break;
		case 7:
			return('G');break;
		case 8:
			return('H');break;
		case 9:
			return('I');break;
		case 10:
			return('L');break;
		case 11:
			return('K');break;
		case 12:
			return('M');break;
		case 13:
			return('F');break;
		case 14:
			return('P');break;
		case 15:
			return('S');break;
		case 16:
			return('T');break;
		case 17:
			return('W');break;
		case 18:
			return('Y');break;
		case 19:
			return('V');break;
		case 20:
			return('B');break;
		case 21:
			return('Z');break;
		case 22:
			return('X');break;
		case 23:
			return('*');break;
	}
	return('*');
}

int set_amino_symbol(char x)
{
	switch(x)
	{
		case 'R':
			return(0);break;
		case 'A':
			return(1);break;
		case 'N':
			return(2);break;
		case 'D':
			return(3);break;
		case 'C':
			return(4);break;
		case 'Q':
			return(5);break;
		case 'E':
			return(6);break;
		case 'G':
			return(7);break;
		case 'H':
			return(8);break;
		case 'I':
			return(9);break;
		case 'L':
			return(10);break;
		case 'K':
			return(11);break;
		case 'M':
			return(12);break;
		case 'F':
			return(13);break;
		case 'P':
			return(14);break;
		case 'S':
			return(15);break;
		case 'T':
			return(16);break;
		case 'W':
			return(17);break;
		case 'Y':
			return(18);break;
		case 'V':
			return(19);break;
		case 'B':
			return(20);break;
		case 'Z':
			return(21);break;
		case 'X':
			return(22);break;
		case '*':
			return(23);break;
		default:
			return(23);
	}
}




double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows><max_cols> <penalty>\n", argv[0]);
	fprintf(stderr, "\t<max_cols><max_rows>      - x and y dimensions. x>y\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Testing Usage: %s 1\n", argv[0]);

	exit(1);
}




void runTest( int argc, char** argv) 
{
	int max_rows, max_cols, idx, index;
	int *nw_matrix, *input1, *input2;
	char *output1, *output2, *tmp1, *tmp2;
	char string1[10], string2[10];
	int penalty = -1;

	int match, delet, insert, S;

	int blosum;

	double t0, t1;

#ifdef _OPENACC
     acc_init(acc_device_not_host);
     printf(" Compiling with OpenACC support \n");
#endif 
	
    
	// the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 3)
	{
		max_cols = atoi(argv[1]);
		max_rows = atoi(argv[2]);
		blosum   = 1;
		if (max_cols<max_rows)
			usage(argc, argv);
	} else {
		if (argc == 2)
		{
			if (atoi(argv[1])==1){ 
				max_rows = 7;
				max_cols = 7;
				strcpy(string1, "GCATGCU");
				strcpy(string2, "GATTACA");
				blosum = 0;
			} else {
				usage(argc, argv);
			}
		} else
			usage(argc, argv);
	}



	max_rows = max_rows + 1;
	max_cols = max_cols + 1;

	nw_matrix = (int *)malloc( max_rows * max_cols * sizeof(int) );
	input1 = (int *)malloc( max_cols * sizeof(int) );
	input2 = (int *)malloc( max_rows * sizeof(int) );

	tmp1 = (char *)malloc( (max_cols+max_rows) * sizeof(char) );
	tmp2 = (char *)malloc( (max_cols+max_rows) * sizeof(char) );
	for( int i=0; i<max_cols+max_rows-1 ; i++){
		tmp1[i]='-';
		tmp2[i]='-';
	}
	tmp1[max_cols+max_rows]='\0';
	tmp2[max_cols+max_rows]='\0';
	output1 = (char *)malloc( (max_cols+max_rows) * sizeof(char) );
	output2 = (char *)malloc( (max_cols+max_rows) * sizeof(char) );

	printf("Start Needleman-Wunsch\n");


	for( int i=0; i< max_cols-1 ; i++){
		if (blosum) input1[i] = rand() % 10 + 1;
		else input1[i] = set_amino_symbol(string1[i]);
	}
	for( int j=0; j< max_rows-1 ; j++){
		if (blosum) input2[j] = rand() % 10 + 1;
		else input2[j] = set_amino_symbol(string2[j]);
	}


	/* Initialization */
	for (int i = 0 ; i < max_rows; i++)
		for (int j = 0 ; j < max_cols; j++)
			nw_matrix[i*max_cols+j]=0;

	for( int i = 1; i< max_rows ; i++)
		nw_matrix[i*max_cols] = i * penalty;

	for( int j = 1; j< max_cols ; j++)
		nw_matrix[j] = j * penalty;


	/********************/
	/* Needleman-Wunsch */
	/********************/

#pragma acc data copyin(blosum62[0:24*24]) copyin(input1[0:max_cols]) copyin(input2[0:max_rows]) copy(nw_matrix[0:max_rows*max_cols])
{
	t0 = gettime();
	/* Compute top-left matrix */
	#pragma acc kernels loop independent
	for( int i = 0 ; i < max_rows-2 ; i++){
		#pragma acc loop seq
		for( idx = 0 ; idx <= i ; idx++){
			index = (idx + 1) * max_cols + (i + 1 - idx);

			if (blosum)
				S = blosum62[input1[i - idx]][input2[idx]];
			else
				if (input1[i - idx] == input2[idx])
					S = 1;
				else S=-1;

			match  = nw_matrix[index-1-max_cols] + S;
			delet  = nw_matrix[index-1] + penalty;
			insert = nw_matrix[index-max_cols] + penalty;

			nw_matrix[index] = MAXIMUM(match, delet, insert);
		}
	}

	/* Compute diagonals matrix */
	#pragma acc kernels loop independent
	for( int i = max_rows-2; i < max_cols-2 ; i++){
		#pragma acc loop seq
		for( idx = 0 ; idx <= max_rows-2; idx++){
			index = (idx + 1) * max_cols + (i + 1 - idx);

			if (blosum)
				S = blosum62[input1[i - idx]][input2[idx]];
			else
				if (input1[i - idx] == input2[idx])
					S = 1;
				else S=-1;

			match  = nw_matrix[index-1-max_cols] + S;
			delet  = nw_matrix[index-1] + penalty;
			insert = nw_matrix[index-max_cols] + penalty;

			nw_matrix[index] = MAXIMUM(match, delet, insert);
		}
	}

	/* Compute bottom-right matrix */
	#pragma acc kernels loop independent
	for( int i = max_rows-2; i >= 0 ; i--){
		#pragma acc loop seq
		for( idx = 0 ; idx <= i; idx++){
			index =  ( idx+max_rows-1-i ) * max_cols + max_cols-idx-1 ;

			if (blosum)
				S = blosum62[input1[max_cols-idx-2]][input2[idx+max_rows-2-i]];
			else
				if (input1[idx+max_rows-2-i] == input2[max_cols-idx-2])
					S = 1;
				else S=-1;

			match  = nw_matrix[index-1-max_cols] + S;
			delet  = nw_matrix[index-1] + penalty;
			insert = nw_matrix[index-max_cols] + penalty;

			nw_matrix[index] = MAXIMUM(match, delet, insert);
		}
	}

	t1 = gettime();
}

	printf("\nPerformance %f GCUPS\n", 1.0e-9*((max_rows-1)*(max_cols-1)/(t1-t0)));
	


#define TRACEBACK
#ifdef TRACEBACK
	printf("        ");
	for (int i = 0 ; i < max_cols-1; i++)
		printf("  %c  ", get_amino_symbol(input1[i]));
	printf("\n");
	for (int i = 0 ; i < max_rows; i++){
		if (i<1) printf("  ");
		else printf("%c ", get_amino_symbol(input2[i-1])); 
		for (int j = 0 ; j < max_cols; j++){
			if (nw_matrix[i*max_cols+j]>=0)
				if (nw_matrix[i*max_cols+j]<10)
					printf("   %i ", nw_matrix[i*max_cols+j]);
				else 
					printf("  %i ", nw_matrix[i*max_cols+j]);
			else
				if (nw_matrix[i*max_cols+j]<=-10)
					printf(" %i ", nw_matrix[i*max_cols+j]);
				else
					printf("  %i ", nw_matrix[i*max_cols+j]);				

		}
		printf("\n");
	}
	printf("\n");
#endif
	int pos1 = 0;
	int pos2 = 0;
    
	for (int i = max_rows - 1,  j = max_cols - 1; i>=0, j>=0;){
		int nw, n, w, traceback;

		if ( i == 0 && j == 0 ) break;
		if ( i > 0 && j > 0 ){
			nw = nw_matrix[(i - 1) * max_cols + j - 1];
			w  = nw_matrix[ i * max_cols + j - 1 ];
			n  = nw_matrix[(i - 1) * max_cols + j];
		} else if ( i == 0 ){
			nw = n = LIMIT;
			w  = nw_matrix[ i * max_cols + j - 1 ];
		} else if ( j == 0 ){
			nw = w = LIMIT;
			n  = nw_matrix[(i - 1) * max_cols + j];
		} else{
		}

		traceback = MAXIMUM(nw, w, n);
		if(traceback == nw)
			traceback = nw;
		if(traceback == w)
			traceback = w;
		if(traceback == n)
            	traceback = n;

		if(traceback == nw )
		{
			tmp1[pos1] = get_amino_symbol(input1[j-1]);
			tmp2[pos2] = get_amino_symbol(input2[i-1]);
			i--; j--; pos1++; pos2++;continue;
		} else if(traceback == w ){
			tmp1[pos1] = get_amino_symbol(input1[j-1]);
			tmp2[pos2] = '-';
			j--;  pos1++; pos2++; continue;
		} else if(traceback == n ){
			tmp1[pos1] = '-';
			tmp2[pos2] = get_amino_symbol(input2[i-1]);
			i--; pos1++; pos2++; continue;
		}
	}
	for (int i=0; i<pos1; i++)
		output1[i] = tmp1[pos1-i-1];
	output1[pos1] = '\0';
	for (int i=0; i<pos2; i++)
		output2[i] = tmp2[pos1-i-1];
	output2[pos2] = '\0';
	
	printf("input: ");
	for (int i = 0 ; i < max_cols-1; i++)
		printf("%c", get_amino_symbol(input1[i])); printf("\n");
	printf("input: ");
	for (int i = 0 ; i < max_rows-1; i++)
		printf("%c", get_amino_symbol(input2[i])); printf("\n");

	printf("\nNeedleman-Wunsch Alignment\n");
	printf("%s\n", output1);
	printf("%s\n", output2);

	free(nw_matrix);
	free(input1);
	free(input2);
	free(tmp1);
	free(tmp2);

#ifdef _OPENACC
     acc_shutdown(acc_device_not_host);
#endif 	

}
