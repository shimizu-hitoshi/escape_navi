#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "twd.h"

void sparse_twd::readASCI_File(char *fn){
	FILE	*fp;
	int	i, j, d, h;
	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	fscanf(fp, "%d %d", &N, &dim);
	vec	= new int[N];
	ind	= new int * [N];
	val	= new double * [N];
	for(i =0; i < N; i++){
		fscanf(fp, "%d", &d);
		vec[i]	= d;
		ind[i]	= new int [d];
		val[i]	= new double [d];

		for(j =0; j < d; j++){
			fscanf(fp, " %d:%lf", &h, &val[i][j]);
			ind[i][j]=h-1;
		}
	}
	fclose(fp);	
}

void	sparse_twd::readBINARY_File(char *fn){
	FILE	*fp;
	fp	= fopen(fn , "rb");
	readBINARY_FilePointer(fp);
	fclose(fp);
}

void	sparse_twd::readBINARY_FilePointer(FILE *fp){
	int	i;
        fread(&N, sizeof(int),1, fp);
        fread(&dim, sizeof(int),1, fp);
	vec	= new int[N];
	ind	= new int * [N];
	val	= new double * [N];
        fread(vec, sizeof(int), N, fp);
	for( i = 0; i < N; i++){
		ind[i]	= new int [vec[i]];
		val[i]	= new double [vec[i]];
		fread(ind[i], sizeof(int) * vec[i], 1, fp);
		fread(val[i], sizeof(double) * vec[i], 1, fp);
	}

}

void	sparse_twd::writeBINARY_File(char *fn){
	FILE	*fp;
	fp	= fopen(fn , "wb");
	writeBINARY_FilePointer(fp);
	fclose(fp);
}
void	sparse_twd::writeBINARY_FilePointer(FILE *fp){
	int	i;
	fwrite(&N, sizeof(int), 1, fp);
	fwrite(&dim, sizeof(int), 1, fp);
	fwrite(vec, sizeof(int) , N, fp);
	for( i = 0; i < N; i++){
		fwrite(ind[i], sizeof(int) ,  vec[i], fp);
		fwrite(val[i], sizeof(double), vec[i], fp);
	}
}

sparse_twd	sparse_twd::clone(){
	sparse_twd	dest;
	int	i, j;
	dest.N		= N;
	dest.dim	= dim;
	dest.vec	= new int [N];
	dest.ind	= new int * [N];
	dest.val	= new double * [N];
	for(i =0; i < N; i++){
		dest.vec[i]	= vec[i];
		dest.ind[i]	= new int [vec[i]];
		dest.val[i]	= new double [vec[i]];
		for(j = 0; j < vec[i]; j++){
			dest.ind[i][j]	= ind[i][j];
			dest.val[i][j]	= val[i][j];
		}
	}
	return dest;
}

sparse_twd	sparse_twd::trans(){
	sparse_twd	dest;
	int	i, j, p;
	dest.N		= dim;
	dest.dim	= N;
	dest.vec	= new int[dim];
	dest.ind	= new int * [dim];
	dest.val	= new double * [dim];
	for(i = 0; i < dim; i++) dest.vec[i] = 0;
	for(i = 0; i < N; i++)
		for(j = 0; j < vec[i]; j++)
			dest.vec[ind[i][j]]++;
	for(i = 0; i < dim; i++){
		dest.ind[i] = new int [dest.vec[i]];
		dest.val[i] = new double [dest.vec[i]];
		dest.vec[i] = 0;
	}
	
	for(i = 0; i < N; i++){
		for(j = 0; j < vec[i]; j++){
			p = dest.vec[ind[i][j]];
			dest.ind[ind[i][j]][p] = i;
			dest.val[ind[i][j]][p] = val[i][j];
			dest.vec[ind[i][j]]++;
		}
	}
	return dest;
}

void	sparse_twd::clear(){
	int	i;
	for(i = 0; i < N; i++){
		delete[] ind[i];
		delete[] val[i];
	}
	delete[] vec;
	delete[] ind;
	delete[] val;
}

void	sparse_twd::normalizeL1(){
	int	i, j;
	double	v;
	for(i = 0; i < N; i++){
		v = 0.0;
		for(j = 0; j < vec[i]; j++) v += val[i][j];
		for(j = 0; j < vec[i]; j++) val[i][j] /= v;
	}
}
void	sparse_twd::normalizeL2(){
	int	i, j;
	double	v;
	for(i = 0; i < N; i++){
		v = 0.0;
		for(j = 0; j < vec[i]; j++) v += val[i][j]*val[i][j];
		if(v != 0.0){
			v = 1.0/sqrt(v);
			for(j = 0; j < vec[i]; j++) val[i][j] /= v;
		}
	}
}

sparse_twd	sparse_twd::dijkstra(){
	int	i, j, k;
	int	src, dest;
	int	q, *que, n, *next, *flg;
	double	v;
	sparse_twd	result;
	if(N != dim){
		printf("error\n");
		exit(1);
	}
	que		= new int [N];
	next		= new int [N];
	flg		= new int [N];

	result.N	= N;
	result.dim	= N;
	result.vec	= new int 	[N];
	result.ind	= new int*	[N];
	result.val	= new double*	[N];
	for( i = 0; i < N; i++){
		result.vec[i] = N;
		result.ind[i]	= new int	[N];
		result.val[i]	= new double	[N];
		for( j = 0; j < N; j++){
			result.ind[i][j] = j;
			result.val[i][j] = DBL_MAX;
		}
		result.val[i][i] = 0.0;
		q = 1;
		que[0] = i;
		while(q){
			n = 0;
			for(j = 0; j < N; j++) flg[j] = 0;
			for(j = 0; j < q; j++){
				src = que[j];
				for(k = 0; k < vec[src]; k++){
					dest	= ind[src][k];
					v	= val[src][k];
					if(result.val[i][dest] > result.val[i][src]+v){
						result.val[i][dest] = result.val[i][src]+v;

						if(flg[dest]==0){
							next[n++] = dest;
							flg[dest] = 1;
						}

					}
				}
			}
			q = n;
			for(j = 0; j < n; j++) que[j] = next[j];
		}
	}
	return result;
}
