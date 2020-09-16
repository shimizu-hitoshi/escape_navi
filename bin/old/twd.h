#include <stdio.h>
#include <stdlib.h>
#include <math.h>

class sparse_twd{
	public:
		int		N;	//size
		int		dim;
		int		*vec;
		int		**ind;
		double		**val;
		void		readASCI_File(char *fn);
		sparse_twd	clone();
		sparse_twd	trans();
		sparse_twd	dijkstra();
		void		clear();
		void		readBINARY_File(char *fn);
		void		readBINARY_FilePointer(FILE *fp);
		void		writeBINARY_FilePointer(FILE *fp);
		void		writeBINARY_File(char *fn);
		void		normalizeL1();
		void		normalizeL2();
};


