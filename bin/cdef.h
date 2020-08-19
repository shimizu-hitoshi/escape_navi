#ifndef __hello
#define __hello

#ifdef __cplusplus
extern "C"{
#endif
void 	init(int argc, char** argv);
int		setStop(int t);
void	iterate();
int		cntDest(int node, double radius);
int		cntSrc(int node, double radius);
void	setBomb( char *fn);
int		cntOnEdge(int fr, int to);
void    setBombDirect(char *text);
void    restart();

void    init_restart(int argc, char** argv);
void	save_ulog(char *fn);
int		goalAgentCnt(int stime, int etime, int node);
int		goalAgent(int stime, int etime,int n,  int result[][3]);
#ifdef __cplusplus
}
#endif

#endif

