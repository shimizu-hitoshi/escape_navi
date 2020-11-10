#include "simulator.h"
#include "cdef.h"
//#define	DEBUG
	
simulator sim;
void init(int argc, char** argv){
    sim.initialize(argc, argv);
	char bin[128];
	sprintf(bin, "%s/init.bin", sim.output);
	sim.save_binary(bin);
}
void init_restart(int argc, char** argv){
	sim.reset_condition();
	sim.sset.nTime	= 0;
	sim.sset.eTime	= 3600;
	sim.graph.clear();
	delete [] sim.uf;
	sim.st.clear();

    sim.sset.alive.clear();
    sim.sset.zombie.clear();
    sim.traffic_logs.clear();

	sim.bpt =0;
	delete sim.born;
 
	//for(unsigned int j = 0; j < sim.st.size(); j++){
	//	delete [] sim.st[j].dijkstra;
	//}
    //delete [] sim.sp;
	
    sim.initialize(argc, argv);
#ifdef DEBUG
	printf("NTIME %d\n", sim.sset.nTime);
    for(int i = 0; i <5; i++){ 
		printf("i:%d t:%f\n",i+1, sim.born[i].second );
    }
#endif
}
void restart(){
	sim.sset.nTime	= 0;
	sim.sset.eTime	= 3600;
	sim.graph.clear();
	delete [] sim.uf;
	sim.st.clear();

    sim.traffic_logs.clear();

	//sim.kanban.clear();

	char bin[128];
	sprintf(bin, "%s/init.bin", sim.output);
	sim.load_binary(bin);
	sim.reset_condition();
	sim.bpt =0;
#ifdef DEBUG
	printf("NTIME %d\n", sim.sset.nTime);
    for(int i = 0; i <5; i++){ 
		printf("i:%d t:%f\n",i+1, sim.born[i].second );
    }
#endif
    for(sim.bpt=0; sim.bpt < sim.N; sim.bpt++){ 
       if(sim.born[sim.bpt].second >= sim.sset.nTime) break;
    }
 
	for(unsigned int j = 0; j < sim.st.size(); j++){
		delete [] sim.st[j].dijkstra;
		sim.graph.dijkstra(sim.st[j]);
	}
  
    delete [] sim.sp;
    sim.sp  = new stop_point[sim.M];
    for(int i = 0; i < sim.M; i++){
        sim.sp[i].id    = i;
        sim.graph.dijkstra(sim.sp[i]);
    }
}


int	setStop(int t){
	if(sim.sset.nTime > t){
		printf("sTime %d > %d \n", sim.sset.nTime, t);
		return 1;
	}
	sim.sset.eTime	=t;
	return 0;
}

void	iterate(){
    while(sim.iterate()){}
}

int	cntDest(int node, double radius){
	int c	= 0;
	for(int i = 0; i <(int) sim.sset.alive.size(); i++){
		int u	= sim.sset.alive[i];
		if(sim.uf[u].edge->dest != node){
			continue;
		}
		if(sim.uf[u].quata > radius) continue;
		c++;
	}
	return c;
}	

int	cntSrc(int node, double radius){
	int c	= 0;
	for(int i = 0; i <(int) sim.sset.alive.size(); i++){
		int u	= sim.sset.alive[i];
		if(sim.uf[u].edge->src != node){
			continue;
		}
		if(sim.uf[u].edge->dist-sim.uf[u].quata > radius) continue;
		c++;
	}
	return c;
}	

int cntOnEdge(int fr, int to){
	int c	= 0;
	for(int i = 0; i <(int) sim.sset.alive.size(); i++){
		int u	= sim.sset.alive[i];
		if(sim.uf[u].edge->src != fr  || sim.uf[u].edge->dest != to){
			continue;
		}
		c++;
	}
	return c;
}




void	setBomb( char *fn){
	sim.tb.readBombs(fn);
	sim.tb.removePastbombs(sim.sset.nTime);
	sim.tb.sort();
}

void	setBombDirect(char *text){
	int     pt, chk;
    char    buff[STR_LENGTH];
    time_bomb::bomb b;
    chk = sscanf(text, "%d%n", &b.time,&pt);
	if(chk<0) return;
    text+=pt;
    b.time--;
    strcpy(b.params, text);
    sim.tb.bombs.push_back(b);

}

void save_ulog(char *fn){
	sim.save_userlog(fn, sim.uf);
}

void save_odlog(char *fn){
	sim.save_od_hist(fn);
}

int	goalAgentCnt(int stime, int etime, int node){
	int c	= 0;
	for(int i = 0; i <(int) sim.N; i++){
		if(
			   sim.uf[i].dead < 0
			|| sim.uf[i].dead < stime
			|| sim.uf[i].dead >etime
		)
			continue;
		if(node >=0 && node != sim.uf[i].logs.back().start)
			continue;
		c++;
	}
	return c;
}
int	goalAgent(int stime, int etime, int node, int n,  int result[][3]){
	int c	= 0;
	for(int i = 0; i <(int) sim.N; i++){
		if(
			   sim.uf[i].dead < 0
			|| sim.uf[i].dead < stime
			|| sim.uf[i].dead >etime
		)
			continue;
		if(node >=0 && node != sim.uf[i].logs.back().start)
			continue;
		result[c][0]	= i;
		result[c][1]	= sim.uf[i].dead- sim.uf[i].born;
		result[c][2]	= sim.uf[i].logs.back().start;
		c++;
		if(n==c)return -1;
	}
	return c;
}

int chgDestAgentCnt(int stime, int etime, int node1, int node2, int dest){
	int c = 0;
	for(int i = 0; i < (int) sim.od_logs.size(); i++){
		if(
				sim.od_logs[i].time < stime
			||	sim.od_logs[i].time > etime
			|| (node1>=0 && node1 != sim.od_logs[i].node)
			|| (node2>=0 && node2 != sim.od_logs[i].node2)
			|| (dest>=0  && dest  != sim.od_logs[i].dest)
		)
			continue;
		c++;
	}
	return c;
}


int chgDestAgent(int stime, int etime, int node1, int node2, int dest, int n, int result[][6]){
	int c = 0;
	for(int i = 0; i < (int) sim.od_logs.size(); i++){
		if(
				sim.od_logs[i].time < stime
			||	sim.od_logs[i].time > etime
			|| (node1>=0 && node1 != sim.od_logs[i].node)
			|| (node2>=0 && node2 != sim.od_logs[i].node2)
			|| (dest>=0  && dest  != sim.od_logs[i].dest)
		)
			continue;
		result[c][0]	= sim.od_logs[i].time;
		result[c][1]	= sim.od_logs[i].uid;
		result[c][2]	= sim.od_logs[i].node;
		result[c][3]	= sim.od_logs[i].node2;
		result[c][4]	= sim.od_logs[i].dest;
		result[c][5]	= sim.od_logs[i].grp;
		c++;
		if(n==c)return -1;
	}
	return c;
}

//
//
//int main(int argc, char** argv){
//
//	simulator sim;
//    sim.initialize(argc, argv);
//    
//    while(sim.iterate()){}
//    
//	char	filename[STR_LENGTH];
//	sprintf(filename, "%s/userlogs.txt",sim.output);
//	sim.save_userlog(filename, sim.uf);
//	sprintf(filename, "%s/navilogs.txt", sim.output);
//	sim.save_navi(filename, sim.navi_index, sim.navi_log);
//
//	sprintf(filename, "%s/trainlogs.txt",sim.output);
//	sim.save_trainlog(filename);
//
//	sprintf(filename, "%s/odlogs.txt",sim.output);
//	sim.save_od_hist(filename);
//
//	sprintf(filename, "%s/signage_logs.txt",sim.output);
//	sim.kanban.save_log(filename);
//
//	if(strcmp(sim.output_resume, ""))
//		sim.save_binary(sim.output_resume);
//
//	if(strcmp(sim.output_dijkstra, ""))
//		writeDijkstra_map(sim.output_dijkstra, sim.st);
//	
//		sim.shrinkTrafficRegulation();
//    if(strcmp(sim.output_traffic, ""))
//		sim.writeTrafficLog(sim.output_traffic);
//
///*
//	sim.run();
//	sim.savelogs();
//	sim.analyse();
//	return 0;
//*/
//}
