#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "simulator.h"
#include <iostream>
//#include <algorithm>
#include <getopt.h>

#define	NAVI		2		//経路選択指針	1:最短経路	2:Dijkstra確率分岐
//#define	NAVI		1		//経路選択指針	1:最短経路	2:Dijkstra確率分岐
#define TRANSITION	4.0
#define MINIMUM_VIEW	1.0

//#define	TERMINATE_MODE	//終了モード
#define	BORN_TROUBLE	1	//発生時トラブル対応 1:ランダムジャンプ 2:強制終了
#define BORN_DELAY		10

//#define	TIGHT_PATH

//#define	DEBUG

//乱数生成アルゴリズム
//0 < x <1	一様乱数
double Uniform( void ){
	return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
}
//正規分布
double rand_normal( double mu, double sigma ){
	double z=sqrt( -2.0*log(Uniform()) ) * sin( 2.0*M_PI*Uniform() );
	return mu + sigma*z;
}


//エージェント情報読み込み部
void	simulator::readData(char *fn){
	FILE	*fp;
	int	i;
	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	fscanf(fp,"%d %d", &N, &G);
	uf	= new user_info [N];
	for( i = 0; i < N; i++){
		fscanf(fp, "%d %d %lf %d %d %d %d", &uf[i].id, &uf[i].born, &uf[i].speed, &uf[i].from, &uf[i].goal, &uf[i].group, &uf[i].tgoal);
		uf[i].id--;
		uf[i].born--;
		uf[i].from--;
		uf[i].goal--;
		uf[i].to	= uf[i].from;
		uf[i].group--;
		uf[i].tgoal--;
		uf[i].tid = -1;
		uf[i].ttime = -1;
		uf[i].quata	= 0.0;
		uf[i].edge	= NULL;
		int v_len;			//経由地長
		fscanf(fp, "%d", &v_len);
		for(int j = 0; j < v_len; j++){
			std::pair<int , int>	via;
			fscanf(fp, "%d", &via.first);
			via.first--;
			via.second	= -1;
			uf[i].via.push_back(via);
		}
	}
	fclose(fp);
}

void	simulator::readDiagram(char *fn){

	FILE	*fp;
	char	buff[STR_LENGTH];

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}

	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		if(buff[0]=='#')continue;
		dia.addTrain(buff, sset.nTime);
	}
	fclose(fp);

}

//駅情報読み込み部
void	simulator::readStation(char *fn){
	FILE	*fp;
	int		j, size, pt, chk;
	char	buff[STR_LENGTH];
	char	*bpt;
	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		bpt = buff;
		station s;
		chk = sscanf(buff, "%d %d %d%n", &s.id, &s.capa, &size, &pt);
		if(chk <0){printf("read station error\n");exit(1);}
		bpt+=pt;
		for(j = 0; j < size;j++){
			station::entrance e;
			chk=sscanf(bpt,"%d:%d%n", &e.id, &e.process, &pt);
			if(chk <0){printf("read station error\n");exit(1);}
			e.id--;
			s.entrances.push_back(e);
			bpt+=pt;
		}
		st.push_back(s);
	}
}

//誘導情報読み込み部
void simulator::readNavigation(char *fn){
	FILE	*fp;
	int	i, l, len, target, grp, t;

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	fscanf(fp, "%d", &len);
	for(i = 0; i < len; i++){
		fscanf(fp, "%d %d %d", &target, &grp, &l);
		target--;
		grp--;
		navigation::condition c;
		c.node	= target;
		c.grp	= grp;
		for(t = 0; t < l; t++){
			navigation::values	v;
			fscanf(fp, "%d:%lf", &v.dest, &v.prob);
			v.dest--;
			navi.addNavigation(c, v);
		}

	}
	fclose(fp);
}

//交通規制読み込み部
void simulator::readTrafficRegulation(char *fn){
	FILE	*fp;
	int		i, l;
	int		from, to, tp;
	double	val;
	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	fscanf(fp,"%d", &l);

	for(i = 0; i < l; i++){
		fscanf(fp, "%d %d %d", &from, &to, &tp);
		from--;
		to--;
		if(tp	== 2){
			fscanf(fp, "%lf", &val);
		}else{
			val = -1;
		}
		editTrafficRegulation(from, to, tp, val);
	}
	fclose(fp);

}
void time_bomb::readBombs(char *fn){
	int		pt, chk;
	char	buff[STR_LENGTH];
	char	*bpt;
	reader *rd = new reader(fn);
	while(rd->readNext(buff)){
		bpt = buff;
		bomb b;
		chk = sscanf(buff, "%d%n", &b.time,&pt);
		if(chk<0) continue;
		bpt+=pt;
		b.time--;
		strcpy(b.params, bpt);
		bombs.push_back(b);
	}
	delete rd;
}

void simulator::bombExplode(){
	char *buff;
	int pt;
	char type[256];
	bool	navi_chg	= false;
	while((buff=tb.chkBomb(sset.nTime))!=NULL){
		sscanf(buff, "%s%n", type, &pt);
#ifdef DEBUG
		printf("%d\t%s\n", sset.nTime+1, type);
#endif
		buff += pt;
		if(strcmp(type, "traffic_regulation")==0){
			int from ,to, tp;
			double val;
			sscanf(buff, "%d %d %d %lf", &from, &to, &tp, &val);
			from--;
			to--;
			if(tp ==0 || tp==1){val = -1.0;}
			editTrafficRegulation(from, to, tp, val);
			//通行止めであろうと解除だろうと再計算
			for(unsigned int j = 0; j < st.size(); j++){
				delete [] st[j].dijkstra;
				graph.dijkstra(st[j]);
			}
			for(int j = 0; j < M; j++){
				delete [] sp[j].dijkstra;
				graph.dijkstra(sp[j]);
			}
		}else if(strcmp(type, "set_navi")==0){
			navi.addNavigation(buff);
			navi_chg	= true;
		}else if(strcmp(type, "clear_navi")==0){
			navi.clear();
			navi_chg		= false;		//clearした誘導は登録不要
			sset.navi_flg	= -1;			
		}else if(strcmp(type, "signage")==0){
			kanban.addSignage(buff);
		}else if(strcmp(type, "del_signage_element")==0){
			kanban.delSignage(buff);
		}else if(strcmp(type, "del_signage_table")==0){
			kanban.conditional_clear(buff);
		}else if(strcmp(type, "clear_signage")==0){
			kanban.clear_signage(buff);
		}else if(strcmp(type, "clear_all_signage")==0){
			kanban.clear();
		}else if(strcmp(type, "wait_on_edge")==0){
			int src, dest, grp, wait, gate;
			int ind;
			//gate!=0: gateにトラップされているエージェントは加算しない
			sscanf(buff, "%d %d %d %d %d", &src, &dest, &grp, &wait, &gate);
			src--;
			dest--;
			grp--;
			ind = -1;
			for(int i = 0; i < graph.vec[src]; i++){
				if(graph.edge[src][i].dest==dest){
					ind = i;break;
				}
			}
			if(ind== -1)return;
			for(unsigned int i = 0; i < graph.edge[src][ind].user.size(); i++){
				int u = graph.edge[src][ind].user[i];
				if(uf[u].group != grp && grp != -1) continue;
				if(uf[u].gate && gate != 0) continue;
				uf[u].wait+=wait;
			}
		}else if(strcmp(type, "set_gate")==0){
			graph.setGate(buff);
			chkGate();
		}else if(strcmp(type, "delete_gate")==0){
			graph.delGate(buff);
			chkGate();
		}else if(strcmp(type, "clear_all_gate")==0){
			graph.resetGate();
			chkGate();
		}else if(strcmp(type, "broadcast_chg_dest")==0){
		//	printf("%s\n", buff);
			changeDestination(buff);
		}else if(strcmp(type, "edit_train")==0){
			dia.addTrain(buff, sset.nTime);
		}else if(strcmp(type, "edit_signal")==0){
			sig.addSignal(buff);
		}else if(strcmp(type, "del_signal")==0){
			sig.delSignal(buff);
		}else if(strcmp(type, "setAlpha")==0){
			graph.setAlpha(buff);
		}else if(strcmp(type, "setBeta")==0){
			graph.setBeta(buff);
		}else if(strcmp(type, "setGamma")==0){
			graph.setGamma(buff);
		}else if(strcmp(type, "set_edgeCost")==0){
			graph.setEdgeCost(buff);
			//再計算
			for(unsigned int j = 0; j < st.size(); j++){
				delete [] st[j].dijkstra;
				graph.dijkstra(st[j]);
			}
			for(int j = 0; j < M; j++){
				delete [] sp[j].dijkstra;
				graph.dijkstra(sp[j]);
			}
		}else if(strcmp(type, "clear_edgeCost")==0){
			graph.resetEdgeCost();
			//再計算
			for(unsigned int j = 0; j < st.size(); j++){
				delete [] st[j].dijkstra;
				graph.dijkstra(st[j]);
			}
			for(int j = 0; j < M; j++){
				delete [] sp[j].dijkstra;
				graph.dijkstra(sp[j]);
			}
		}
	}
	if(navi_chg){
		navi_log.push_back(navi);
		sset.navi_flg	= (int) navi_log.size();
	}
}

void	simulator::changeDestination(char *buff){
		int pt;
		int grp;
		int flg;	//登場済みのみか、未登場含むか
		int chk, l, d;
		std::pair<int, int> p;
		std::vector<signage::values>	val;
		signage::values					v;
		sscanf(buff, "%d %d%n", &grp, &flg, &pt);
		grp--;
		buff+=pt;
		while((chk=sscanf(buff, "%d %d %lf %d%n", &v.dest, &v.grp, &v.prob, &l, &pt))>0){

			printf("%d %d %lf %d\n", v.dest, v.grp, v.prob, l);
			buff+=pt;
			v.dest--;
			v.grp--;
			v.keep_via	= (l < 0);
			v.via.clear();
			for(int i = 0; i < l; i++){
				sscanf(buff,"%d%n", &d, &pt );
				d--;
				v.via.push_back(d);
				buff+=pt;
			}
			val.push_back(v);
		}
		for(int u = 0; u < N; u++){
			if(uf[u].dead>=0) continue; //既に駅に到達済み
			if(flg==0 && uf[u].born >= sset.nTime) continue;	//未発生
			if(uf[u].group != grp && grp != -1) continue;	//グループ判定
			double r = Uniform();
			for(unsigned int i = 0; i < val.size(); i++){
				v	= val[i];
				r	-= v.prob;
				if(r <=0.0){
					uf[u].goal = v.dest;
					if(v.grp>=0) uf[u].group = v.grp;
					if(v.keep_via==false){
						while(!uf[u].via.empty()){
							if(uf[u].via.back().second < 0){
								uf[u].via.pop_back();
								continue;
							}
							break;
						}
						for(unsigned int j=0; j < v.via.size(); j++){
							p.first = v.via[j];
							p.second	= -1;
							uf[u].via.push_back(p);
						}
					}
					od_change_log l;
					l.uid	= u;
					l.time	= sset.nTime;
					l.dest	= v.dest;
					l.node	= uf[u].from;
					l.node2	= uf[u].to;
					l.grp	= v.grp;
					l.via	= v.keep_via;
					od_logs.push_back(l);


					break;
				}
			}
		}
}

void simulator::editTrafficRegulation(int from, int to, int tp, double val){
	int f = graph.setRegulation(from, to, tp, val);
	 if(f==1){
		traffic_log log;
		log.time	= sset.nTime;
		log.from	= from;
		log.to		= to;
		log.type	= tp;
		log.value	= val;
		traffic_logs.push_back(log);
	}else if(f==-1){
		traffic_log log;
		log.time	= sset.nTime;
		log.from	= from;
		log.to		= to;
		log.type	= 0;
		log.value	= -1;
		traffic_logs.push_back(log);

	}
}

void simulator::resetTrafficRegulation(){
	int from, to, tp;
	double val;
	tp	= 0;
	val	= -1;
	for(from = 0; from < graph.M; from++){
		for(int j = 0; j < graph.vec[from]; j++){
			to	= graph.edge[from][j].dest;

			editTrafficRegulation(from, to, tp, val);
		}
	}
}

void simulator::updateStation(){
	std::vector<int> mundane;
	int s, d;
	int uid, ind;
	dia.setCurrent(sset.nTime);
	for(unsigned int i = 0; i < sset.zombie.size();i++){
		uid = sset.zombie[i];
		s	= uf[uid].goal;
		d	= uf[uid].tgoal;
		ind	= dia.chkRide(s, d);
		if(ind < 0){
			mundane.push_back(uid);
			continue;
		}
		uf[uid].tid = dia.diagrams[ind].tid;
		uf[uid].ttime = sset.nTime;
		st[s].num--;
		dia.diagrams[ind].rideOn();
	}
	sset.zombie.clear();
	sset.zombie	= mundane;
	mundane.clear();
}

//
bool tmsort(traffic_log a, traffic_log b){
	if(a.time < b.time){
		return true;
	}
	return false;
}

bool pairsort(traffic_log a, traffic_log b){
	if(a.from < b.from) return true;
	if(a.from == b.from && a.to < b.to) return true;
	return false;
}
void simulator::shrinkTrafficRegulation(){
	//事前に時間順にソートする
	//stable_sortであることが重要
	std::stable_sort(traffic_logs.begin(), traffic_logs.end(), tmsort);
	std::stable_sort(traffic_logs.begin(), traffic_logs.end(), pairsort);
	//同じ時間なら最新の変更を残して削除
	for(int i = traffic_logs.size()-2; i >= 0; i--){
		if( traffic_logs[i].time == traffic_logs[i+1].time
		&&	traffic_logs[i].from == traffic_logs[i+1].from
		&&	traffic_logs[i].to == traffic_logs[i+1].to
		){
			traffic_logs.erase(traffic_logs.begin()+i);
		}
	}
	//同じ変更内容ならもっとも古い時間の変更を残して削除
	for(int i = traffic_logs.size()-2; i >= 0; i--){
		if( traffic_logs[i].from == traffic_logs[i+1].from
		&&	traffic_logs[i].to == traffic_logs[i+1].to
		&&	traffic_logs[i].type == traffic_logs[i+1].type
		&&	traffic_logs[i].value == traffic_logs[i+1].value
		){
			traffic_logs.erase(traffic_logs.begin()+i+1);
		}
	}

	for(int i = traffic_logs.size()-1; i >= 0; i--){
		if( traffic_logs[i].type == 0 && i ==0){
			traffic_logs.erase(traffic_logs.begin());
		}else if(traffic_logs[i].type == 0
		&&	(traffic_logs[i].from != traffic_logs[i-1].from ||	traffic_logs[i].to != traffic_logs[i-1].to))
		{
			traffic_logs.erase(traffic_logs.begin()+i);
		}

	}

	std::stable_sort(traffic_logs.begin(), traffic_logs.end(), tmsort);


}
void simulator::chkSignage(user_info &u){
	signage::values v;
	int to	= u.to;
	v = kanban.chkSignage(u.to, u.group);
	std::pair<int, int> p;
	if(v.dest >= 0 ){
		u.goal = v.dest;
		if(v.grp >=0) u.group	= v.grp;
		if(v.keep_via == false){
			while(!u.via.empty()){
				if(u.via.back().second < 0){
					u.via.pop_back();
					continue;
				}
				break;
			}
			for(unsigned int j = 0; j < v.via.size(); j++){
				p.first		= v.via[j];
				p.second	= -1;
				u.via.push_back(p);
			}
		}
		od_change_log l;
		l.uid	= u.id;
		l.time	= sset.nTime;
		l.dest	= v.dest;
		l.node	= to;
		l.node2	= -1;
		l.grp	= v.grp;
		l.via	= v.keep_via;
		od_logs.push_back(l);
	}
}

void simulator::chkGate(){
	for(int i = 0; i < M; i++){
		for(int j = 0; j < graph.vec[i]; j++){
			graph.edge[i][j].g_cnt	= 0;
			if(graph.edge[i][j].g_capa >= 0){
				for(unsigned int k = 0; k < graph.edge[i][j].user.size(); k++){
					int u = graph.edge[i][j].user[k];
					if(uf[u].gate){
						graph.edge[i][j].g_cnt++;
					}
				}
			}else{
				for(unsigned int k = 0; k < graph.edge[i][j].user.size(); k++){
					int u = graph.edge[i][j].user[k];
					if(uf[u].gate){
						////////////////////////
						//ゲート関係なく滞留させられているエージェントも考えられるので
						//ゲートフラグ立っているエージェントのみ開放
						////////////////////////
						uf[u].gate = false;
						uf[u].wait = 0;
					}
				}
			}
		}
	}
}

//シミュレーションマップ読み込み部
void graph_tensor::readGraph(char *fn){
	FILE	*fp;
	int	i, j;

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	fscanf(fp,"%d %d", &M, &j);
	vec		= new int[M];
	edge		= new edge_info *[M];

	height		= new double[M];
	for (i = 0; i < M; i++){
		fscanf(fp,"%d", &vec[i]);
		edge[i]		= new edge_info[vec[i]];
		height[i]	= 0.0;
		for(j = 0; j < vec[i]; j++){
			fscanf(fp,"%d:%lf:%lf", &edge[i][j].dest, &edge[i][j].width, &edge[i][j].dist);
			edge[i][j].src	= i;
			edge[i][j].dest--;
			edge[i][j].flg.first	= 0;
		}
	}

	rev_check();
}

void	graph_tensor::readGate(char *fn){
	FILE	*fp;
	char	buff[STR_LENGTH];
	int		from, to, capa, wait;
	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		sscanf(buff, "%d %d %d %d", &from, &to, &capa, &wait);
		from--;
		to--;
		setGate(from, to, capa,wait);
	}
	fclose(fp);
}

void	graph_tensor::readPoints(char *fn){
	FILE	*fp;
	int	i, j;
	double lat, lng;

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	flg_points	= 1;
	points		= new coord[M];
	for (i = 0; i < M; i++){
		fscanf(fp,"%d %lf %lf", &j, &lng, &lat);
		points[i].lat	= lat;
		points[i].lng	= lng;
	}
	fclose(fp);
}

void	graph_tensor::readHeight(char *fn){
	FILE	*fp;
	int	i, j;
	double	v;

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	flg_height	= 1;
	for (i = 0; i < M; i++){
		fscanf(fp,"%d %lf", &j, &v);
		height[j-1]	= v;
	}
	fclose(fp);
}

double	graph_tensor::deg2rad(double deg){
	return (deg *(2*M_PI)/ 360.0);
}
double	graph_tensor::Hubeny(double lat1, double lng1, double lat2, double lng2){
		double	a = 6378137.000;			//赤道半径
		double	b = 6356752.314140;			//極半径
		double	e = sqrt( (a*a - b*b) / pow(a,2));
		double	e2 = pow(e,2);
		double	mnum =a * (1 - e2);

		double my =deg2rad((lat1+lat2) /2.0);
		double dy =deg2rad( lat1-lat2);
		double dx =deg2rad( lng1-lng2);
		double s =sin(my);
		double w =sqrt(1.0-e2 * s *s);
		double m =mnum /(w *w *w);
		double n =a/w;
		double dym =dy*m;
		double dxncos=dx*n*cos(my);
		return( sqrt( pow(dym,2) + pow(dxncos,2)) );

}

void	graph_tensor::readCurve(char *fn){
	FILE	*fp;
	int		i, j, k;
	double	lat, lng;
	char	buff[STR_LENGTH];
	int		from, to, pt;
	char	*bpt;

	//std::map <int, std::map <int, std::vector<int> > > mapmap;

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		sscanf(buff, "%d\t%d%n", &from, &to, &pt);
		from--;
		to--;
		bpt = buff;
		bpt+= pt;
/*
		while(1){
			while(1){
				if(bpt[0] == '\0' ||	bpt[0] == '[')break;
				bpt++;
			}
			if(bpt[0]=='\0')break;
			bpt++;
			sscanf(bpt, "%lf, %lf%n", &lat, &lng, &pt);
			printf("%d %d %f, %f\n", from +1, to+1, lat, lng);
			//mapmap[from][to].push_back(-1);
			curve cv;
			cv.lat	= lat;
			cv.lng	= lng;
			curves[from][to].push_back(cv);
		}
*/
/**/
		for(int i = 0; i < STR_LENGTH; i++){
			if(bpt[i]=='\0')break;
			if(bpt[i]==',')bpt[i]=' ';
			if(bpt[i]=='[')bpt[i]=' ';
			if(bpt[i]==']')bpt[i]=' ';
		}
		while(1){
			int chk = sscanf(bpt, "%lf %lf%n",&lat, &lng, &pt);
			bpt+=pt;
			if(chk == 2){
				curve cv;
				cv.lat	= lat;
				cv.lng	= lng;
				curves[from][to].push_back(cv);
				continue;
			}
			break;
		}
/**/
	}
	fclose(fp);
	for(i = 0; i < M; i++){
		for(j = 0; j < vec[i]; j++){
			to	= edge[i][j].dest;
			printf("%d %d\n", i+1, to+1);
			if(curves[i][to].size()==0){
			}
			curves[i][to][0].dist	 = 0.0;
			for(k=1; k <(int) curves[i][to].size(); k++){

				//printf("%d %d %f, %f\n", i+1, to+1, curves[i][to][k].lat, curves[i][to][k].lng);
				curves[i][to][k].dist	= Hubeny(curves[i][to][k-1].lat, curves[i][to][k-1].lng, curves[i][to][k].lat, curves[i][to][k].lng);
			}
		}
	}
}

void	graph_tensor::readAlpha(char *fn){
	FILE	*fp;
	int		from, to;
	double	val;
	char	buff[STR_LENGTH];

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	clearAlpha();
	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		sscanf(buff, "%d\t%d\t%lf", &from, &to, &val);
		from--;
		to--;
		setAlpha(from, to, val);
	}
	fclose(fp);
}


void	graph_tensor::readGamma(char *fn){
	FILE	*fp;
	int		from, to;
	double	val;
	char	buff[STR_LENGTH];

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	clearGamma();
	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		sscanf(buff, "%d\t%d\t%lf", &from, &to, &val);
		from--;
		to--;
		setGamma(from, to, val);
	}
	fclose(fp);
}

void	graph_tensor::readEdgeCost(char *fn){
	FILE	*fp;
	char	buff[STR_LENGTH];
	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	resetEdgeCost();
	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		setEdgeCost(buff);
	}
	fclose(fp);
}

//Dijkstra計算部
//Dijkstraはエージェントが目的地へ向かう際、経路を決める指標に用いる
void	graph_tensor::dijkstra(station &st){
	int	i, j, from, to;
	int	l, *list, n, *next;
	double	*value;
	int depth		= 0;
	value			= new double[M];
	list			= new int[M];
	next			= new int[M];

	for(i = 0; i < M; i++)
		value[i] = DBL_MAX;

	l	= st.entrances.size();
	for(i = 0; i < l; i++){
		value[st.entrances[i].id]=0.0;
		list[i]	= st.entrances[i].id;
	}

	//有向重みつきInverse Matrix
/**/
	graph_tensor	rgraph	= inverse_graph();
	double	v;
	while(1){
		n = 0;
		depth++;
		for(i = 0; i < l; i++){
			from	= list[i];
			for(j = 0; j < rgraph.vec[from]; j++){
				to	= rgraph.edge[from][j].dest;
				if(rgraph.edge[from][j].flg.first==1) continue;		//進入禁止
				v = value[from] + rgraph.edge[from][j].dist * rgraph.edge[from][j].e_cost;
				if(value[to] > v){
					value[to] = v;
					next[n++] = to;
				}
			}
		}
		if(n==0)
			break;
		for(i = 0; i < n; i++) list[i]=next[i];
		l = n;
	}

	if(depth>diameter) diameter = depth;
	rgraph.clear();
/**/
	delete	[] list;
	delete	[] next;
	st.len_dijkstra	= M;
	st.dijkstra	= value;

	//目的地から非接続だった時、エラーを吐いて終了する
	// for(i = 0; i < M; i++)
	// 	if(value[i] == DBL_MAX){
	// 		fprintf(stderr, "No route station(%d) was found \n", i+1);
	// 		//exit(1);
	// 	}
}

//dijkstra経由地Ver
void	graph_tensor::dijkstra(stop_point &st){
	int	i, j, from, to;
	int	l, *list, n, *next;
	double	*value;
	int depth		= 0;
	value			= new double[M];
	list			= new int[M];
	next			= new int[M];

	for(i = 0; i < M; i++)
		value[i] = DBL_MAX;


	value[st.id]=0.0;
	list[0]	= st.id;
	l	= 1;

	//有向重みつきInverse Matrix
/**/
	graph_tensor	rgraph	= inverse_graph();
	double v;
	while(1){
		n = 0;
		depth++;
		for(i = 0; i < l; i++){
			from	= list[i];
			for(j = 0; j < rgraph.vec[from]; j++){
				to	= rgraph.edge[from][j].dest;
				if(rgraph.edge[from][j].flg.first==1) continue;		//進入禁止
				v = value[from] + rgraph.edge[from][j].dist * rgraph.edge[from][j].e_cost;
				if(value[to] > v){
					value[to] = v;
					next[n++] = to;
				}
			}
		}
		if(n==0)
			break;
		for(i = 0; i < n; i++) list[i]=next[i];
		l = n;
	}
	if(depth>diameter) diameter = depth;
	rgraph.clear();
/**/
	delete	[] list;
	delete	[] next;
	st.len_dijkstra	= M;
	st.dijkstra	= value;


	//非接続だった時、エラーを吐いて終了する
	// for(i = 0; i < M; i++)
	// 	if(value[i] == DBL_MAX){
	// 		fprintf(stderr,"No route station was found\n");
	// 		//exit(1);
	// 	}
}


//交差点にさしかかった際、次の進行方向(道路)を決定する
//edge_info*	graph_tensor::relay(int p, station st, int grp, navigate navi){
edge_info*	graph_tensor::relay(int p, station st, int grp, navigation navi){
#if NAVI	== 1
	int		i, n;
	double		v, w;
	edge_info	*e;
	e=NULL;
	if(st.dijkstra[p]==DBL_MAX){
		for(i = 0, n = 0; i < vec[p]; i++){
			if(edge[p][i].flg.first == 1){
				continue;
			}
			n++;
		}
		n = random() % n;
		for(i = 0; i < vec[p]; i++){
			if(edge[p][i].flg.first == 1){
				continue;
			}
			n--;
			if(n<0){
				 e = &edge[p][i];
				break;
			}
		}
	}
	for(i = 0, v = DBL_MAX; i < vec[p]; i++){
		n = edge[p][i].dest;
		w = edge[p][i].dist;

		//次の経由地が進入禁止であれば却下
		if(edge[p][i].flg.first == 1){
			continue;
		}

		if(st.dijkstra[n] + w < v){
			v	= st.dijkstra[n]+w;
			e	= &edge[p][i];
		}
	}
#elif NAVI	== 2
	int		i, n;
	int		warning	= 0;
	double		v, w;
	double	 	*candi;
	edge_info	*e;

	e=NULL;
	candi		= new double[vec[p]];
	for(i = 0, v = 0.0; i < vec[p]; i++){
		n		= edge[p][i].dest;
		//次の経由地が現在よりも遠ければ却下
		if(st.dijkstra[n] > st.dijkstra[p]){
			candi[i] = -1.0;
			continue;
		}
		//次の経由地が進入禁止であれば却下
		if(edge[p][i].flg.first == 1){
			candi[i] = -1.0;
			continue;
		}
		//誘導がかかっていれば誘導に従い、そうでなければ距離を選択確率に採用
		if(navi.check_navi(p, grp)){
			candi[i]	= navi.get_val(p, n, grp);
		}else if(st.dijkstra[n]== DBL_MAX){
			warning	= 1;
#ifdef	TERMINATE_MODE
			exit(1);
#endif
			//DBL_MAXでありながらここまでくると、他の候補もDBL_MAXだと予想されるので適当な値に変換してオーバーフローしないようにする
			candi[i]	= 999999.9;
		}else{
			w		= edge[p][i].dist * edge[p][i].e_cost;
			candi[i]	= st.dijkstra[n] + w ;
			candi[i]	= 1.0 / (candi[i]+DBL_MIN);
		}
		v		+= candi[i];
	}

	if(v==0.0){	//誘導が全滅した場合は進行可能方向からランダムに選択
		for(i = 0; i < vec[p]; i++){
			if(candi[i] < 0.0) continue;
			n		= edge[p][i].dest;
			if(st.dijkstra[n]==DBL_MAX){
				candi[i]	= 999999.9;
			}else{
				w			= edge[p][i].dist * edge[p][i].e_cost;
				candi[i]	= st.dijkstra[n] + w ;
				candi[i]	= 1.0 / (candi[i]+DBL_MIN);
			}
			v			+= candi[i];
		}
	}

	if(warning	==1){
		fprintf(stderr,"goal %d is disconnected from %d!\n", st.id+1, p+1);
	}
	//確率にしたがって進行方向の決定
	w	= Uniform() * v;
	for(i = 0; i < vec[p]; i++){
		if(candi[i] < 0.0) continue;
		w -= candi[i];
		if(w <= 0.0){
			e	= &edge[p][i];
			break;
		}
	}

	delete [] candi;
#else
	int		i, n;
	double		v, w;
	edge_info	*e;
	for(i = 0, v = 0.0; i < vec[p]; i++){
		n = edge[p][i].dest;
		w = st.dijkstra[p] -st.dijkstra[n];
		if(w > 0.0){
			v += w;
		}
	}
	v = Uniform() * v;
	for(i = 0; i < vec[p]; i++){
		n = edge[p][i].dest;
		w = st.dijkstra[p] -st.dijkstra[n];
		if(w > 0.0){
			v -= w;
			e	= &edge[p][i];
			if(v <=0.0)break;
		}
	}

#endif
	return e;
}


//交差点にさしかかった際、次の進行方向(道路)を決定する経由地Ver
edge_info*	graph_tensor::relay(int p, stop_point st, int grp, navigation navi){
#if NAVI	== 1
	int		i, n;
	double		v, w;
	edge_info	*e;
	e=NULL;
	if(st.dijkstra[p]==DBL_MAX){
		for(i = 0, n = 0; i < vec[p]; i++){
			if(edge[p][i].flg.first == 1){
				continue;
			}
			n++;
		}
		n = random() % n;
		for(i = 0; i < vec[p]; i++){
			if(edge[p][i].flg.first == 1){
				continue;
			}
			n--;
			if(n<0){
				e = &edge[p][i];
				break;
			}
		}
	}
	for(i = 0, v = DBL_MAX; i < vec[p]; i++){
		n = edge[p][i].dest;
		w = edge[p][i].dist;

		//次の経由地が進入禁止であれば却下
		if(edge[p][i].flg.first == 1){
			continue;
		}

		if(st.dijkstra[n] + w < v){
			v	= st.dijkstra[n]+w;
			e	= &edge[p][i];
		}
	}
#elif NAVI	== 2
	int		i, n;
	int		warning	= 0;
	double		v, w;
	double	 	*candi;
	edge_info	*e;

	e=NULL;
	candi		= new double[vec[p]];
	for(i = 0, v = 0.0; i < vec[p]; i++){
		n		= edge[p][i].dest;
		//次の経由地が現在よりも遠ければ却下
		if(st.dijkstra[n] > st.dijkstra[p]){
			candi[i] = -1.0;
			continue;
		}
		//次の経由地が進入禁止であれば却下
		if(edge[p][i].flg.first == 1){
			candi[i] = -1.0;
			continue;
		}
		//誘導がかかっていれば誘導に従い、そうでなければ距離を選択確率に採用
		if(navi.check_navi(p, grp)){
			candi[i]	= navi.get_val(p, n, grp);
		}else if(st.dijkstra[n]== DBL_MAX){
			warning	= 1;
#ifdef	TERMINATE_MODE
			exit(1);
#endif
			//DBL_MAXでありながらここまでくると、他の候補もDBL_MAXだと予想されるので適当な値に変換してオーバーフローしないようにする
			candi[i]	= 999999.9;
		}else{
			w		= edge[p][i].dist * edge[p][i].e_cost;
			candi[i]	= st.dijkstra[n] + w ;
			candi[i]	= 1.0 / (candi[i]+DBL_MIN);
		}
		v		+= candi[i];
	}

	if(v==0.0){	//誘導が全滅した場合は進行可能方向からランダムに選択
		for(i = 0; i < vec[p]; i++){
			if(candi[i] < 0.0) continue;
			n		= edge[p][i].dest;
			if(st.dijkstra[n]==DBL_MAX){
				candi[i]	= 999999.9;
			}else{
				w			= edge[p][i].dist * edge[p][i].e_cost;
				candi[i]	= st.dijkstra[n] + w ;
				candi[i]	= 1.0 / (candi[i]+DBL_MIN);
			}
			v			+= candi[i];
		}
	}
	if(warning	==1){
		fprintf(stderr,"%d is disconnected from %d!\n", st.id+1, p+1);
	}
	//確率にしたがって進行方向の決定
	w	= Uniform() * v;
	for(i = 0; i < vec[p]; i++){
		if(candi[i] < 0.0) continue;
		w -= candi[i];
		if(w <= 0.0){
			e	= &edge[p][i];
			break;
		}
	}

	delete [] candi;
#else
	int		i, n;
	double		v, w;
	edge_info	*e;
	for(i = 0, v = 0.0; i < vec[p]; i++){
		n = edge[p][i].dest;
		w = st.dijkstra[p] -st.dijkstra[n];
		if(w > 0.0){
			v += w;
		}
	}
	v = Uniform() * v;
	for(i = 0; i < vec[p]; i++){
		n = edge[p][i].dest;
		w = st.dijkstra[p] -st.dijkstra[n];
		if(w > 0.0){
			v -= w;
			e	= &edge[p][i];
			if(v <=0.0)break;
		}
	}

#endif
	return e;
}


edge_info*	graph_tensor::relayDirect(int p, station st){
	edge_info	*e;
	e	= NULL;
	std::vector<edge_info *> candi;
	for(int i = 0; i < vec[p]; i++){
		int n		= edge[p][i].dest;
		if(edge[p][i].flg.first == 1) continue;	//進入禁止はパス
		if(st.dijkstra[n] ==0.0){
			candi.push_back(&edge[p][i]);
		}
	}
	if(candi.size()>0){
		e	= candi[random() % candi.size()];
	}
	return e;
}


edge_info*	graph_tensor::relayDirect(int p, stop_point st){
	edge_info	*e;
	e	= NULL;
	std::vector<edge_info *> candi;
	for(int i = 0; i < vec[p]; i++){
		int n		= edge[p][i].dest;
		if(edge[p][i].flg.first == 1) continue;	//進入禁止はパス
		if(st.dijkstra[n] ==0.0){
			candi.push_back(&edge[p][i]);
		}
	}
	if(candi.size()>0){
		e	= candi[random() % candi.size()];
	}
	return e;
}

void	signal::readSignal(char *fn){
	FILE	*fp;
	char	buff[STR_LENGTH];

	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		addSignal(buff);
	}
	fclose(fp);
}

bool	signal::checkSignal(int t, int from, int to, int src){
	signal::condition c;
	c.src	= src;
	c.from	= from;
	c.to	= to;
	if(signals.count(c)==0)return false;
	prop	p = signals[c];
	int f;
	f	= (t - p.gap +p.cycle) % p.cycle;
	if(f < p.blue) return false;
	return true;
}

void signage::readSignage(char *fn){
	FILE	*fp;
	char	buff[STR_LENGTH];
	if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	while(fgets(buff, STR_LENGTH, fp )!= NULL){
		if(buff[0]== '#')continue;
		addSignage(buff);
	}
	fclose(fp);
}
signage::values signage::chkSignage(int node, int grp){
	condition	c;
	values		v;
	c.node	= node;
	c.grp	= grp;
	v.dest	= -1;				//変更無し用
	if(signages.count(c)==0){	//個別グループ設定
		c.grp	= -1;			//全体設定
		if(signages.count(c)==0){
			 return v;
		}
	}

	double r = Uniform();
	for(unsigned int i = 0; i < signages[c].size(); i++){
		r -= signages[c][i].prob;
		if(r <=0.0){
			return signages[c][i];
		 }
	}
	return v;
}


//ソートアルゴリズム
//バブルソート
void	bubble(std::pair<int, double> **list,int s, int l){
	int	i, j;
	std::pair<int, double> v;
	for(i = s; i < l-1; i++){
		if((*list)[i].second <= (*list)[i+1].second) continue;
		v = (*list)[i]; (*list)[i] = (*list)[i+1];(*list)[i+1] =v;
		for(j = i-2; j >= 0; j--){
			if((*list)[j].second <= (*list)[j+1].second) break;
			v = (*list)[j]; (*list)[j] = (*list)[j+1];(*list)[j+1] =v;
		}
	}
}
//クイックソート
void	quick(std::pair<int, double> **list,int s, int l){
	int	i, spt, ept;
	if(l-s < 1) return;
	double	a, b, c, p;
	std::pair<int, double> v;

	a	= (*list)[s].second;
	b	= (*list)[(l+s)/2].second;
	c	= (*list)[l-1].second;

	if( (a-b)*(b-c)*(c-a) != 0){
		p = std::max(std::min(a, b), std::min(std::max(a, b), c));
	}else{

		bool	flg	= true;
		for(i = s; i < l-1; i++){
			if((*list)[i].second != (*list)[i+1].second){
				p	= std::max((*list)[i].second, (*list)[i+1].second);
				flg	= false;
				break;
			}
		}
		if(flg) return;
	}
	spt	= s;
	ept	= l - 1;
	while(1){
		while((*list)[spt].second < p)spt++;
		while((*list)[ept].second >= p)ept--;
		if(spt>=ept) break;
		v	= (*list)[spt]; (*list)[spt] = (*list)[ept]; (*list)[ept] = v;
	}
	quick(list, s, spt);
	quick(list,spt, l);
}

//ログ出力
void	simulator::savelog(char	*fn, user_info* &uf){
	FILE	*fp;
	int	i;
	fp	= fopen(fn, "w");
	for(i = 0; i < N; i++){
		if(uf[i].edge==NULL) continue;
		fprintf(fp, "%d\t%d\t%d\t%lf\t%lf\t%lf\n", i+1, uf[i].from+1, uf[i].to+1, (*uf[i].edge).dist- uf[i].quata, uf[i].delta, uf[i].dense);
	}
	fclose(fp);
}

//エッジログ（各道路・進行方向毎にエージェントが何人いるかを記載）出力
void	simulator::save_edgelog(char *fn, graph_tensor graph){
	FILE	*fp;
	int	i, j;
	fp	= fopen(fn, "w");
	for(i = 0; i < M; i++){
		for(j = 0; j < graph.vec[i]; j++){
			if((int) graph.edge[i][j].user.size() <= 0)continue;
			fprintf(fp, "%d %d %d\n" , i+1, graph.edge[i][j].dest+1, (int) graph.edge[i][j].user.size());
		}
	}
	fclose(fp);
}

//駅ログ(駅に何人滞留していて、何人が電車にのって帰ったか）出力
void	simulator::save_stationlog(char *fn, std::vector<station> st){
	FILE	*fp;
	int		l, s;
	fp	= fopen(fn, "w");
	std::pair<int, int> p;
	std::map<std::pair<int, int>, int>				counter;
	std::map<std::pair<int, int>, int>::iterator	itrF;

	for(unsigned int i = 0; i < sset.zombie.size(); i++){
		p.first		= uf[sset.zombie[i]].goal;
		p.second	= uf[sset.zombie[i]].tgoal;
		if(counter.count(p)==0){counter[p]=0;}
		counter[p]++;
	}

	for(unsigned int i = 0; i < st.size(); i++){
		s = i;
		fprintf(fp, "%d\t%d\t%d\t%d\t", i+1, st[i].num, st[i].cumilative, st[i].cumilative-st[i].num);

		for(l = 0,itrF = counter.begin(); itrF != counter.end(); itrF++){
			if(itrF->first.first==s)l++;
		}
		fprintf(fp, "%d", l);
		for(itrF = counter.begin(); itrF != counter.end(); itrF++){
			if(itrF->first.first==s){
				fprintf(fp, " %d:%d", itrF->first.second+1, itrF->second);
			}
		}
		fprintf(fp, " \n");
	}
	fclose(fp);
}


//ユーザログ（エージェントの行動記録）出力
void	simulator::save_userlog(char *fn, user_info* &uf){
	FILE	*fp;
	int	i, j, flg, dest;
	trajectory	trj;
	fp	= fopen(fn, "w");
	for(i = 0; i < N; i++){
		flg	= 0;
		if(uf[i].dead < 0){
			flg	= 1;
		}
		dest	= uf[i].to;
		if(flg	== 0){
			dest	= -1;
		}
		fprintf(fp, "%d\t%d %d\t%d %d\t%d", i+1,flg, dest+1, uf[i].tid+1, uf[i].ttime+1, (int) uf[i].logs.size());

		for(j = 0; j <(int) uf[i].logs.size()-1; j++){
			fprintf(fp, " %d:%d:%.2lf", uf[i].logs[j].start+1, uf[i].logs[j].time+1, uf[i].logs[j].dist);
		}
		if(uf[i].logs.size()>0){
			trj = uf[i].logs.back();
			if(uf[i].dead<0){
				trj.dist	*=-1;
				trj.dist	-= uf[i].quata;
			}
			fprintf(fp, " %d:%d:%.2lf", trj.start+1, trj.time+1, trj.dist);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void	simulator::save_od_hist(char *fn){
	FILE	*fp;
	fp	= fopen(fn, "w");
	fprintf(fp,"#time\tuid\tfrom\tto\tdest\tgrp\tkeep_via\n");
	for(unsigned int i = 0; i < od_logs.size(); i++){
		fprintf(fp, "%d\t%d\t%d\t%d\t%d\t%d\t%d\n", od_logs[i].time+1, od_logs[i].uid+1, od_logs[i].node+1, od_logs[i].node2+1, od_logs[i].dest+1, od_logs[i].grp+1, od_logs[i].via);
	}
	fclose(fp);
}

//誘導ログの出力
void	simulator::save_navi(char *fn, std::vector<int> index, std::vector<navigation> log){
	FILE	*fp;
	int	size;
	int	i;
	size	= index.size();
	fp	= fopen(fn, "w");
	fprintf(fp, "%d", size);
	for(i = 0; i < size; i++){
		fprintf(fp, " %d:%d", i+1, index[i]);
	}
	fprintf(fp, "\n");
	size	= log.size();
	fprintf(fp, "%d\n", size);
	for(i = 0; i < size; i++){
		log[i].write2ascii(fp);
/*
		l = 0;
		std::map<navigation::condition, std::vector<navigation::values> >::iterator itrF;
		for(itrF = log[i].begin(); itrF != log[i].end(); itrF++){
			navigation::condition c = itrF->first;
			for(unsigned int i = 0; i < log[i][c].size(); i++){
				l++;
			}
		}
		fprintf(fp,"%d", l);
		for(itrF = log[i].begin(); itrF != log[i].end(); itrF++){
			navigation::condition c = itrF->first;
			for(unsigned int i = 0; i < log[i][c].size(); i++){
				navigation::values	v		= log[i][c][i];
				fprintf(fp, " %d:%d:%d:%lf", c.node+1, c.grp.+1,v.dest+1, v.prob);

			}
		}
		fprintf(fp, "\n");
*/
	}
	fclose(fp);
}

void	simulator::save_trainlog(char *fn){
	FILE	*fp;
	fp	= fopen(fn, "w");
	for(unsigned int i = 0; i < dia.diagrams.size(); i++){
		fprintf(fp, "%d\t%d\t%d", dia.diagrams[i].tid+1, dia.diagrams[i].capa, dia.diagrams[i].ride);
		int l_sche	= dia.diagrams[i].schedules.size();
#if 1	//過去ログのみバージョン
		for(unsigned int j = 0; j < dia.diagrams[i].schedules.size() ;j++){
			if(dia.diagrams[i].schedules[j].time > sset.nTime){

				l_sche = j;
				break;
			}
		}
#endif

		fprintf(fp, "\t%d", l_sche);
		for(int j = 0; j < l_sche;j++){
			fprintf(fp, " %d", dia.diagrams[i].schedules[j].time+1);
			fprintf(fp, ":%d", dia.diagrams[i].schedules[j].station+1);
			fprintf(fp, ":%d", dia.diagrams[i].schedules[j].ride);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

}

//Dijkstraマップの出力（目的地毎に交差点からの距離が記載）
void	writeDijkstra_map(char	*fn, std::vector<station> st){
	FILE	*fp;
	int	j, M, size;
	size	= st.size();
	M		= st[0].len_dijkstra;
	fp		= fopen(fn, "w");
	fprintf(fp, "%d %d\n", size, M);
	for(int i = 0; i < size; i++){
		fprintf(fp, "%lf", st[i].dijkstra[0]);
		for(j = 1; j < M; j++){
			fprintf(fp, " %lf", st[i].dijkstra[j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void	simulator::writeTrafficLog(char *fn){
	FILE	*fp;
	fp	= fopen(fn, "w");
	for(int i = 0; i < (int) traffic_logs.size(); i++){
		fprintf(fp, "%d\t%d\t%d\t%d\t%lf\n", traffic_logs[i].time+1, traffic_logs[i].from+1, traffic_logs[i].to+1, traffic_logs[i].type,traffic_logs[i].value);
	}
	fclose(fp);
}

//シミュレータ本体
//iteration
int	simulator::iterate(){
	int	i, j, k, u;
	int	m, n;
	char	filename[STR_LENGTH];
	double	v, w, spd;
	edge_info	*e;
	clock_t	start, end;
	double	t1, t2, t3, t4;
	trajectory	trj;

	t1 = t2 = t3 = t4 = 0;
	if(sset.nTime >= sset.eTime)return 0;
		srand(sset.nTime);
		bombExplode();
		navi_index.push_back(sset.navi_flg);
		//userの追加
		for(;bpt < N; bpt++){
			if(uf[born[bpt].first].born	> sset.nTime) break;
			u	= born[bpt].first;
/**/
			//int org = uf[u].from;
			uf[u].to = uf[u].from;
			user_info org	= uf[u].clone();
			chkSignage(uf[u]);
			while(1){
				//経由地の判定
				if(uf[u].via.size()>0){
#ifdef TIGHT_PATH
					uf[u].edge	= graph.relayDirect(uf[u].from, sp[uf[u].via[0].first]);
					if(uf[u].edge != NULL) break;
#endif
					uf[u].edge	= graph.relay(uf[u].from, sp[uf[u].via[0].first], uf[u].group, navi);
				}else{
#ifdef TIGHT_PATH
					uf[u].edge	= graph.relayDirect(uf[u].from, st[uf[u].goal]);
					if(uf[u].edge != NULL) break;
#endif
					uf[u].edge	= graph.relay(uf[u].from, st[uf[u].goal], uf[u].group, navi);
				}
				if(uf[u].edge != NULL) break;
#if BORN_TROUBLE == 1
				while(1){
					uf[u]	= org.clone();	//reset
					uf[u].from = random() % graph.M;

					uf[u].to = uf[u].from; chkSignage(uf[u]);
					if(uf[u].via.size()>0 && uf[u].via[0].first == uf[u].from) continue;
					else if(st[uf[u].goal].dijkstra[uf[u].from]== 0.0)continue;
					break;
				}
#elif BORN_TROUBLE == 2
	fprintf(stderr,"user %d can't born\n", u+1);
	exit(1);
#elif BORN_TROUBLE == 3
				break;
		
#endif
			}
#if BORN_TROUBLE == 3
			if(uf[u].edge==NULL){
				uf[u].born	+= BORN_DELAY;
				born[bpt].second+=BORN_DELAY;
				bubble(&born, bpt, N);
				bpt--;
				fprintf(stderr, "Agent %d chage start time from %d to %d\n",u,  uf[u].born -BORN_DELAY+1, uf[u].born+1);
				continue;
			}
#endif
			if(org.from != uf[u].from)printf("Agent %d change origin node from %d to %d\n", u+1, org.from+1, uf[u].from+1);
			uf[u].to	= (*uf[u].edge).dest;
			uf[u].quata	= (*uf[u].edge).dist;
			(*uf[u].edge).user.push_back(u);
			trj.start	= uf[u].from;
			trj.time	= sset.nTime;
			trj.dist	= -(*uf[u].edge).dist;
			uf[u].logs.push_back(trj);

/**/
			sset.alive.push_back(u);
		}
#ifdef	DEBUG
		printf("%d\t%d\t%d\n",sset.nTime+1, bpt,(int) sset.alive.size());
#endif

		start	= clock();
		updateStation();
		//駅滞留している人数を減少
		for( unsigned int i = 0; i < st.size(); i++){
			//st[i].update(sset.nTime);
			//出入口のカウンタリセット
			for(unsigned int j = 0; j <st[i].entrances.size();j++){
				st[i].entrances[j].pass	= 0;
			}
		}
		end	= clock();
		t1	+= (double)(end-start) /CLOCKS_PER_SEC;


		//並び替え
		start	= clock();
		for(i = 0; i < M; i++){
			for(j = 0; j < graph.vec[i]; j++){
				graph.edge[i][j].scnt	= 0;
				int ucnt	= graph.edge[i][j].user.size();
				if(ucnt	== 0)continue;
				//残り距離順に並び替え
				for(k = 0; k < ucnt; k++){
					quata[k].first	= graph.edge[i][j].user[k];
					quata[k].second	= uf[graph.edge[i][j].user[k]].quata;
				}
				//quick(&quata, 0, k);
				bubble(&quata, 0, k);
				for(k = 0; k < ucnt; k++){
					graph.edge[i][j].user[k]	= quata[k].first;
				}
#if 0
#elif 1
				ucnt	= graph.edge[i][j].user.size();
				for(n = 0, m = 0; m < ucnt;m++ ){
					u		= quata[m].first;
					for(;n < m;n++){
						if(quata[m].second-quata[n].second < VIEW_RANGE)break;
					}
					//視界
					w		= std::min(VIEW_RANGE, quata[m].second);
					w		= std::max(w, MINIMUM_VIEW);
					if(graph.edge[i][j].flg.first==2){
						w		*= graph.edge[i][j].flg.second;
					}else{
						w		*= graph.edge[i][j].width;
					}
					//視界の密度
					uf[u].dense	= (m - n) / (w+DBL_MIN);
					int from	= graph.edge[i][j].src;
					int to		= graph.edge[i][j].dest;
					int src		= -1;
					if(uf[u].logs.size() > 1){
						src = uf[u].logs[uf[u].logs.size()-2].start;
					}
					if(uf[u].quata == graph.edge[i][j].dist && sig.checkSignal(sset.nTime, from, to, src)){
						uf[u].delta =0.0;
						graph.edge[i][j].scnt++;
						continue;
					}
					spd		= graph.edge[i][j].alpha / (uf[u].dense+DBL_MIN)	- graph.edge[i][j].beta;
					spd		= std::max(std::min(spd, uf[u].speed),0.0);
					spd	 *= graph.edge[i][j].gamma;
					uf[u].delta	= spd;		//移動予定距離
					///if(v < 0.0 || v>2.5){printf("err %d %lf\n",u+1, v);exit(1);}
				}
#endif
			}
		}

		end	= clock();
		t2	+= (double)(end-start) /CLOCKS_PER_SEC;
		start	= clock();
		end	= clock();
		t3	+= (double)(end-start) /CLOCKS_PER_SEC;


		//移動処理
		//[移動]と[道変更or残留]を同時にやるべく分けるか思案
		//分けると一気に分岐先に流入する可能性がある
		start	= clock();
		for( i = 0; i <(int) sset.alive.size(); i++){
			u		= sset.alive[i];
			if(uf[u].wait>0) uf[u].wait--;

			if(uf[u].wait > 0 )continue;
			if(uf[u].gate){
				(*uf[u].edge).g_cnt--;
				uf[u].gate	= false;
				trj.start	= uf[u].from;
				trj.time	= sset.nTime;
				trj.dist	= -(*uf[u].edge).dist;
				uf[u].logs[uf[u].logs.size()-1].dist = -0.0;
				uf[u].logs.push_back(trj);
			}

			bool mv_chk =true;
			// if(uf[u].quata>0.0) mv_chk = true;
			uf[u].quata	-= uf[u].delta;
			if(uf[u].quata <= 0.0){

				int to	= uf[u].to;
				int chk	= -1;

				if(mv_chk){
					chkSignage(uf[u]);
				}
				//経由地到着判定
				for(int s = 0; s < (int)uf[u].via.size(); s++){
					if(uf[u].via[s].second>=0) continue;

					//printf("u = %d to= %d s= %d next%d\n",u, to, s, uf[u].via[s].first );
					if(uf[u].via[s].first	== to){
						uf[u].via[s].second= sset.nTime;
						continue;
					}
					//printf("after u = %d to= %d s= %d next%d\n",u, to, s, uf[u].via[s].first );
					chk = uf[u].via[s].first;
					break;
				}
				//経由地がある場合
				if(chk != -1){
					//printf("Inchk %d %d\n", chk, uf[u].to);
#ifdef TIGHT_PATH
					e = graph.relayDirect(uf[u].to, sp[chk]);
					if(e==NULL)
						e = graph.relay(uf[u].to, sp[chk], uf[u].group, navi);
#else
					e = graph.relay(uf[u].to, sp[chk], uf[u].group, navi);
#endif
					//printf("after Inchk %d %d %e\n", chk, uf[u].via.size(), (*e).dist);
					if(e==NULL){
						uf[u].quata	= 0.0;
						continue;
					}
					v	= (*e).dist	- VIEW_RANGE;
					int ucnt	= (*e).user.size();
					for(j = ucnt-1, k= 0; j >=0; j--){
						m = (*e).user[j];
						if(uf[m].quata >= v)k++;
						else break;
					}


					if((*e).flg.first==2){
						w	= k/((*e).flg.second* std::min((*e).dist, VIEW_RANGE));
					}else{
						w	= k/((*e).width* std::min((*e).dist, VIEW_RANGE));
					}
					if(w > TRANSITION || ((*e).g_capa >=0 && (*e).g_capa <= (*e).g_cnt) ){
						uf[u].quata	= 0.0;
						continue;
					}
					ucnt	= (*uf[u].edge).user.size();
					for(j = 0; j < ucnt; j++){
						if((*uf[u].edge).user[j] == u){
							(*uf[u].edge).user.erase((*uf[u].edge).user.begin()+j);
							break;
						}
					}

					uf[u].from	= uf[u].to;
/**/
					uf[u].edge	= e;
					uf[u].to	= (*uf[u].edge).dest;
					uf[u].quata	= (*uf[u].edge).dist;
					(*uf[u].edge).user.push_back(u);


					uf[u].logs.back().dist	*= -1;
					trj.start	= uf[u].from;
					trj.time	= sset.nTime;
					trj.dist	= -(*uf[u].edge).dist;
					uf[u].logs.push_back(trj);

					if((*e).g_capa>=0){
						(*e).g_cnt++;
						uf[u].wait	= (*e).g_wait;
						uf[u].gate	= true;
					}
/**/
					continue;
				}

				//if(uf[u].to != uf[u].goal){
				if(st[uf[u].goal].dijkstra[uf[u].to] > 0.0){

#ifdef TIGHT_PATH
					e = graph.relayDirect(uf[u].to, st[uf[u].goal]);
					if(e==NULL)
						e = graph.relay(uf[u].to, st[uf[u].goal], uf[u].group, navi);
#else
					e = graph.relay(uf[u].to, st[uf[u].goal], uf[u].group, navi);
#endif
					if(e==NULL){
						uf[u].quata	= 0.0;
						continue;
					}
					v	= (*e).dist	- VIEW_RANGE;
					int ucnt = (*e).user.size();
					for(j = ucnt - 1, k= 0; j >=0; j--){
						m = (*e).user[j];
						if(uf[m].quata >= v)k++;
						else break;
					}


					if((*e).flg.first==2){
						w	= k/((*e).flg.second* std::min((*e).dist, VIEW_RANGE));
					}else{
						w	= k/((*e).width* std::min((*e).dist, VIEW_RANGE));
					}
					//if(w > TRANSITION ){
					if(w > TRANSITION || ((*e).g_capa >=0 && (*e).g_capa <= (*e).g_cnt) ){
						uf[u].quata	= 0.0;
						continue;
					}

					ucnt	= (*uf[u].edge).user.size();
					for(j = 0; j < ucnt; j++){
						if((*uf[u].edge).user[j] == u){
							(*uf[u].edge).user.erase((*uf[u].edge).user.begin()+j);
							break;
						}
					}

					uf[u].from	= uf[u].to;
/**/
					uf[u].edge	= e;
					uf[u].to	= (*uf[u].edge).dest;
					uf[u].quata	= (*uf[u].edge).dist;
					(*uf[u].edge).user.push_back(u);


					uf[u].logs.back().dist	*= -1;
					trj.start	= uf[u].from;
					trj.time	= sset.nTime;
					trj.dist	= -(*uf[u].edge).dist;
					uf[u].logs.push_back(trj);
/**/
					if((*e).g_capa>=0){
						(*e).g_cnt++;
						uf[u].wait	= (*e).g_wait;
						uf[u].gate	= true;
					}
					continue;

				}else{
					if( st[uf[u].goal].check(uf[u].to) == false){
						uf[u].quata=0.0;
						continue;
					}

					uf[u].dead	= sset.nTime;
					int ucnt	= (*uf[u].edge).user.size();
					for(j = 0; j < ucnt; j++){
						if((*uf[u].edge).user[j] == u){
							(*uf[u].edge).user.erase((*uf[u].edge).user.begin()+j);
							break;
						}
					}
					sset.zombie.push_back(sset.alive[i]);
					sset.alive[i] = sset.alive.back();
					sset.alive.pop_back();
					uf[u].edge	= NULL;

					uf[u].logs.back().dist	*= -1;
					trj.start	= uf[u].to;
					trj.time	= sset.nTime;
					trj.dist	= -9999.0;
					uf[u].logs.push_back(trj);
					i--;
					continue;
				}

				uf[u].from	= uf[u].to;
				int ucnt	= (*uf[u].edge).user.size();
				for(j = 0; j < ucnt; j++){
					if((*uf[u].edge).user[j] == u){
						(*uf[u].edge).user.erase((*uf[u].edge).user.begin()+j);
						break;
					}
				}
			}

		}
		end	= clock();
		t4	+= (double)(end-start) /CLOCKS_PER_SEC;

		if((sset.nTime+1) % sset.log_interval == 0){
			sprintf(filename, "%s/log%06d.txt", output, sset.nTime+1);
			savelog(filename, uf);
			if(sset.edge_log){
				sprintf(filename, "%s/log%06d_edge.txt", output, sset.nTime+1);
				save_edgelog(filename, graph);
			}
			if(sset.station_log){
				sprintf(filename, "%s/log%06d_station.txt", output, sset.nTime+1);
				save_stationlog(filename, st);
			}
		}

	sset.nTime++;
	return 1;
}


//resume用バイナリデータ出力
void	simulator::save_binary(char *fn){
	FILE	*fp;
	int	i, size;
	fp	= fopen(fn, "wb");
	fwrite(&N, sizeof(int), 1, fp);
	fwrite(&M, sizeof(int), 1, fp);
	fwrite(&G, sizeof(int), 1, fp);

	for(i = 0; i < N; i++){
		uf[i].write2bin(fp);
	}
	graph.write2bin(fp);
	sset.write2bin(fp);
	sig.write2bin(fp);
	kanban.write2bin(fp);
	tb.write2bin(fp);

	size	= st.size();
	fwrite(&size, sizeof(int), 1, fp);
	for(unsigned int i = 0; i < st.size(); i++){
		st[i].write2bin(fp);
	}

	size = navi_index.size();
	fwrite(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		fwrite(&navi_index[i], sizeof(int), 1, fp);
	}

	size = navi_log.size();
	fwrite(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		navi_log[i].write2bin(fp);
	}

	size = traffic_logs.size();
	fwrite(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		fwrite(&traffic_logs[i], sizeof(traffic_log), 1, fp);
	}
	size	= od_logs.size();
	fwrite(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		fwrite(&od_logs[i], sizeof(od_change_log), 1, fp);
	}

	dia.write2bin(fp);
	fclose(fp);

}

//resume用バイナリデータ読み込み
void	simulator::load_binary(char *fn){
	FILE	*fp;
	int	i, j, size;
	if((fp = fopen(fn, "rb")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	fread(&N, sizeof(int), 1, fp);
	fread(&M, sizeof(int), 1, fp);
	fread(&G, sizeof(int), 1, fp);
	uf	= new user_info [N];
	for(i = 0; i < N; i++){
		uf[i].read2bin(fp);
	}
	graph.read2bin(fp);
	sset.read2bin(fp);
	sig.read2bin(fp);

	kanban.read2bin(fp);
	tb.read2bin(fp);
#ifdef	DEBUG
	printf("times %d %d\n", sset.sTime, sset.eTime);
	printf("%d %d\n", N, M);
#endif
	fread(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		station s;
		s.read2bin(fp);
		st.push_back(s);
#ifdef	DEBUG
		printf("%d len entrance%d\n", i, (int)st[i].entrances.size());
#endif
	}

	navi_index.clear();
	std::vector<int>().swap(navi_index);
	fread(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		fread(&j, sizeof(int), 1, fp);
		navi_index.push_back(j);
	}

	navi_log.clear();
	std::vector<navigation>().swap(navi_log);
	fread(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		navigation	navi;
		navi.read2bin(fp);
		navi_log.push_back(navi);
	}

	traffic_logs.clear();
	std::vector<traffic_log>().swap(traffic_logs);
	fread(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		traffic_log log;
		fread(&log, sizeof(traffic_log), 1, fp);
		traffic_logs.push_back(log);
	}

	od_logs.clear();
	fread(&size, sizeof(int), 1, fp);
	for(i = 0; i < size; i++){
		od_change_log log;
		fread(&log, sizeof(od_change_log), 1, fp);
		od_logs.push_back(log);
	}

	dia.clear();
	dia.read2bin(fp);
	fclose(fp);

	sset.sTime	= sset.nTime;

	graph.rev_check();
	for(int i = 0; i< N;i++){
		uf[i].edge	= NULL;
		if(uf[i].dead >= 0)continue;
		if(uf[i].born >= sset.sTime)continue;
		int f	= uf[i].from;
		int t	= uf[i].to;
		int c	= 0;
		for(int j = 0; j < graph.vec[f];j++){
			int d 	= graph.edge[f][j].dest;
			if(d== t){
				uf[i].edge	= &graph.edge[f][j];
				c=1;
				break;
			}
		}
		if(c==0){printf("err\n");exit(1);}
	}
}

void simulator::reset_condition(){
	sset.navi_flg	= -1;
	dia.resetSchedule(sset.nTime);
	resetTrafficRegulation();
	graph.resetGate();
	graph.resetEdgeCost();
	kanban.clear();
	sig.clear();
	navi.clear();
	tb.removeUnexploded();
#ifdef	DEBUG
	printf("remove\n");
#endif
}

void simulator::initialize(int argc,char **argv ){

	int		opt;


	std::map<std::string, std::vector<char *> > input;

	sset.navi_flg	= -1;
	sset.sTime	= 0;
	sset.eTime	= 3600;
	sprintf(output,".");

	strcpy(output_resume, "");
	strcpy(output_dijkstra, "");
	strcpy(output_traffic, "");

	kanban.global_t = &sset.nTime;
/*
	struct option longopts[] = {
		{"costOfEdge",		required_argument,	NULL, 2},
		{"dijkstra",		required_argument,	NULL, 2},
		{"Edgelog",			no_argument,		NULL, 2},
		{"end",				required_argument,	NULL, 2},
		{"help",			no_argument,		NULL, 2},
		{"outputdir",		required_argument,	NULL, 2},
		{"resume",			required_argument,	NULL, 2},
		{"BombEvent",		required_argument,	NULL, 2},
		{"Diagrams",		required_argument,	NULL, 2},
		{"Stationlog",		no_argument,	NULL, 2},
		{0, 0, 0, 0},
	};
	int longindex;
	while ((opt = getopt_long_only(argc, argv, "", longopts, &longindex)) != -1) {
		printf("%d %d %s %s\n",opt, longindex, longopts[longindex].name, optarg);
	}
	for(int i = 0; i < argc; i++){
	}
//	exit(1);
*/

	optind	=1;
	while((opt = getopt(argc,argv,"ho:n:e:l:Er:R:b:B:d:D:St:T:p:C:s:a:g:G:H:k:c:"))!=-1){
		switch(opt){
			case 'o':
				strcpy(output, optarg);
				break;
			//resume関連
			case 'r':
				input["resume"].push_back(optarg);
				break;
			case 'R':
				input["resumeC"].push_back(optarg);//resume all statuses are carried over
				break;
			case 'b':
				strcpy(output_resume, optarg);
				break;

			case 'h':
				printf("-h	:ヘルプの表示\n");
				printf("-o	:ログ出力ディレクトリ指定(ディレクトリ作成は行いません)\n");
				printf("-b	:終了状態のbinary出力指定\n");
				printf("-d	:終了状態のdijkstra出力指定\n");
				printf("-r	:指定binaryからのresume	※再開\n");
				printf("-r	:指定binaryからのresume	状況引継※再開\n");
				printf("-l	:ログ出力間隔\n");
				printf("-e	:終了時刻指定\n");
				printf("-E	:Edge logの出力\n");
				printf("-T	:traffic logの出力\n");
				printf("-s	:signalのファイル指定\n");
				printf("-S	:Station logの出力\n");
				printf("-n	:navigation(誘導)ファイル指定\n");
				printf("-t	:traffic regulation(交通規制)\n");
				printf("	:1		通行止め\n");
				printf("	:2 (value)	道幅制限\n");
				printf("-a	:alpha値設定ファイル指定\n");
				printf("-g	:gamma値設定ファイル指定\n");
				printf("-c	:edge cost値設定ファイル指定\n");
				printf("-G	:Gate設定ファイル指定\n");
				printf("-H	:ノード高さ設定ファイル指定\n");
				printf("-C	:カーブ設定ファイル指定\n");
				printf("-p	:ノード位置設定ファイル指定\n");
				printf("-k	:目的地変更看板ファイル指定\n");
				printf("-D	:Diagramファイル指定\n");
				printf("-B	:時限式イベントファイル指定\n");
				exit(1);

				break;
			case 'c':
				input["edgeCost"].push_back(optarg);
				break;
			case 'd':
				strcpy(output_dijkstra, optarg);
				break;
			case 'B':
				input["bombs"].push_back(optarg);
				break;
			case 'D':
				input["diagram"].push_back(optarg);
				break;
			case 'G':
				input["gate"].push_back(optarg);
				break;
			case 'k':
				input["signage"].push_back(optarg);
				break;
			case 's':
				input["signal"].push_back(optarg);
				break;
			case 't':
				input["traffic"].push_back(optarg);
				break;

			case 'a':
				input["alpha"].push_back(optarg);
				break;
			case 'g':
				input["gamma"].push_back(optarg);
				break;
			//可視化用オプション
			case 'p':
				input["points"].push_back(optarg);
				break;
			case 'H':
				input["height"].push_back(optarg);
				break;
			case 'C':
				input["curve"].push_back(optarg);
				break;
//
			case 'n':
				input["navi"].push_back(optarg);
				break;
			case 'e':
				input["endTime"].push_back(optarg);
				break;
			case 'E':
				input["edgeLog"].push_back(NULL);
				break;
			case 'l':
				input["logInterval"].push_back(optarg);
				break;
			case 'S':
				input["stationLog"].push_back(NULL);
				break;
	
			case 'T':
				strcpy(output_traffic, optarg);
				break;
			default:
				break;
		}
	}
	if(input.count("resume") ){
		load_binary(input["resume"].back());
		reset_condition();

		//
		for(unsigned int i = 0; i < st.size(); i++)
			delete [] st[i].dijkstra;
	}else if(input.count("resumeC") ){

		load_binary(input["resumeC"].back());

		for(unsigned int i = 0; i < st.size(); i++)
			delete [] st[i].dijkstra;
	}else{
		readData(argv[optind]);
		graph.readGraph(argv[optind+1]);
		readStation(argv[optind+2]);
		M	= graph.M;
	}

	if(input.count("gate")){
		for(unsigned int i = 0; i < input["gate"].size(); i++){
			graph.readGate(input["gate"][i]);
		}
	}
	chkGate();

	grp.clear();
	grp.build(N, uf);
#ifdef DEBUG
	printf("start time %d\tend time %d\n",sset.sTime, sset.eTime);
#endif


	born	= new std::pair<int, double> [N];
	quata	= new std::pair<int, double> [N];
	for(int i = 0; i < N; i++){
		born[i].first	= i;
		born[i].second	= (double)uf[i].born;
	}
	quick(&born, 0, N);
	for(bpt = 0;bpt < N; bpt++)
		if(uf[born[bpt].first].born	> sset.sTime - 1) break;

#ifdef DEBUG
	printf("diameter	= %d\n", graph.diameter-1);
#endif
	sset.nTime	= sset.sTime;

				
	if(input.count("endTime")){
		sset.eTime		= atoi(input["endTime"].back());
	}
	if(input.count("logInterval")){
		sset.log_interval	= atoi(input["logInterval"].back());
	}
	if(input.count("edgeLog")){
		sset.edge_log		= true;
	}
	if(input.count("stationLog")){
		sset.station_log	= true;
	}
	if(input.count("navi")){
/**/
		navi.clear();
		readNavigation(input["navi"].back());
		navi_log.push_back(navi);
		sset.navi_flg	= (int) navi_log.size();
/**/
	}

	if(input.count("diagram")){
		readDiagram(input["diagram"].back());
	}
	if(input.count("bombs")){
		for(unsigned int i = 0; i < input["bombs"].size(); i++){
				tb.readBombs(input["bombs"][i]);
		}
	}
	if(input.count("signage")){
		kanban.readSignage(input["signage"].back());
/*
		for(unsigned int i = 0; i < input["signage"].size(); i++){
			kanban.readSignage(input["signage"][i]);
		}
*/
	}
	if(input.count("signal")){
		sig.readSignal(input["signal"].back());
	}

	if(input.count("traffic")){
		for(unsigned int i = 0; i < input["traffic"].size(); i++){
			readTrafficRegulation(input["traffic"][i]);
		}
	}

	if(input.count("alpha")){
		graph.readAlpha(input["alpha"].back());
	}
	if(input.count("gamma")){
		graph.readGamma(input["gamma"].back());
	}

	if(input.count("edgeCost")){
		for(unsigned int i = 0; i < input["edgeCost"].size(); i++){
			graph.readEdgeCost(input["edgeCost"][i]);
		}
	}

	if(input.count("points")){
		graph.readPoints(input["points"].back());
	}
	if(input.count("height")){
		graph.readHeight(input["height"].back());
	}
	if(input.count("curve")){
		graph.readCurve(input["curve"].back());
	}
	//経由地ダイクストラのマップはバイナリに含まないものとする
	for(unsigned int i = 0; i < st.size(); i++){
		graph.dijkstra( st[i]);
	}
	sp	= new stop_point[M];
	for(int i = 0; i < M; i++){
		sp[i].id	= i;
		graph.dijkstra( sp[i]);
	}

	tb.removePastbombs(sset.nTime);
	tb.sort();
/**/
	kanban.shrinkLog();

		
	std::map<std::string, std::vector<char *> >::iterator itr;
	for(itr = input.begin(); itr != input.end(); itr++){
		printf("%s\n", itr->first.c_str());
		itr->second.clear();
		std::vector<char *>().swap(itr->second);
	}
	input.clear();
}
