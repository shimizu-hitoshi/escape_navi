#pragma once

#include <pthread.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <map>
#include "twd.h"
#define MAX_THREADS	35
#define	STR_LENGTH	2048
#define	STR_LENGTH_MIN	64
#define	VIEW_RANGE	6.0
//#define RAIN
#ifdef  RAIN
	#define ALPHA   2.0
	#define BETA	0.5
#else
	#define ALPHA   1.8
	#define BETA	0.3
#endif
#define GAMMA   1.0


class simulator;

class reader{
public:
	FILE	*fp;
	char buff[STR_LENGTH];
	reader(char *fn){
		if((fp = fopen(fn, "r")) == NULL) {
		printf("Unknown File = %s\n", fn); exit(1);}
	}
	~reader(){
		fclose(fp);
	}
	bool readNext(char *res){
		//char	res[STR_LENGTH] = "";	
		int		slen;
		strcpy(buff, "");
		strcpy(res, "");
		while(fgets(buff, STR_LENGTH, fp )!= NULL){
			if(buff[0]=='#')continue;
/**/
//空白の連打を詰める
			slen=strlen(buff);
			for(int p=0, q=0; p <= slen; p++){
				if( p < slen && (buff[p]==' ' && buff[p+1]==' '))continue;
				buff[q++]=buff[p];
			}
//空白の連打を詰める
/**/
			//行末が\バックスラッシュなら継続して読み込み
			if( (slen=strlen(buff))>2 && strcmp("\\\n", &buff[slen-2])==0){
				buff[slen -2 ]= ' ';
				strncat(res, buff, slen-1);
				continue;
			}

			strcat(res, buff);
			return true;
		}
		return false;
	}
};


class	signal{
public:
	class condition{
	public:
		int src;
		int from;
		int to;
		bool operator==(const condition c) const{
			return (src == c.src) && (from == c.from) && (to == c.to);
		}
		bool operator!=(const condition c) const{
			return (src != c.src) || (from != c.from) || (to != c.to);
		}
		bool operator<(const condition c) const{
			if(src < c.src ) return true;
			else if(src > c.src ) return false;
			if(from < c.from ) return true;
			else if(from > c.from ) return false;
			if(to < c.to ) return true;
			else if(to > c.to ) return false;
			return false;	//完全一致 
		}

		bool operator>(const condition c) const{
			if(src > c.src ) return true;
			else if(src < c.src ) return false;
			if(from > c.from ) return true;
			else if(from < c.from ) return false;
			if(to > c.to ) return true;
			else if(to < c.to ) return false;
			return false;	//完全一致 
		}
	};

	class prop{
	public:
		int cycle;
		int blue;
		int gap;
	};
	
	std::map<condition, prop> signals;

	void	readSignal(char *fn);
	bool	checkSignal(int t, int from, int to, int src);

	void	addSignal(condition c, prop p){
		signals[c]	= p;
	}
	void	addSignal(char *buff){
		signal::condition c;
		signal::prop p;

		sscanf(buff, "%d\t%d\t%d\t%d\t%d\t%d", &c.src, &c.from, &c.to, &p.cycle, &p.blue, &p.gap);
		c.from--;
		c.to--;
		c.src--;
		addSignal(c, p);

	}
	void	delSignal(condition c){
		signals.erase(c);
	}
	void	delSignal(char *buff){
		signal::condition c;

		sscanf(buff, "%d\t%d\t%d", &c.src, &c.from, &c.to);
		c.from--;
		c.to--;
		c.src--;
		delSignal(c);

	}
	void	clear(){
		//最初からclearでも構わないがログを排出できるようキー毎に処理
		while(!signals.empty()){
			condition c = signals.begin()->first;
			delSignal(c);
		}
		signals.clear();
	}
	void	write2bin(FILE	*fp){
		int l	= 0;	//signals.size();	//でも可
		std::map<condition, prop>::iterator itrF;
		for(itrF = signals.begin(); itrF != signals.end(); itrF++){
			l++;
		}
		fwrite(&l, sizeof(int), 1, fp);
		for(itrF = signals.begin(); itrF != signals.end(); itrF++){
			condition c = itrF->first;
			fwrite(&c, sizeof(condition), 1, fp);
			fwrite(&signals[c], sizeof(prop), 1, fp);
		}
	}
	void	read2bin(FILE	*fp){
		clear();
		int l;
		fread(&l, sizeof(int), 1, fp);
		for(int i = 0; i < l; i++){
			condition	c;
			prop		p;
			fread(&c, sizeof(condition), 1, fp);
			fread(&p, sizeof(prop), 1, fp);
			signals[c]	= p;
		}
	}
};

class signage{
public:
	class condition{
		public:
		int node, grp;
		bool	operator==(const condition c) const{
			return (node == c.node) && (grp && c.grp);
		};
		bool	operator!=(const condition c) const{
			return (node != c.node) || (grp != c.grp);
		};
		bool	operator>(const condition c) const{
			if(node > c.node) return true;
			else if( node == c.node && grp > c.grp) return true;
			return false;
		};
		bool	operator<(const condition c) const{
			if(node < c.node) return true;
			else if( node == c.node && grp < c.grp) return true;
			return false;
		};
	};
	class values{
		public:
		int		dest, grp;
		int		id;
		bool	keep_via;
		double	prob;
		std::vector<int> via;
		~values(){
			via.clear();
			std::vector<int>().swap(via);
		}
		bool operator==(const values v) const{
			if(dest != v.dest || grp != v.grp || keep_via != v.keep_via || prob != v.prob)return false;
			if(via.size()== v.via.size() && std::equal(via.begin(), via.end(), v.via.begin())) return true;
			return false;
		};
		bool operator!=(const values v) const{
			if(dest != v.dest || grp != v.grp || keep_via != v.keep_via || prob != v.prob)return true;
			if(via.size() == v.via.size() && std::equal(via.begin(), via.end(), v.via.begin())) return false;
			return true;
		};

		bool operator>(const values v) const{
			if(dest > v.dest) return true;
			else if(dest < v.dest) return false;
			//dest == v.dest
			if(grp > v.grp) return true;
			else if(grp < v.grp) return false;
			//grp == v.grp
			return false;
		};
	
		bool operator<(const values v) const{
			if(dest < v.dest) return true;
			else if(dest > v.dest) return false;
			//dest == v.dest
			if(grp < v.grp) return true;
			else if(grp > v.grp) return false;
			//grp == v.grp
			return false;
		};


		void read2bin(FILE *fp){
			int size;
			fread(&id, sizeof(int), 1, fp);
			fread(&dest, sizeof(int), 1, fp);
			fread(&grp, sizeof(int), 1, fp);
			fread(&prob, sizeof(double), 1, fp);
			fread(&keep_via, sizeof(bool), 1, fp);
			fread(&size, sizeof(int), 1, fp);
			for(int i = 0; i < size; i++){
				int h;
				fread(&h, sizeof(int), 1, fp);
				via.push_back(h);
			}
		}
		void write2bin(FILE *fp){
			int size;
			fwrite(&id, sizeof(int), 1, fp);
			fwrite(&dest, sizeof(int), 1, fp);
			fwrite(&grp, sizeof(int), 1, fp);
			fwrite(&prob, sizeof(double), 1, fp);
			fwrite(&keep_via, sizeof(bool), 1, fp);
			size = via.size();
			fwrite(&size, sizeof(int), 1, fp);
			for(int i = 0; i < size; i++)
				fwrite(&via[i], sizeof(int), 1, fp);
		}
	};

	class log{
		public:
		int time;
		int act;
		condition c;
		values v;

		void read2bin(FILE *fp){
			fread(&time, sizeof(int), 1, fp);
			fread(&act, sizeof(int), 1, fp);
			fread(&c, sizeof(condition), 1, fp);
			v.read2bin(fp);
		}
		void write2bin(FILE *fp){
			fwrite(&time, sizeof(int), 1, fp);
			fwrite(&act, sizeof(int), 1, fp);
			fwrite(&c, sizeof(condition), 1, fp);
			v.write2bin(fp);
		}

		static bool tmsort(log a, log b){
			if(a.time < b.time) return true;
			return false;
		}

		static bool cvsort(log a, log b){
			if(a.c < b.c)return true;
			if(a.c == b.c && a.v < b.v)return true;
			return false;
		}

	};
	int *global_t, log_id;
	signage(){
		log_id	= 0;
	}
	~signage(){
	}

	std::map<condition, std::vector<values> > signages;
	std::vector<log> logs;

	int checkID(condition c, values v){
		for(unsigned int i = 0; i < logs.size();i++){
			if(logs[i].c == c && logs[i].v==v)return logs[i].v.id;
		}
		return -1;
	}
	void	shrinkLog(){
		std::stable_sort(logs.begin(), logs.end(), log::tmsort);
		std::stable_sort(logs.begin(), logs.end(), log::cvsort);

		for(int i = logs.size()-2; i >= 0; i--){
			if(logs[i].time != logs[i+1].time) continue;
			if(logs[i].c != logs[i+1].c) continue;
			if(logs[i].v != logs[i+1].v) continue;
			logs.erase(logs.begin()+i);
		}

		for(int i = logs.size()-2; i >= 0; i--){
			if(logs[i].c != logs[i+1].c) continue;
			if(logs[i].v != logs[i+1].v) continue;
			if(logs[i].act != logs[i+1].act) continue;
			logs.erase(logs.begin()+i+1);
		}


		std::stable_sort(logs.begin(), logs.end(), log::tmsort);
	}

	void addSignage(condition c, values v){
		if(c.node<0 || c.grp<-1 ||v.dest<0 || v.grp<-1)return;
		v.id	= checkID(c, v);
		if(v.id==-1) v.id = log_id++;

		if(signages.count(c)==0){
			signages[c].push_back(v);
			log l;
			l.c = c;
			l.v = v;
			l.act	= 1;
			l.time	= *global_t;
			logs.push_back(l);
			return;
		}
		std::vector<values>::iterator itrF;
		for(itrF = signages[c].begin(); itrF != signages[c].end(); itrF++){
			if(itrF->dest==v.dest && itrF->grp == v.grp){
				*itrF	= v;
				log l;
				l.c = c;
				l.v = v;
				l.act   = 2;
				l.time  = *global_t;
				logs.push_back(l);
				return;
			}
		}	

		log l;
		l.c = c;
		l.v = v;
		l.act   = 1;
		l.time  = *global_t;
		logs.push_back(l);

		signages[c].push_back(v);
	}
	void	addSignage(char *buff){
		condition c;
		values  v;
		int l;
		int pt;
		sscanf(buff, "%d %d %d %d %lf %d%n", &c.node, &c.grp, &v.dest, &v.grp, &v.prob, &l, &pt);
		c.node--;
		c.grp--;
		v.dest--;
		v.grp--;

		buff+= pt;
		v.keep_via  = (l < 0);
		for(int i = 0; i < l; i++){
			int j;
			sscanf(buff,"%d%n",&j, &pt );
			j--;
			v.via.push_back(j);
			buff+=pt;
		}
		addSignage(c, v);
	}

	void delSignage(int node, int grp, int dest, int grp2){
		condition c;
		c.node  = node;
		c.grp   = grp; 
		if(signages.count(c)==0) return;
		for(unsigned int i = 0; i < signages[c].size(); i++){
			values  v   = signages[c][i];
			if(v.dest ==dest && v.grp == grp2){
//				v.id	= log_id++;
				log l;
				l.c = c;
				l.v = v;
				l.act   = 99;
				l.time  = *global_t;
				logs.push_back(l);

					signages[c].erase(signages[c].begin()+i);
					if(signages[c].empty()){
						signages.erase(c);
					}
					return;
			}
		}
	}	
	void	delSignage(char *buff){
		int node, grp, dest, grp2;
		sscanf(buff, "%d %d %d %d", &node, &grp, &dest, &grp2);
		node--;
		grp--;
		dest--;
		grp2--;
		delSignage(node, grp, dest, grp2);
		
	}
	void	conditional_clear(condition c){
		values v;
		while(signages.count(c)!= 0  && signages[c].size()>0){
			v = signages[c].front();
			delSignage(c.node, c.grp, v.dest, v.grp);
		}
	}
	void	conditional_clear(char *buff){
		condition c;
		sscanf(buff, "%d %d", &c.node, &c.grp);
		c.node--;
		c.grp--;
		conditional_clear(c); 
	}

	void	clear_signage(int node){
		std::vector<condition> del_list;
		std::map<condition, std::vector<values> >::iterator itrF;
		for(itrF = signages.begin(); itrF != signages.end(); itrF++){
			if(itrF->first.node==node){
				del_list.push_back(itrF->first);
			}
		}
		for(unsigned int i = 0; i < del_list.size(); i++){
			conditional_clear(del_list[i]);
		}

	}
	void	clear_signage(char *buff){
		int	node;
		sscanf(buff, "%d", &node);
		node--;
		clear_signage(node);
	}
	
	void	clear(){
		while(!signages.empty()){
			condition c = signages.begin()->first;
			while(!signages[c].empty()){
				////////////////////
				//ログに吐き出すならここ
				////////////////////
				log l;
				l.c = c;
				l.v = signages[c].front();
//				l.v.id	= log_id++;
				l.act	= 99;
	
				l.time	= *global_t;
				logs.push_back(l);

				signages[c].erase(signages[c].begin());
			}
			std::vector<values>().swap(signages[c]);
			signages.erase(c);
		}
		signages.clear();

	}
	values chkSignage(int node, int grp);
	void write2bin(FILE *fp){
		int l;
		fwrite(&log_id, sizeof(int), 1, fp);
		l	= logs.size();
		fwrite(&l, sizeof(int), 1, fp);
		for(int i = 0; i < l; i++){
			logs[i].write2bin(fp);
		}
		l   = 0;
		std::map<condition, std::vector<values> >::iterator itrF;
		for(itrF = signages.begin(); itrF != signages.end(); itrF++){
			l += itrF->second.size();
		}
		
		fwrite(&l, sizeof(int), 1, fp);
		for(itrF = signages.begin(); itrF != signages.end(); itrF++){
			std::vector<values>::iterator itrV;
			condition c = itrF->first;
			for(itrV = signages[c].begin(); itrV != signages[c].end(); itrV++){
				values v = *itrV;
				fwrite(&c, sizeof(condition), 1, fp);
				v.write2bin(fp);
			}
		}
	}

	void read2bin(FILE *fp){
		int i, l;
		clear();
		fread(&log_id, sizeof(int), 1, fp);

		logs.clear();
		fread(&l, sizeof(int), 1, fp);
		for(int i = 0; i < l; i++){
			log lg;
			lg.read2bin(fp);
			logs.push_back(lg);
		}

		fread(&l, sizeof(int), 1, fp);
		for(i = 0; i < l; i++){
			condition c;
			values v;
			fread(&c, sizeof(condition), 1, fp);
			v.read2bin(fp);
			addSignage(c, v);
		}
	}
	void readSignage(char *fn);
	void save_log(char *fn){
		FILE	*fp;
		fp	= fopen(fn, "w");
		fprintf(fp, "#time\tid\tact\tnode\tgrp\tdest\tnew_grp\tprob\tkeep_via\tvia\n");
		for(unsigned int i = 0; i < logs.size(); i++){
			values  v = logs[i].v;
			fprintf(fp, "%d\t%d\t%d", logs[i].time+1,v.id+1,  logs[i].act);
			fprintf(fp, "\t%d\t%d", logs[i].c.node+1, logs[i].c.grp+1);
			fprintf(fp, "\t%d\t%d\t%f", v.dest+1, v.grp+1, v.prob);
			if(logs[i].v.keep_via){
				fprintf(fp, "\ttrue");
			}else{
				fprintf(fp, "\tfalse\t%d",(int) v.via.size());
				for(unsigned int j = 0; j < v.via.size(); j++){
					fprintf(fp, " %d", v.via[j]+1);
				}
				
			}
			fprintf(fp, "\n");
			
		}
		fclose(fp);
	}
};


class time_bomb{
public:
	class bomb{
		public:
		int time;
		bool flg;
		char params[STR_LENGTH];
		bomb(){
			flg = false;
		}
	};
	std::vector<bomb> bombs;
	char *chkBomb(int t){
		for(unsigned int i = 0; i < bombs.size();i++){
			if(bombs[i].flg) continue;
			if(bombs[i].time != t) continue;
			bombs[i].flg = true;
			return bombs[i].params;
		}
		return NULL;
	}

	static bool tmsort(time_bomb::bomb a, time_bomb::bomb b){
		if(a.time < b.time)	return true;
		return false;
	}
	void	sort(){
		std::stable_sort(bombs.begin(), bombs.end(), tmsort);
	}

	//未発火の未来のイベントを削除
	//時系列に並んでいるはずなので,pop_backを使用
	void	removeUnexploded(){
		while(1){
			if(bombs.size()>0 && bombs.back().flg	== false){
				//printf("removeUnexploded\t%s\n", bombs.back().params);
				bombs.pop_back();
				continue;
			}
			break;
		}
	}
	//未発火の過去のイベントを削除
	//過去に混じっているので、eraseを使用
	void	removePastbombs(int t){	
		while(1){
			bool	flg	= true;
			for(unsigned int i = 0; i < bombs.size();i++){
				if(bombs[i].flg) continue;
				if(bombs[i].time >= t) continue;
				//printf("removePast\t%s\n", bombs[i].params);
				bombs.erase(bombs.begin()+i);
				flg=false;
				break; 
			}
			if(flg) break;
		}
	}
	void	write2bin(FILE	*fp){
		int		size;
		size	= bombs.size();
		fwrite(&size, sizeof(int), 1, fp);
		for(int i = 0; i < size; i++){
			fwrite(&bombs[i].time, sizeof(int), 1, fp);
			fwrite(&bombs[i].flg, sizeof(bool), 1, fp);
			fwrite(&bombs[i].params, STR_LENGTH, 1, fp);
		}
		
	}
	void	read2bin(FILE	*fp){
		int		size;
		bomb	b;
		bombs.clear();
		std::vector<bomb>().swap(bombs);
		fread(&size, sizeof(int), 1, fp);
		for(int i = 0; i < size; i++){
			fread(&b.time, sizeof(int), 1, fp);
			fread(&b.flg, sizeof(bool), 1, fp);
			fread(&b.params, STR_LENGTH, 1, fp);
			bombs.push_back(b);
		}
	}
	
	void	readBombs(char *fn);
};

class	traffic_log{
public:
	int		time, from, to, type;
	double	value;
	bool tm(traffic_log left, traffic_log right){
		return (left.time >right.time);
	}
};

class	od_change_log{
public:
	int uid, node, node2, time, dest, grp, via;
};

class navigation{
	public:
	class	condition{
		public:
		int	node;
		int	grp;

		bool operator==(const condition c) const{
			return (node == c.node) && (grp == c.grp);
		}
		bool operator!=(const condition c) const{
			return (node != c.node) || (grp != c.grp);
		}
		bool operator<(const condition c) const{
			if(node < c.node ) return true;
			else if(node > c.node ) return false;
			if(grp < c.grp ) return true;
			else if(grp > c.grp ) return false;
			return false;	//完全一致 
		}
		bool operator>(const condition c) const{
			if(node > c.node ) return true;
			else if(node < c.node ) return false;
			if(grp > c.grp ) return true;
			else if(grp < c.grp ) return false;
			return false;	//完全一致 
		}

	};
	class	values{
		public:
		int		dest;
		double	prob;
	};
	std::map<navigation::condition, std::vector<values> > navi;
	bool	check_navi(int p, int grp){
		condition c;
		c.node	= p;
		c.grp	= grp;
		if(navi.count(c)>0) return true;
		c.grp	= -1;
		if(navi.count(c)>0) return true;
		return false;
	}
	bool	get_val(int p , int to, int grp ){
		condition c;
		c.node	= p;
		c.grp	= grp;
		if(navi.count(c) == 0) c.grp	= -1;
		for(unsigned int i = 0; i < navi[c].size(); i++){
			if(navi[c][i].dest==to) return navi[c][i].prob;
		}
		return 0.0;
	}
	void	addNavigation(condition c, values v){
			navi[c].push_back(v);
	}
	void	addNavigation(char *buff){
		int			pt, l;
		condition	c;
		sscanf(buff, "%d %d %d%n", &c.node, &c.grp, &l, &pt);
		c.node--;
		c.grp--;
		buff+= pt;
		if(navi.count(c)>0){navi[c].clear();std::vector<values>().swap(navi[c]);}
		for(int i = 0; i < l; i++){
			values v;
			sscanf(buff, "%d:%lf%n", &v.dest, &v.prob, &pt);
			v.dest--;
			addNavigation(c, v);
			buff+=pt;
		}
	}

	void	clear(){
		std::map<navigation::condition, std::vector<values> >::iterator itrF;
		for(itrF = navi.begin(); itrF != navi.end(); itrF++){
			navigation::condition c	 = itrF->first;
			navi[c].clear();
			std::vector<values>().swap(navi[c]);
		}
		navi.clear();
	}
	void	write2ascii(FILE *fp){
		int l = 0;
		std::map<navigation::condition, std::vector<values> >::iterator itrF;
		for(itrF = navi.begin(); itrF != navi.end(); itrF++){
			navigation::condition c	 = itrF->first;
			for(unsigned int i = 0; i < navi[c].size(); i++){
				l++;
			}
		}
		fprintf(fp, "%d", l);
		for(itrF = navi.begin(); itrF != navi.end(); itrF++){
			navigation::condition c	 = itrF->first;
			for(unsigned int i = 0; i < navi[c].size(); i++){
				values	v		= navi[c][i];
				fprintf(fp, " %d:%d:%d:%lf", c.node+1, c.grp+1, v.dest+1, v.prob);
			}
		}
		fprintf(fp, "\n");
	}

	void	write2bin(FILE	*fp){
		int l = 0;
		std::map<navigation::condition, std::vector<values> >::iterator itrF;
		for(itrF = navi.begin(); itrF != navi.end(); itrF++){
			navigation::condition c	 = itrF->first;
			for(unsigned int i = 0; i < navi[c].size(); i++){
				l++;
			}
		}
		fwrite(&l, sizeof(int), 1, fp);
		for(itrF = navi.begin(); itrF != navi.end(); itrF++){
			navigation::condition c	 = itrF->first;
			for(unsigned int i = 0; i < navi[c].size(); i++){
				values	v		= navi[c][i];
				fwrite(&c, sizeof(condition), 1, fp);
				fwrite(&v, sizeof(values), 1, fp);
			}
		}
	}
	void	read2bin(FILE *fp){
		clear();
		int l;
		fread(&l, sizeof(int), 1, fp);
		for(int i = 0; i < l; i++){
			condition	c;
			values		v;
			fread(&c, sizeof(condition), 1, fp);
			fread(&v, sizeof(values), 1, fp);
			navi[c].push_back(v);
		}
	}
};


class	simulation_setting{
public:
	int			sTime, eTime, nTime;	//sTime:開始時刻, eTime:終了時刻
	int			navi_flg;
	bool		edge_log, station_log;		//ログ出力判定（Trueの場合、それぞれエッジログ、駅ログを出力）
	int			log_interval;				//ログデータ出力間隔
	std::vector<int>	alive;				//生存エージェントリスト
	std::vector<int>	zombie;				//電車待ちエージェント


	simulation_setting(){
		log_interval	=	60;
		edge_log	=	false;
		station_log	=	false;
	}
	~simulation_setting(){
		alive.clear();
		std::vector<int>().swap(alive);
		zombie.clear();
		std::vector<int>().swap(zombie);
	}
	void	write2bin(FILE	*fp){
		int	i, l;
		fwrite(&sTime, sizeof(int), 1, fp);
		fwrite(&nTime, sizeof(int), 1, fp);
		fwrite(&eTime, sizeof(int), 1, fp);
		fwrite(&navi_flg, sizeof(int), 1, fp);
		l	= (int)alive.size();
		fwrite(&l, sizeof(int), 1, fp);
		for(i =0; i < l; i++){
			fwrite(&alive[i], sizeof(int), 1, fp);
		}
		l	= (int)zombie.size();
		fwrite(&l, sizeof(int), 1, fp);
		for(i =0; i < l; i++){
			fwrite(&zombie[i], sizeof(int), 1, fp);
		}
	}
	void	read2bin(FILE	*fp){
		int	i, j, l;
		fread(&sTime, sizeof(int), 1, fp);
		fread(&nTime, sizeof(int), 1, fp);
		fread(&eTime, sizeof(int), 1, fp);
		fread(&navi_flg, sizeof(int), 1, fp);
		alive.clear();
		fread(&l, sizeof(int), 1, fp);
		for(i =0; i < l; i++){
			fread(&j, sizeof(int), 1, fp);
			alive.push_back(j);
		}
		zombie.clear();
		fread(&l, sizeof(int), 1, fp);
		for(i =0; i < l; i++){
			fread(&j, sizeof(int), 1, fp);
			zombie.push_back(j);
		}
	}
	
};

class	station{
public:
	class entrance{
		public:
			int process;
			int pass;
			int id;
	};
	int			id;					//駅番号
	int			capa;				//駅収容人数
	int			num;				//駅構内人数
	std::vector<entrance>	entrances;			//駅出口
	int			len_dijkstra;			//dijkstraマップ数
	double		*dijkstra;			//dijkstraマップ本体
	int			cumilative;			//累計利用者数

	station(){
		num		= 0;
		cumilative	= 0;
	}
	~station(){
		entrances.clear();
		std::vector<entrance>().swap(entrances);
	}

	bool	check(int p){				//駅への入場判定
		if(num >= capa) return false;
		for(unsigned int i=0; i < entrances.size();i++){
			if(entrances[i].id ==p &&entrances[i].process > entrances[i].pass){
				entrances[i].pass++;
				num++;
				cumilative++;
				return true;
			}
		}
		return false;
	}
	void	write2bin(FILE	*fp){
		int	i,l;
		fwrite(&id, sizeof(int), 1, fp);
		fwrite(&capa, sizeof(int), 1, fp);
		fwrite(&num, sizeof(int), 1, fp);
		fwrite(&cumilative, sizeof(int), 1, fp);
		l	= entrances.size();
		fwrite(&l, sizeof(int), 1, fp);
		for(i = 0; i < l; i++){
			fwrite(&entrances[i], sizeof(entrance), 1, fp);
		}
		fwrite(&len_dijkstra, sizeof(int), 1, fp);
		for(i = 0; i < len_dijkstra; i++){
			fwrite(&dijkstra[i], sizeof(double), 1, fp);
		}
	}	

	void	read2bin(FILE	*fp){
		int	i,l;
		fread(&id, sizeof(int), 1, fp);
		fread(&capa, sizeof(int), 1, fp);
		fread(&num, sizeof(int), 1, fp);
		fread(&cumilative, sizeof(int), 1, fp);
		fread(&l, sizeof(int), 1, fp);
		entrances.clear();
		for(i = 0; i < l; i++){
			entrance e;
			fread(&e, sizeof(entrance), 1, fp);
			entrances.push_back(e);
		}

		fread(&len_dijkstra, sizeof(int), 1, fp);
		dijkstra	= new double[len_dijkstra];
		for(i = 0; i < len_dijkstra; i++){
			fread(&dijkstra[i], sizeof(double), 1, fp);
		}
	}	
};

class	stop_point{
public:
	int				id;
	int				len_dijkstra;
	double			*dijkstra;			//dijkstraマップ本体
	stop_point(){
		id	= -1;
	}
	~stop_point(){
		if( id != -1)
			delete [] dijkstra;
		id	= -1;
	}
};


class train{
public:
	int		tid;
	int		capa;
	int		ride;
	unsigned int	pt;
	class schedule{
		public:
			int time, ride, station;
	};   
 
	std::vector< schedule> schedules;
	std::map<int, bool> customer; 
	train(){
		ride	= 0;
	}
	static bool tmsort(schedule a, schedule b){
		if(a.time < b.time)	return true;
		return false;
	}
	std::vector<int> get_customer_list(){
		std::map<int, bool >::iterator itrF;
		std::vector<int> list;
		for(itrF = customer.begin(); itrF != customer.end(); itrF++){
			list.push_back(itrF->first);
		}
		return list;
	}
	void sort(){
		std::stable_sort(schedules.begin(), schedules.end(), tmsort);
		pt	= 0;
	}
	bool checkTime(int t){
		for(;pt < schedules.size(); pt++){
			if(schedules[pt].time < t)	continue;
			if(schedules[pt].time > t)	break;
			if(capa <= ride) break;
			return true;
		}
		return false;
	}
	void resetForward(int t){
		while( schedules.size()){
			int i = schedules.size()-1;
			if(schedules[i].time >= t)
				schedules.pop_back();
			else break;
		}
	}
	bool checkRide(int st, int dest){
		if(ride >= capa) return false;
		if(schedules[pt].station==st 
		&& customer.count(dest) > 0)return true;
		return false;
	}
	void rideOn(){
		ride++;
		schedules[pt].ride++;
	}

	void	clear(){
		schedules.clear();
		std::vector<schedule>().swap(schedules);
		customer.clear();
	}

	void	write2bin(FILE	*fp){
		int	size;
		fwrite(&tid, sizeof(int), 1, fp);
		fwrite(&capa, sizeof(int), 1, fp);
		fwrite(&ride, sizeof(int), 1, fp);
		fwrite(&pt, sizeof(int), 1, fp);
		size	= schedules.size();
		fwrite(&size, sizeof(int), 1, fp);
		for(int i = 0; i < size; i++){
			fwrite(&schedules[i], sizeof(schedule), 1, fp);
		}
		std::map<int, bool >::iterator itrF;
		size	= 0;
		for(itrF = customer.begin(); itrF != customer.end(); itrF++){
			size++;
		}
		fwrite(&size, sizeof(int), 1, fp);
		for(itrF = customer.begin(); itrF != customer.end(); itrF++){
			fwrite(&itrF->first, sizeof(int), 1, fp);
		}
	}

	void	read2bin(FILE	*fp){
		int	size;
		fread(&tid, sizeof(int), 1, fp);
		fread(&capa, sizeof(int), 1, fp);
		fread(&ride, sizeof(int), 1, fp);
		fread(&pt, sizeof(int), 1, fp);
		fread(&size, sizeof(int), 1, fp);
		for(int i = 0; i < size; i++){
			schedule sche;
			fread(&sche, sizeof(schedule), 1, fp);
			schedules.push_back(sche);
		}
		fread(&size, sizeof(int), 1, fp);
		for(int i = 0; i < size; i++){
			int j;
			fread(&j, sizeof(int), 1, fp);
			customer[j]=true;
		}
	}

};
class diagram{
public:
	std::vector<train> diagrams;
	std::vector<int>	current_index;
	int len_current;
	int slide;

	void	addTrain(char *buff, int time =0 ){
		int		pt, tid, capa, l, grp;
		int		ind = -1;
		if(buff[0]=='#')return;
		train *t;
		sscanf(buff, "%d %d %n", &tid, &capa, &pt);
		tid--;
		buff+=pt;
		for(unsigned int i = 0; i < diagrams.size(); i++){
			if(diagrams[i].tid==tid){
				ind	= i;
				t	= &diagrams[i];
				break;
			}
		}
		if(ind == -1){
			train tr;
			diagrams.push_back(tr);
			t=&diagrams[diagrams.size()-1];
		}
		t->resetForward(time);
		t->tid	= tid;
		t->capa	= capa;
		sscanf(buff, "%d%n", &l, &pt);
		buff += pt;
		//客層の変更
		t->customer.clear();
		for(int i = 0; i < l; i++){
			sscanf(buff, "%d%n", &grp, &pt);
			grp--;
			t->customer[grp] = true;
			buff+= pt;
		}
		//スケジュールの変更
		sscanf(buff, "%d%n", &l, &pt);
		buff+= pt;
		for(int i = 0; i < l; i++){
			train::schedule sche; 
			sscanf(buff, "%d:%d%n", &sche.time, &sche.station, &pt);
			sche.ride	= 0;
			sche.time--;
			sche.station--;
			if(sche.time >= time)
			   	t->schedules.push_back(sche);
			buff+= pt;
		}
		t->sort();
 

	}
	//時刻tに駅発着の電車を調べる
	void	setCurrent(int t){
		current_index.clear();
		for(unsigned int i = 0; i < diagrams.size(); i++){
			bool flg	= diagrams[i].checkTime(t);
			if(flg) current_index.push_back(i);
		}
		len_current = current_index.size();
		slide = t;
	}
	void resetSchedule(int t){
		for(unsigned int i = 0; i < diagrams.size(); i++){
			diagrams[i].resetForward(t);
		}
	}
	int		chkRide(int st, int dest){
		static int slide = 0;
		int t;
		slide++;
		for(int i = 0; i < len_current; i++){
			t = current_index[(i + slide) % len_current];
			if(diagrams[t].checkRide(st, dest))return t;
		}
		return -1;
	}
	void	clear(){
		for(unsigned int i = 0; i < diagrams.size(); i++){
			diagrams[i].clear();
		}
		diagrams.clear();
		std::vector<train>().swap(diagrams);
	}

	void	write2bin(FILE	*fp){
		int	size;
		size	= diagrams.size();
		fwrite(&size, sizeof(int), 1, fp);
		for(int i = 0; i < size; i++){
			diagrams[i].write2bin(fp);
		}
	}
	void	read2bin(FILE	*fp){
		int	size;
		for(unsigned int i = 0; i < diagrams.size();i++){
			diagrams[i].clear();
		}
		diagrams.clear();
		fread(&size, sizeof(int), 1, fp);
		for(int i = 0; i < size; i++){
			train tr;
			tr.read2bin(fp);
			diagrams.push_back(tr);
		}
	}


};



class	edge_info{
public:
	int			src;				//接続元
	int			dest;				//接続先
	double		dist;				//距離
	double		width;				//道幅
	int			capacity;			//
	double		alpha;				//人口密度に対する感度	高い程鋭敏
	double		beta;				//
	double		gamma;				//徐行割合	0-1.0
	double		e_cost;				//edge_cost for dijkstra
	int			scnt;				//signal wait agent counter （信号待ち）
	int			g_cnt;				//gate待ちカウンタ
	int			g_capa;				//gate待ちキャパシティ
	int			g_wait;				//gate待ち時間


	std::vector<int>	user;
	std::pair<int, double>	flg;	//traffic_reguration
	edge_info		*rev;
	edge_info(){
		scnt		= 0;
		flg.first	= 0;
		flg.second	= -1;
		rev			= NULL;
		alpha		= ALPHA;
		beta		= BETA;
		gamma		= GAMMA;
		e_cost		= 1.0;
		g_cnt		= 0;
		g_capa		= -1;
		g_wait		= -1;
	};
	~edge_info(){
		std::vector<int>().swap(user);
		rev	= NULL ;
	};
	edge_info clone(){
		edge_info copy;
		copy.dest		= dest;				//接続先
		copy.dist		= dist;
		copy.width		= width;
		copy.capacity		= copy.capacity;				//
		copy.flg		= flg;
		copy.alpha		= alpha;
		copy.beta		= beta;
		copy.gamma		= gamma;
		copy.e_cost		= e_cost;
		copy.g_cnt		= g_cnt;
		copy.g_wait		= g_wait;
		copy.g_capa		= g_capa;
		for(int i = 0; i < (int)user.size(); i++ ){
			copy.user.push_back(user[i]);
		}
		
		return copy;
	}

};


class	graph_tensor{
private:

public:

	class	coord{
		public:
		double	lat;
		double	lng;
	};
	class	curve{
		public:
		double	lat;
		double	lng;
		double	dist;
	};
	int		M;

	edge_info	**edge;
	int			*vec;
	void		readGraph(char	*fn);
	void		readGate(char *fn);
	void		dijkstra(station &st);
	void		dijkstra(stop_point &st);
	edge_info*	relay(int p, station st, int grp, navigation navi);
	edge_info*	relay(int p, stop_point st, int grp, navigation navi);

	edge_info*	relayDirect(int p, station st);
	edge_info*	relayDirect(int p, stop_point st);

	double		Hubeny(double lat1, double lng1, double lat2, double lng2);
	double		deg2rad(double deg);
	coord		*points;
	void		readPoints(char *fn);
	void		readHeight(char *fn);
	void		readCurve(char *fn);
	void		readAlpha(char *fn);
	void		readGamma(char *fn);
	void		readEdgeCost(char *fn); 
	int			diameter;
	int			flg_points;
	int			flg_height;
	double		*height;

	std::map <int, std::map <int, std::vector<curve> > > curves;
	graph_tensor(){
		flg_points	= 0;
		flg_height	= 0;
		diameter	= 0;
	}

	graph_tensor	clone(){
		int	i, j;
		graph_tensor	copy;
		copy			= *this;
		copy.M			= M;		
		copy.vec		= new int[M];
		copy.edge		= new edge_info * [M];


		for(i = 0; i < M; i++){
			copy.vec[i]	= vec[i];
			copy.edge[i]	= new edge_info[vec[i]];
			for(j = 0; j < vec[i];j++){
				copy.edge[i][j]		= edge[i][j].clone();
				copy.edge[i][j].rev	= NULL;
				

			}
		}
		copy.rev_check();	
		return copy;
	}

	//転置グラフの作成
	//Dijkstraを実行時に使用	
	graph_tensor	inverse_graph(){
		int	i, j, dest;
		graph_tensor	copy;
		copy.M			= M;
		copy.edge		= new edge_info * [M];
		copy.vec		= new int[M];
		for(i = 0; i < M; i++){
			copy.vec[i]		= 0;
		}
		for(i = 0; i < M; i++){
			for(j = 0; j < vec[i]; j++){
				dest	= edge[i][j].dest;
				copy.vec[dest]++;
			}
		}
		for(i = 0; i < M; i++){
			copy.edge[i]	= new edge_info [copy.vec[i]];
			copy.vec[i]		= 0;
		}
	
		for(i = 0; i < M; i++){
			for(j = 0; j < vec[i]; j++){
				dest	= edge[i][j].dest;

				copy.edge[dest][copy.vec[dest]] = edge[i][j];
				copy.edge[dest][copy.vec[dest]].dest	= i;
				copy.vec[dest]++;

			}
		}	
		copy.rev_check();	
		return copy;
	}

	//バイナリ出力
	void	write2bin(FILE	*fp){
		int	i, j, k;
		fwrite(&M, sizeof(int), 1, fp);
		for(i = 0; i < M; i++){
			fwrite(&vec[i], sizeof(int), 1, fp);
			for(j = 0; j < vec[i];j++){
				int ucnt	= edge[i][j].user.size();
				fwrite(&edge[i][j].src, sizeof(int), 1, fp);
				fwrite(&edge[i][j].dest, sizeof(int), 1, fp);
				fwrite(&edge[i][j].dist, sizeof(double), 1, fp);
				fwrite(&edge[i][j].width, sizeof(double), 1, fp);
				fwrite(&ucnt, sizeof(int), 1, fp);
				fwrite(&edge[i][j].capacity, sizeof(int), 1, fp);
				fwrite(&edge[i][j].flg.first, sizeof(int), 1, fp);
				fwrite(&edge[i][j].flg.second, sizeof(double), 1, fp);

				fwrite(&edge[i][j].alpha, sizeof(double), 1, fp);
				fwrite(&edge[i][j].beta, sizeof(double), 1, fp);
				fwrite(&edge[i][j].gamma, sizeof(double), 1, fp);
				fwrite(&edge[i][j].e_cost, sizeof(double), 1, fp);

				fwrite(&edge[i][j].g_cnt, sizeof(int), 1, fp);
				fwrite(&edge[i][j].g_capa, sizeof(int), 1, fp);
				fwrite(&edge[i][j].g_wait, sizeof(int), 1, fp);
				for(k = 0; k < (int) edge[i][j].user.size(); k++){
					fwrite(&edge[i][j].user[k], sizeof(int), 1, fp);
				}
			}
		}
		fwrite(&flg_points, sizeof(int), 1, fp);
		if(flg_points	== 1){
				fwrite(points, sizeof(coord), M, fp);
		}
		fwrite(&flg_height, sizeof(int), 1, fp);
		if(flg_height	== 1){
				fwrite(height, sizeof(double), M, fp);
		}
	}

	//バイナリ入力(読み込み)
	void	read2bin(FILE	*fp){
		int	i, j, k, u;
		fread(&M, sizeof(int), 1, fp);
		vec	= new int [M];
		edge	= new edge_info * [M];
		for(i = 0; i < M; i++){
			fread(&vec[i], sizeof(int), 1, fp);
			edge[i]	= new edge_info [vec[i]];
			for(j = 0; j < vec[i];j++){
				int ucnt;
				fread(&edge[i][j].src, sizeof(int), 1, fp);
				fread(&edge[i][j].dest, sizeof(int), 1, fp);
				fread(&edge[i][j].dist, sizeof(double), 1, fp);
				fread(&edge[i][j].width, sizeof(double), 1, fp);
				fread(&ucnt, sizeof(int), 1, fp);
				fread(&edge[i][j].capacity, sizeof(int), 1, fp);
				fread(&edge[i][j].flg.first, sizeof(int), 1, fp);
				fread(&edge[i][j].flg.second, sizeof(double), 1, fp);
				fread(&edge[i][j].alpha, sizeof(double), 1, fp);
				fread(&edge[i][j].beta, sizeof(double), 1, fp);
				fread(&edge[i][j].gamma, sizeof(double), 1, fp);
				fread(&edge[i][j].e_cost, sizeof(double), 1, fp);
				fread(&edge[i][j].g_cnt, sizeof(int), 1, fp);
				fread(&edge[i][j].g_capa, sizeof(int), 1, fp);
				fread(&edge[i][j].g_wait, sizeof(int), 1, fp);
				for(k = 0; k < ucnt; k++){
					fread(&u, sizeof(int), 1, fp);
					edge[i][j].user.push_back(u);
				}
			}
		}
		fread(&flg_points, sizeof(int), 1, fp);
		if(flg_points   == 1){
				points  = new coord [M];
				fread(points, sizeof(coord), M, fp);
		}
		fread(&flg_height, sizeof(int), 1, fp);
		if(flg_height   == 1){
				height  = new double [M];
				fread(height, sizeof(double), M, fp);
		}
	}


	int	setRegulation(int from, int to, int tp, double val){
		for(int i = 0; i < vec[from]; i++){
			if(to == edge[from][i].dest){
				if(edge[from][i].flg.first==tp && edge[from][i].flg.second==val){
					return 0;
				}else if(tp==2 && edge[from][i].width==val){
					//default値に道幅変更
					edge[from][i].flg.first	 = 0;
					edge[from][i].flg.second	= -1.0;
					return -1;
				}else{
					edge[from][i].flg.first	 = tp;
					edge[from][i].flg.second	= val;
					return 1;
				}
			}
		}
		return -1;
	}

	int setGate(int from,int to, int capa, int wait){
		for(int i = 0; i < vec[from]; i++){
			if(to == edge[from][i].dest){
				edge[from][i].g_capa	= capa;
				edge[from][i].g_wait	= wait;
				return 1;
			}
		}
		return -1;
	}
	int setGate(char *buff){
		int from, to, capa, wait;
		sscanf(buff, "%d %d %d %d", &from, &to, &capa, &wait);
		from--;
		to--;
		return setGate(from, to, capa,wait);
	}

	int delGate(int from,int to){
		for(int i = 0; i < vec[from]; i++){
			if(to == edge[from][i].dest){
				edge[from][i].g_capa	= -1;
				edge[from][i].g_wait	= -1;
				return 1;
			}
		}
		return -1;
	}
	int delGate(char *buff){
		int from, to;
		sscanf(buff, "%d %d", &from, &to);
		from--;
		to--;
		return delGate(from, to);
	}


	void resetGate(){
		for(int i = 0; i < M; i++){
			for(int j = 0; j < vec[i]; j++){
				edge[i][j].g_cnt	= 0;
				edge[i][j].g_capa	= -1;
				edge[i][j].g_wait	= -1;
			}
		}
	}
	int		setAlpha(int from, int to, double value){
		for(int i = 0; i < vec[from]; i++){
			if(to == edge[from][i].dest){
					edge[from][i].alpha = ALPHA *value;
					return 0;
			}
		}
		return -1;
	}

	int		setAlpha(char *buff){
		int from, to;
		double value;
		sscanf(buff, "%d %d %lf", &from, &to, &value);
		from--;
		to--;
		return setAlpha(from, to, value);
	}


	int		setBeta(int from, int to, double value){
		for(int i = 0; i < vec[from]; i++){
			if(to == edge[from][i].dest){
					edge[from][i].beta = BETA * value;
					return 0;
			}
		}
		return -1;
	}

	int		setBeta(char *buff){
		int from, to;
		double value;
		sscanf(buff, "%d %d %lf", &from, &to, &value);
		from--;
		to--;
		return setBeta(from, to, value);
	}


	int		setGamma(int from, int to, double value){
		for(int i = 0; i < vec[from]; i++){
			if(to == edge[from][i].dest){
					edge[from][i].gamma = GAMMA *value;
					return 0;
			}
		}
		return -1;
	}

	int		setGamma(char *buff){
		int from, to;
		double value;
		sscanf(buff, "%d %d %lf", &from, &to, &value);
		from--;
		to--;
		return setGamma(from, to, value);
	}

	int		setEdgeCost(int from, int to, double value){
		for(int i = 0; i < vec[from]; i++){
			if(to == edge[from][i].dest){
					edge[from][i].e_cost = value;
					return 0;
			}
		}
		return -1;
	}
	int		setEdgeCost(char *buff){
		int		from, to;
		double	val;
		sscanf(buff, "%d %d %lf", &from, &to, &val);
		from--;
		to--;
		return setEdgeCost(from, to, val);
	}

	void resetEdgeCost(){
		for(int i = 0; i < M; i++){
			for(int j = 0; j < vec[i]; j++){
				edge[i][j].e_cost	= 1.0;
			}
		}
	}
	void	clearAlpha(){
		for(int i = 0; i < M; i++){
			for(int j = 0; j < vec[i]; j++){
				edge[i][j].alpha  = ALPHA;		
			}
		}
	}

	void	clearBeta(){
		for(int i = 0; i < M; i++){
			for(int j = 0; j < vec[i]; j++){
				edge[i][j].beta  = BETA;		
			}
		}
	}

	void	clearGamma(){
		for(int i = 0; i < M; i++){
			for(int j = 0; j < vec[i]; j++){
				edge[i][j].gamma  = GAMMA;		
			}
		}
	}

	void	clear(){
		for(int i = 0; i < M; i++){
			delete [] edge[i];
		}
		delete [] vec;
		delete [] edge;
		if(flg_points==1){
			delete [] points;
			flg_points=0;
		}
		if(flg_height==1){
			delete [] height;
			flg_height=0;
		}
	}

	void	rev_check(){
		int	i, j, m, dest;
		for (i = 0; i < M; i++){
			for(j = 0; j < vec[i]; j++){
				edge[i][j].rev	= NULL;
				dest	= edge[i][j].dest;
				for(m = 0; m < vec[dest]; m++){
					if(edge[dest][m].dest==i){
						edge[i][j].rev	= &edge[dest][m];
						break;
					}
				}
			}
		}
	}
	void	point2curve(){
		for(int fr = 0; fr < M; fr++){
			for(int	ind	= 0; ind < vec[fr]; ind++){
				int	to	= edge[fr][ind].dest;
				if(curves[fr].count(to)== 0){
					curve	cv;
					cv.lat	= points[fr].lat;
					cv.lng	= points[fr].lng;
					cv.dist	= 0.0;
					curves[fr][to].push_back(cv);
					cv.lat	= points[to].lat;
					cv.lng	= points[to].lng;
					cv.dist	= edge[fr][ind].dist;
					curves[fr][to].push_back(cv);
				}
			}
		}
	}
	
};


//軌跡　start:移動起点 time:時刻 dist:移動距離
class	trajectory{
public:
	int	start, time;
	double	dist;
};

//エージェント情報
class user_info{
	public:
	int		id;			//uid
	int		from;			//移動中の道路起点（交差点）
	int		to;			//移動中の道路終点（交差点）
	int		goal;			//目的地
	int		group;			//グループ
	int		born;			//登場時刻
	int		dead;			//ゴール時刻
	int		tgoal;			//電車の目的地方面
	int		ttime;			//電車の搭乗時間
	int		tid;			//搭乗した電車ID
	int		wait;			//滞留カウンタ
	bool	gate;			//gate滞留フラグ
	double		speed;			//移動速度
	double		quata;			//移動中の道路残り距離
	double		delta;			//temporary移動距離
	double		dense;			//前方密度
	edge_info	*edge;			//移動中の道路ポインタ

	std::vector<trajectory>	logs;						//移動軌跡
	std::vector<std::pair<int, int> >		via;		//経由地リスト
	user_info(){
		quata	= -9999;
		dead	= -9999;
		wait	= 0;
		gate	= false;
		tid		= -9999;
		ttime	= -9999;
	};
	~user_info(){
		std::vector<trajectory>().swap(logs);
		std::vector<std::pair<int, int> >().swap(via);
	};
	user_info clone(){
		user_info	copy;
		copy		= *this;
		return copy;
	};
	void	write2bin(FILE	*fp){

		fwrite(&id, sizeof(int), 1, fp);
		fwrite(&from, sizeof(int), 1, fp);
		fwrite(&to, sizeof(int), 1, fp);
		fwrite(&goal, sizeof(int), 1, fp);
		fwrite(&group, sizeof(int), 1, fp);
		fwrite(&born, sizeof(int), 1, fp);
		fwrite(&dead, sizeof(int), 1, fp);
		fwrite(&wait, sizeof(int), 1, fp);
		fwrite(&tgoal, sizeof(int), 1, fp);
		fwrite(&ttime, sizeof(int), 1, fp);
		fwrite(&tid, sizeof(int), 1, fp);
		fwrite(&gate, sizeof(bool), 1, fp);

//		printf("%d %d %d %d %d %d\n", id, from, to, goal, born, dead);
//		return;
		fwrite(&speed, sizeof(double), 1, fp);
		fwrite(&quata, sizeof(double), 1, fp);
		fwrite(&delta, sizeof(double), 1, fp);
		fwrite(&dense, sizeof(double), 1, fp);
//		return;
		int	i, l;
		l	= logs.size();	
		fwrite(&l, sizeof(int), 1, fp);
		for(i	= 0; i < l; i++){
			fwrite(&logs[i], sizeof(trajectory), 1, fp);
		}
		l	= via.size();
		fwrite(&l, sizeof(int), 1, fp);
		for(i	= 0; i < l; i++){
			fwrite(&via[i].first, sizeof(int), 1, fp);
			fwrite(&via[i].second, sizeof(int), 1, fp);
		}

	};
	void	read2bin(FILE	*fp){
		fread(&id, sizeof(int), 1, fp);
		fread(&from, sizeof(int), 1, fp);
		fread(&to, sizeof(int), 1, fp);
		fread(&goal, sizeof(int), 1, fp);
		fread(&group, sizeof(int), 1, fp);
		fread(&born, sizeof(int), 1, fp);
		fread(&dead, sizeof(int), 1, fp);
		fread(&wait, sizeof(int), 1, fp);
		fread(&tgoal, sizeof(int), 1, fp);
		fread(&ttime, sizeof(int), 1, fp);
		fread(&tid, sizeof(int), 1, fp);
		fread(&gate, sizeof(bool), 1, fp);
	
		fread(&speed, sizeof(double), 1, fp);
		fread(&quata, sizeof(double), 1, fp);
		fread(&delta, sizeof(double), 1, fp);
		fread(&dense, sizeof(double), 1, fp);
		int	i, l;
		trajectory	trj;
		fread(&l, sizeof(int), 1, fp);
		for(i	= 0; i < l; i++){
			fread(&trj, sizeof(trajectory), 1, fp);
			logs.push_back(trj);
		}
		fread(&l, sizeof(int), 1, fp);
		for(i	= 0; i < l; i++){
			std::pair<int, int> tmp;
			fread(&tmp.first, sizeof(int), 1, fp);
			fread(&tmp.second, sizeof(int), 1, fp);
			via.push_back(tmp);
		}
		
	};
	
};

class	group{
public:
	std::map<int, std::vector<int>  > member;
	std::vector <int> mid, gid;
	void	clear(){
		std::map<int, std::vector<int> >::iterator itrF;
		for( itrF = member.begin(); itrF != member.end(); itrF++){
			int grp = itrF->first;
			member[grp].clear();
			std::vector<int>().swap(member[grp]);
			member.erase(grp);
		}
	}
	void	build(int N, user_info *uf){
		for(int i = 0; i < N; i++){
			member[uf[i].group].push_back(i);
		}
	}
	std::vector<int>   get_memberId(int grp){
/*
		mid.clear();
		std::vector<int>().swap(mid);
*/
		
		return member[grp];
	}

	std::vector<int>   get_groupId(){
		gid.clear();
		std::vector<int>().swap(gid);
		std::map<int, std::vector<int> >::iterator itrF;
		for( itrF = member.begin(); itrF != member.end(); itrF++){
			int grp = itrF->first;
			gid.push_back(grp);
		}
		return gid;
	}
};

class simulator{
public:
	//N	:エージェント数
	//M	:交差点数
	//G	:グループ数
	int							N, M, G;	
	simulation_setting			sset;
	graph_tensor				graph;
	user_info					*uf;
	//std::vector<user_info>	uf;
	std::vector<station>		st;
	stop_point					*sp;
	navigation					navi;
	signal						sig;
	group						grp;
	diagram						dia;
	time_bomb					tb;
	std::vector<int>			navi_index;			
	std::vector<navigation>		navi_log;
	std::vector<traffic_log>	traffic_logs;
	std::vector<od_change_log>	od_logs;
	std::pair<int, double>		*born;
	std::pair<int, double>		*quata;
	int							bpt;
	signage						kanban;

	char	output[STR_LENGTH];
	char	output_resume[STR_LENGTH];
	char	output_dijkstra[STR_LENGTH];
	char	output_traffic[STR_LENGTH];

	void	readData(char *fn);
	void	readStation(char *fn);
	void	readNavigation(char *fn);
	void	readDiagram(char *fn);
	void	run(simulation_setting &sset, user_info* &uf, graph_tensor &graph, station * &st, navigation &navi);
	void	readTrafficRegulation(char *fn);
	void	editTrafficRegulation(int from, int to, int tp, double val);
	void	shrinkTrafficRegulation();
	void	resetTrafficRegulation();
	void	changeDestination(char *buff);
	void	chkGate();
	void	chkSignage(user_info &u);
	void	updateStation();
	void	readTrigger(char *fn);
	void	savelog(char *fn, user_info* &uf);
	void	save_userlog(char *fn, user_info* &uf);
	void	save_od_hist(char *fn);
	void	save_edgelog(char *fn, graph_tensor graph);
	void	save_stationlog(char *fn, std::vector<station> st);
	void	save_trainlog(char *fn);
	void	save_navi(char *fn, std::vector<int> index, std::vector<navigation> log);	
	void	save_binary(char *fn);
	void	load_binary(char *fn);
	int		iterate();
	void	writeTrafficLog(char *fn);
	void	bombExplode();
	void	initialize(int argc,char **argv );
	void	reset_condition();
	//simulator();
	~simulator(){
		delete []	born;
		delete []	quata;
	};
};


void	writeDijkstra_map(char	*fn,std::vector<station> st);

