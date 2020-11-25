# python main.py --save --configfn config_speed.ini &
# python main.py --save --configfn config_edge.ini &
# python main.py --save --configfn config_edge_wo_T0.ini &
# python main.py --save --configfn config_goal.ini &
# python main.py --save --configfn config_goal_cum.ini &
# python main.py --save --configfn config_speed_w_V0.ini
# python main.py --save --configfn config_time.ini &
# python main.py --save --configfn config_total_time_once.ini &

# 手元で学習するために
python main.py --save --configfn config.ini

# python main.py --save --configfn config_zero_one.ini
# python main.py --save --configfn config_time_once.ini
# python main.py --save --configfn config_time.ini
# python main.py --save --configfn config_zero_one.ini --checkpoint --inputfn logs/Curriculum/model

for i in `seq 1 10000`
do
# ファイルがあれば終了する
if [ -f terminate.txt ]
	then
		echo "終了します : terminate.txt"
		break
	fi
echo "$i 回目のループです"
python main.py --save --configfn config.ini --checkpoint --inputfn logs/Curriculum/model ; cp logs/Curriculum/model logs/Curriculum/model_$i ; cp logs/Curriculum/model.score logs/Curriculum/model_$i.score
done

# 共用計算機で学習したモデルを動画にするために
# python main.py --test --configfn config.ini --checkpoint --inputfn logs/Curriculum20200826/model
# python main.py --test --configfn config.ini --checkpoint --inputfn logs/Curriculum20200824170428/model
# ベースラインを動画にするために
# python main.py --test --configfn config.ini
# python main.py --base --configfn config.ini

# 手元で学習したモデルを読み込むテスト
# python main.py --test --configfn config.ini --checkpoint --inputfn logs/Curriculum/model


# 再現実験
# ../simulator/simulator ../mkUserlist/data/kawaramachi/agentlist.txt ../mkUserlist/data/kawaramachi/graph.twd ../mkUserlist/data/kawaramachi/goallist.txt -B logs/edge/sim_result_1/history_events.txt -o tmp_result -e 9000 -l 10 -S
# cp logs/edge/sim_result_1/history_events.txt ./tmp_result
# ../simulator/simulator ../mkUserlist/data/kawaramachi/agentlist.txt ../mkUserlist/data/kawaramachi/graph.twd ../mkUserlist/data/kawaramachi/goallist.txt -B results/event.txt -o tmp_result2 -e 9000 -l 10 -S
# cp results/event.txt ./tmp_result2
# ../simulator/simulator data/agentlist.txt data/graph.twd data/goallist.txt -B results/event.txt -o result_base -e 9000 -l 10 -S
# cp results/event.txt ./result_base/base_event.txt

# 再現実験（共用計算機で学習したモデルを動画にするために）
# ../simulator/simulator ../mkUserlist/data/kawaramachi/agentlist.txt ../mkUserlist/data/kawaramachi/graph.twd ../mkUserlist/data/kawaramachi/goallist.txt -B results/event.txt -o tmp_result -e 9000 -l 10 -S
# cp results/event.txt ./tmp_result

# 動画生成
# cd ../mkUserlist
# ./analyselog.py -d /home/shimizu/project/2020/escape_navi/navi_curriculum/tmp_result
# ../mkUserlist/analyselog.py -d /home/shimizu/project/2020/escape_navi/navi_curriculum/tmp_result -u /home/shimizu/project/2020/escape_navi/mkUserlist/data/kawaramachi
