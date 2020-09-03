# python main.py --save --configfn config_speed.ini &
# python main.py --save --configfn config_edge.ini &
# python main.py --save --configfn config_edge_wo_T0.ini &
# python main.py --save --configfn config_goal.ini &
# python main.py --save --configfn config_goal_cum.ini &
# python main.py --save --configfn config_speed_w_V0.ini
# python main.py --save --configfn config_time.ini &
# python main.py --save --configfn config_total_time_once.ini &

# 手元で学習するために
# python main.py --save --configfn config.ini
# python main.py --save --configfn config_time_once.ini
# python main.py --save --configfn config_time.ini

# 共用計算機で学習したモデルを動画にするために
# python main.py --test --configfn config.ini --checkpoint --inputfn logs/Curriculum20200826/model
# python main.py --test --configfn config.ini --checkpoint --inputfn logs/Curriculum20200824170428/model
# ベースラインを動画にするために
# python main.py --test --configfn config.ini

# 手元で学習したモデルを読み込むテスト
python main.py --test --configfn config.ini --checkpoint --inputfn logs/Curriculum/model


# 再現実験
# ../simulator/simulator ../mkUserlist/data/kawaramachi/agentlist.txt ../mkUserlist/data/kawaramachi/graph.twd ../mkUserlist/data/kawaramachi/goallist.txt -B logs/edge/sim_result_1/history_events.txt -o tmp_result -e 9000 -l 10 -S
# cp logs/edge/sim_result_1/history_events.txt ./tmp_result
../simulator/simulator ../mkUserlist/data/kawaramachi/agentlist.txt ../mkUserlist/data/kawaramachi/graph.twd ../mkUserlist/data/kawaramachi/goallist.txt -B results/event.txt -o tmp_result2 -e 9000 -l 10 -S
cp results/event.txt ./tmp_result2

# 再現実験（共用計算機で学習したモデルを動画にするために）
# ../simulator/simulator ../mkUserlist/data/kawaramachi/agentlist.txt ../mkUserlist/data/kawaramachi/graph.twd ../mkUserlist/data/kawaramachi/goallist.txt -B results/event.txt -o tmp_result -e 9000 -l 10 -S
# cp results/event.txt ./tmp_result

# 動画生成
# cd ../mkUserlist
# ./analyselog.py -d /home/shimizu/project/2020/escape_navi/navi_curriculum/tmp_result
# ../mkUserlist/analyselog.py -d /home/shimizu/project/2020/escape_navi/navi_curriculum/tmp_result -u /home/shimizu/project/2020/escape_navi/mkUserlist/data/kawaramachi
