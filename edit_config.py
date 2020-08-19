with open("models.txt","r") as f:
    models = [i.strip() for i in f.readlines()]
print(models)

for model in models:
    base_format = """
# Settings for simulation
[SIMULATION]
datadirlistfn=train_datadirlist.txt
sim_time=15000
interval=600
actionfn=actions39.txt
[TRAINING]
obs_step=4
obs_degree=12
num_processes=16
num_advanced_step=25
num_episodes=200
resdir=logs/%s
outputfn=model
gamma=0.99
loss_coef=0.5
entropy_coef=0.01
max_grad_norm=0.5
flg_reward=%s"""%(model,model)
    print(base_format)
    with open("config/config_%s.ini"%model,"w") as f:
        f.write(base_format)
