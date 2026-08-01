import sys, numpy as np, torch, random
sys.path.insert(0,"/home/pandya.kei/CS6120/run")
from _libero_probe import make_probe_vec
from lerobot.envs import make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import ACTION
hf="HuggingFaceVLA/smolvla_libero"
vec = make_probe_vec("libero_goal", 1, 300); vec.reset(seed=7)
ec=LiberoEnvConfig(task="libero_goal",task_ids=[1])
pc=PreTrainedConfig.from_pretrained(hf); pc.pretrained_path=hf; pc.device="cuda:0"
policy=make_policy(cfg=pc,env_cfg=ec); policy.eval()
pre,post=make_pre_post_processors(policy_cfg=pc,pretrained_path=hf,preprocessor_overrides={"device_processor":{"device":"cuda"}})
epre,epost=make_env_pre_post_processors(env_cfg=ec,policy_cfg=pc)
vec.set_attr("init_state_id",0); vec.set_attr("task_description","Put the bowl on the stove")
random.seed(7); np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
policy.reset(); obs,info=vec.reset(seed=7)
print(">> calling probe_state after reset")
pr=vec.call("probe_state")[0]; print(">> probe OK hash",pr["hash"],"nposes",len(pr["poses"]))
for i in range(3):
    o=preprocess_observation(obs); o["task"]=list(vec.call("task_description"))
    o=epre(o); o=pre(o)
    with torch.inference_mode(): a=policy.select_action(o)
    a=post(a); a=epost({ACTION:a})[ACTION]; anp=a.to("cpu").numpy()
    obs,r,term,trunc,info=vec.step(anp); print("  step",i,"succ",info.get("is_success"))
print(">> OK"); vec.close()
