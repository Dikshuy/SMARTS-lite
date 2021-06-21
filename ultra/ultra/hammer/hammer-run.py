import argparse, os, numpy as np, torch, json
from numpy.lib.function_base import hamming 

from itertools import count
from tensorboardX import SummaryWriter
from pathlib import Path 
from pettingzoo.mpe import simple_spread_v2 
from pettingzoo.sisl import multiwalker_v6 
from npy_append_array import NpyAppendArray

from hammer import PPO 
from utils import read_config 

def preprocess_one_obs(obs, which=1, limit=10): 
    agent = "agent_" + str(which) 
    obs[agent][limit:] = [0.]*(len(obs["agent_0"])-(limit)) 
    return obs 

def preprocess_obs(obs, limit): 
    for i in obs: 
        obs[i] = obs[i][:limit] 
    return obs 

def run(args): 
    if args.partialobs: 
        print("Using Partial Observations") 
    
    if args.heterogeneity: 
        print("Using Heterogeneous Local Agents") 

    if args.envname == "cn": 
        env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=args.maxcycles) 
        env.reset()
        agents = [agent for agent in env.agents] 
        if args.heterogeneity: 
            obs_dim = len(preprocess_one_obs(env.reset(), limit=args.limit)["agent_0"]) 
        elif args.partialobs:
            obs_dim = len(preprocess_obs(env.reset(), limit=args.limit)["agent_0"]) 
        else:
            obs_dim = env.observation_spaces[env.agents[0]].shape[0]

        action_dim = env.action_spaces[env.agents[0]].n 
        agent_action_space = env.action_spaces[env.agents[0]] 
    
    elif args.envname == "mw": 
        env = multiwalker_v6.parallel_env(n_walkers=args.nagents) 
        env.reset()
        agents = [agent for agent in env.agents] 
        obs_dim = env.observation_spaces[env.agents[0]].shape[0]        
        action_dim = env.action_spaces[env.agents[0]].shape[0] 
        agent_action_space = env.action_spaces[env.agents[0]]


    config = read_config(args.config) 
    if not config:
        print("config required")
        return
    
    random_seed = args.randomseed 
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed) 
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    name = "--".join([
        "env_" + args.envname, 
        "partialobs_" + str(args.partialobs), 
        "heterogeneity_" + str(args.heterogeneity), 
        "n_" + str(args.nagents), 
        "dru_" + str(args.dru_toggle), 
        "meslen_" + str(args.meslen), 
        "sharedparams_" + str(args.sharedparams), 
        "randomseed_" + str(args.randomseed) 
    ]) 

    # name = "exp"
    
    expname = name if args.expname == None else args.expname 
    
    writer = SummaryWriter(logdir=os.path.join("./save/", expname, args.logdir)) 
    # log_dir = Path('./logs/HAMMER-gradient-analysis/')
    # for i in count(0):
    #     temp = log_dir/('run{}'.format(i)) 
    #     if temp.exists():
    #         pass
    #     else:
    #         writer = SummaryWriter(logdir=temp)
    #         log_dir = temp
    #         break

    betas = (0.9, 0.999)

    HAMMER = PPO(
        agents=agents,
        single_state_dim=obs_dim, 
        single_action_dim=action_dim,
        meslen = args.meslen, 
        n_agents=len(agents), # required for discrete messages
        lr=config["lr"],
        betas=betas,
        gamma = config["gamma"],
        K_epochs=config["K_epochs"],
        eps_clip=config["eps_clip"],        
        actor_layer=config["actor_layer"],
        critic_layer=config["critic_layer"], 
        dru_toggle=args.dru_toggle, 
        is_discrete=config["is_discrete"], 
        sharedparams=0
    ) 
    if args.eval: 
        HAMMER.load(args.eval_path)

    if args.dru_toggle: 
        print("Using DRU") 
    else: 
        print("Not Using DRU")

    # logging variables

    if args.heterogeneity: 
        obs = preprocess_one_obs(env.reset(), limit=args.limit) 
    elif args.partialobs: 
        obs = preprocess_obs(env.reset(), limit=args.limit)
    else:  
        obs = env.reset() 

    i_episode = 1 
    episode_rewards = 0 
    
    NUMPY_SAVE_FILE = []

    for timestep in count(1):
        actions, messages, local_state = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory) 
        
        if args.envname == "mw": 
            actions = {agent : np.clip(actions[agent], agent_action_space.low, agent_action_space.high) for agent in agents}     
        next_obs, rewards, is_terminals, infos = env.step(actions) 

        HAMMER.memory_record(rewards, is_terminals)
        episode_rewards += np.mean(np.array(list(rewards.values()))) 

        # update if its time
        if (not args.eval) and (timestep % config["update_timestep"] == 0): 
            HAMMER.update() 
            [mem.clear_memory() for mem in HAMMER.memory]
            HAMMER.global_memory.clear_memory() 

        if args.partialobs: 
            next_obs = preprocess_obs(next_obs, limit=args.limit) 
        elif args.heterogeneity: 
            next_obs = preprocess_one_obs(next_obs, limit=args.limit) 
        obs = next_obs

        grad_values = local_state.grad.numpy()
        local_state_values = local_state.detach().numpy()
        NUMPY_SAVE_FILE.append(np.array([grad_values, local_state_values]))

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):

            # recording episodic rewards per agent 
            writer.add_scalar('Episodic Reward', episode_rewards, i_episode) 
            
            # recording mean of messages of each agent 
            for i, agent in enumerate(agents): 
                for m in range(args.meslen): 
                    writer.add_scalar(str(agent)+"--"+ 'message_feature_'+str(m), np.mean(np.array(HAMMER.memory[i].messages).reshape((-1))), i_episode) 

            # save model periodically 
            if i_episode % args.saveinterval == 0: 
                if args.eval: 
                    where = os.path.join("./comm_analysis/saliency/") 
                    if not os.path.exists(where): os.makedirs(where)
                    filenames= {
                        # "hammer_states": os.path.join(where, "hammer_states.npy"), 
                        # "hammer_messages": os.path.join(where, "hammer_messages.npy"), 
                        "local_states": os.path.join(where, "local_states.npy")
                    } 

                    # npaa = NpyAppendArray(filenames["hammer_messages"]) 
                    # [npaa.append(np.array(x).reshape(1, -1)) for x in HAMMER.global_memory.messages] 

                    # npaa = NpyAppendArray(filenames["hammer_states"]) 
                    # [npaa.append(i.detach().numpy()) for i in HAMMER.global_memory.states] 

                    np.save(filenames["local_states"], NUMPY_SAVE_FILE)

                    [mem.clear_memory() for mem in HAMMER.memory]
                    HAMMER.global_memory.clear_memory() 

                else: 
                    save_dir = os.path.join("./save/", expname, args.savedir, "checkpoint_ep_"+str(i_episode)) 
                
                    if not os.path.exists(save_dir): 
                        os.makedirs(os.path.join(save_dir))  
                    HAMMER.save(save_dir) 

            if args.heterogeneity: 
                obs = preprocess_one_obs(env.reset(), limit=args.limit) 
            elif args.partialobs: 
                obs = preprocess_obs(env.reset(), limit=args.limit)
            else: 
                obs = env.reset() 
            print('Episode {} \t  Episodic reward per agent: {}'.format(i_episode, episode_rewards)) 
            episode_rewards = 0 
            i_episode += 1 


        if i_episode == args.maxepisodes+1: 
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/cn.yaml', help="config file name")

    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--envname", type=str, default='cn')
    parser.add_argument("--nagents", type=int, default=3) 
    parser.add_argument("--eval", type=int) 
    parser.add_argument("--eval_path", type=str) 

    parser.add_argument("--sharedparams", type=int) 

    parser.add_argument("--maxepisodes", type=int, default=500_000) 
    parser.add_argument("--maxcycles", type=int, default=25) 
    parser.add_argument("--partialobs", type=int) 
    parser.add_argument("--heterogeneity", type=int) 
    parser.add_argument("--limit", type=int) # 10 for cn

    parser.add_argument("--dru_toggle", type=int) # 0 for HAMMERv2 and 1 for HAMMERv3 

    parser.add_argument("--meslen", type=int, help="message length")
    parser.add_argument("--randomseed", type=int)

    parser.add_argument("--saveinterval", type=int, default=10_000) 
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="model_checkpoints/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args) 
