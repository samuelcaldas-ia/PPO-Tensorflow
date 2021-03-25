using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//
using SixLabors.ImageSharp;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
//
using NumSharp;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using CustomRandom;
//

// import gym
// import numpy as np
// import tensorflow as tf
// from policy_net import Policy_net
// from ppo import PPOTrain

var ITERATION = int(3 * 10e5);
var GAMMA = 0.95;


static object Main(){
    env = new CartPoleEnv(WinFormEnvViewer.Factory);
    env.Seed(0);
    ob_space = env.ObservationSpace;
    Policy = Policy_net("policy", env);
    Old_Policy = Policy_net("old_policy", env);
    PPO = PPOTrain(Policy, Old_Policy, gamma: GAMMA);
    saver = tf.train.Saver();

    using (var sess = tf.Session()){
        writer = tf.summary.FileWriter("./log/test", sess.graph);
        sess.run(tf.global_variables_initializer());
        saver.restore(sess, "model/model.ckpt");
        obs = env.Reset();
        reward = 0;
        success_num = 0;

        for (int iteration in range(ITERATION)){  // episode
            observations = new List<double>();
            actions = new List<double>();
            v_preds = new List<double>();
            rewards = new List<double>();
            run_policy_steps = 0
            while (true){  // run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1;
                obs = np.stack([obs]).astype(dtype: np.float32);  // prepare to feed placeholder Policy.obs
                var (act, v_pred) = Policy.act(obs: obs, stochastic: false);

                act = np.asscalar(act);
                v_pred = np.asscalar(v_pred);

                observations.Add(obs);
                actions.Add(act);
                v_preds.Add(v_pred);
                rewards.Add(reward);

                var (next_obs, reward, done, info) = env.Step(act);

                if (done) {
                    v_preds_next = v_preds["1:"] + [0];  // next state of terminate state has value 0
                    obs = env.Reset();
                    reward = -1;
                    break;
                }else{
                    obs = next_obs;
                }
            }
            writer.add_summary(tf.Summary(value:[tf.Summary.Value(tag: "episode_length", simple_value: run_policy_steps)]), iteration);
            writer.add_summary(tf.Summary(value:[tf.Summary.Value(tag:"episode_reward", simple_value:sum(rewards))]), iteration);

            // end condition of test
            if (sum(rewards) >= 195){
                success_num += 1;
                if (success_num >= 100){
                    Console.WriteLine("Iteration: ", iteration);
                    Console.WriteLine("Clear!!");
                    break;
                }
            }else{
                success_num = 0;
            }

            gaes = PPO.get_gaes(rewards:rewards, v_preds:v_preds, v_preds_next:v_preds_next);

            // convert list to numpy array for (int feeding tf.placeholder
            observations = np.reshape(observations, newshape:[-1] + list(ob_space.shape));
            actions = np.array(actions).astype(dtype:np.int32);
            rewards = np.array(rewards).astype(dtype:np.float32);
            v_preds_next = np.array(v_preds_next).astype(dtype:np.float32);
            gaes = np.array(gaes).astype(dtype:np.float32);
            gaes = (gaes - gaes.mean()) / gaes.std();

            inp = [observations, actions, rewards, v_preds_next, gaes];

            summary = PPO.get_summary(obs:inp[0],
                                      actions:inp[1],
                                      rewards:inp[2],
                                      v_preds_next:inp[3],
                                      gaes:inp[4])[0];

            writer.add_summary(summary, iteration);
        }
        writer.close();
    }
}
Main(){
    Main();
}