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

// import tensorflow as tf
// import copy


class PPOTrain {
    static object PPOTrain(object Policy, object Old_Policy, object gamma = 0.95, object clip_value = 0.2, object c_1 = 1, object c_2 = 0.01){
        /*/
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        /*/

        var pi_trainable = Policy.get_trainable_variables();
        var old_pi_trainable = Old_Policy.get_trainable_variables();

        // assign_operations for policy parameter values to old policy parameters
        using (tf.variable_scope("assign_op")) {
            var assign_ops = new List<double>();
            foreach (var (v_old, v) in zip(old_pi_trainable, pi_trainable))
                assign_ops.Add(tf.assign(v_old, v));
        }
        // inputs for (int train_op
        using (tf.variable_scope("train_inp")) {
            var actions = tf.placeholder(dtype: tf.int32, shape:(null), name: "actions");
            var rewards = tf.placeholder(dtype: tf.float32, shape:(null), name: "rewards");
            var v_preds_next = tf.placeholder(dtype: tf.float32, shape:(null), name: "v_preds_next");
            var gaes = tf.placeholder(dtype: tf.float32, shape:(null), name: "gaes");
        }
        var act_probs = Policy.act_probs;
        var act_probs_old = Old_Policy.act_probs;

        // probabilities of actions which agent took using (policy
        act_probs = act_probs * tf.one_hot(indices: actions, depth: act_probs.shape[1]);
        act_probs = tf.reduce_sum(act_probs, axis: 1);

        // probabilities of actions which agent took using old policy
        act_probs_old = act_probs_old * tf.one_hot(indices: actions, depth: act_probs_old.shape[1]);
        act_probs_old = tf.reduce_sum(act_probs_old, axis: 1);

        using (tf.variable_scope("loss/clip")) {
            // ratios = tf.divide(act_probs, act_probs_old);
            var ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old));
            var clipped_ratios = tf.clip_by_value(ratios, clip_value_min: 1 - clip_value, clip_value_max: 1 + clip_value);
            var loss_clip = tf.minimum(tf.multiply(gaes, ratios), tf.multiply(gaes, clipped_ratios));
            loss_clip = tf.reduce_mean(loss_clip);
            tf.summary.scalar("loss_clip", loss_clip);
        }
        // construct computation graph for (int loss of value function
        using (tf.variable_scope("loss/vf")) {
            var v_preds = Policy.v_preds;
            var loss_vf = tf.squared_difference(rewards + gamma * v_preds_next, v_preds);
            loss_vf = tf.reduce_mean(loss_vf);
            tf.summary.scalar("loss_vf", loss_vf);
        }
        // construct computation graph loss of entropy bonus
        using (tf.variable_scope("loss/entropy")) {
            var entropy = -tf.reduce_sum(Policy.act_probs * tf.log(tf.clip_by_value(Policy.act_probs, 1e-10, 1.0)), axis: 1);
            entropy = tf.reduce_mean(entropy, axis: 0);  // mean of entropy of pi(obs)
            tf.summary.scalar("entropy", entropy);
        }
        using (tf.variable_scope("loss")) {
            var loss = loss_clip - c_1 * loss_vf + c_2 * entropy;
            loss = -loss;  // minimize -loss == maximize loss
            tf.summary.scalar("loss", loss);
        }
        var merged = tf.summary.merge_all();
        var optimizer = tf.train.AdamOptimizer(learning_rate: 1e-4, epsilon: 1e-5);
        var train_op = optimizer.minimize(loss, var_list: pi_trainable);

    } static object train(object obs, object actions, object rewards, object v_preds_next, object gaes){
        tf.get_default_session().run([train_op], feed_dict: new[]{Policy.obs: obs,
                                                                 Old_Policy.obs: obs,
                                                                 actions: actions,
                                                                 rewards: rewards,
                                                                 v_preds_next: v_preds_next,
                                                                 gaes: gaes});

    } static object get_summary(object obs, object actions, object rewards, object v_preds_next, object gaes){
        return tf.get_default_session().run([merged], feed_dict: new[]{Policy.obs: obs,
                                                                      Old_Policy.obs: obs,
                                                                      actions: actions,
                                                                      rewards: rewards,
                                                                      v_preds_next: v_preds_next,
                                                                      gaes: gaes});

    } static object assign_policy_parameters(){
        // assign policy parameter values to old policy parameters
        return tf.get_default_session().run(assign_ops);

    } static object get_gaes(object rewards, object v_preds, object v_preds_next) {
        var deltas = new List<double>();
        foreach (var (r_t, v_next, v) in zip(rewards, v_preds_next, v_preds))
        {
            deltas.Add(r_t + gamma * v_next - v);
        }
        // calculate generative advantage estimator(lambda:1), see ppo paper eq(11)
        var gaes = copy.deepcopy(deltas);
        for (int t=reversed(range(len(gaes) - 1)); t>=0;t--)  // is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + gamma * gaes[t + 1];
        return gaes;
    }

