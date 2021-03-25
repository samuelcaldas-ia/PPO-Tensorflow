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
        :param c_1: parameter for (int value difference
        :param c_2: parameter for (int entropy bonus
        /*/

        this.Policy = Policy;
        this.Old_Policy = Old_Policy;
        this.var GAMMA = gamma;

        pi_trainable = this.Policy.get_trainable_variables();
        old_pi_trainable = this.Old_Policy.get_trainable_variables();

        // assign_operations for (int policy parameter values to old policy parameters
        using (tf.variable_scope("assign_op")) {
            this.assign_ops = new List<double>();
            for (int v_old, v in zip(old_pi_trainable, pi_trainable))
                this.assign_ops.Add(tf.assign(v_old, v));
        }
        // inputs for (int train_op
        using (tf.variable_scope("train_inp")) {
            this.actions = tf.placeholder(dtype: tf.int32, shape:[null], name: "actions");
            this.rewards = tf.placeholder(dtype: tf.float32, shape:[null], name: "rewards");
            this.v_preds_next = tf.placeholder(dtype: tf.float32, shape:[null], name: "v_preds_next");
            this.gaes = tf.placeholder(dtype: tf.float32, shape:[null], name: "gaes");
        }
        act_probs = this.Policy.act_probs;
        act_probs_old = this.Old_Policy.act_probs;

        // probabilities of actions which agent took using (policy
        act_probs = act_probs * tf.one_hot(indices: this.actions, depth: act_probs.shape[1]);
        act_probs = tf.reduce_sum(act_probs, axis: 1);

        // probabilities of actions which agent took using old policy
        act_probs_old = act_probs_old * tf.one_hot(indices: this.actions, depth: act_probs_old.shape[1]);
        act_probs_old = tf.reduce_sum(act_probs_old, axis: 1);

        using (tf.variable_scope("loss/clip")) {
            // ratios = tf.divide(act_probs, act_probs_old);
            ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old));
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min: 1 - clip_value, clip_value_max: 1 + clip_value);
            loss_clip = tf.minimum(tf.multiply(this.gaes, ratios), tf.multiply(this.gaes, clipped_ratios));
            loss_clip = tf.reduce_mean(loss_clip);
            tf.summary.scalar("loss_clip", loss_clip);
        }
        // construct computation graph for (int loss of value function
        using (tf.variable_scope("loss/vf")) {
            v_preds = this.Policy.v_preds;
            loss_vf = tf.squared_difference(this.rewards + this.gamma * this.v_preds_next, v_preds);
            loss_vf = tf.reduce_mean(loss_vf);
            tf.summary.scalar("loss_vf", loss_vf);
        }
        // construct computation graph loss of entropy bonus
        using (tf.variable_scope("loss/entropy")) {
            entropy = -tf.reduce_sum(this.Policy.act_probs * tf.log(tf.clip_by_value(this.Policy.act_probs, 1e-10, 1.0)), axis: 1);
            entropy = tf.reduce_mean(entropy, axis: 0);  // mean of entropy of pi(obs)
            tf.summary.scalar("entropy", entropy);
        }
        using (tf.variable_scope("loss")) {
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy;
            loss = -loss;  // minimize -loss == maximize loss
            tf.summary.scalar("loss", loss);
        }
        this.merged = tf.summary.merge_all();
        optimizer = tf.train.AdamOptimizer(learning_rate: 1e-4, epsilon: 1e-5);
        this.train_op = optimizer.minimize(loss, var_list: pi_trainable);

    } static object train(object obs, object actions, object rewards, object v_preds_next, object gaes){
        tf.get_default_session().run([this.train_op], feed_dict:{this.Policy.obs: obs,
                                                                 this.Old_Policy.obs: obs,
                                                                 this.actions: actions,
                                                                 this.rewards: rewards,
                                                                 this.v_preds_next: v_preds_next,
                                                                 this.gaes: gaes});

    } static object get_summary(object obs, object actions, object rewards, object v_preds_next, object gaes){
        return tf.get_default_session().run([this.merged], feed_dict:{this.Policy.obs: obs,
                                                                      this.Old_Policy.obs: obs,
                                                                      this.actions: actions,
                                                                      this.rewards: rewards,
                                                                      this.v_preds_next: v_preds_next,
                                                                      this.gaes: gaes});

    } static object assign_policy_parameters(){
        // assign policy parameter values to old policy parameters
        return tf.get_default_session().run(this.assign_ops);

    } static object get_gaes(object rewards, object v_preds, object v_preds_next) {
        deltas = [r_t + this.gamma * v_next - v for (int r_t, v_next, v in zip(rewards, v_preds_next, v_preds)];
        // calculate generative advantage estimator(lambda:1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas);
        for (int t in reversed(range(len(gaes) - 1)))  // is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + this.gamma * gaes[t + 1];
        return gaes;
    }

