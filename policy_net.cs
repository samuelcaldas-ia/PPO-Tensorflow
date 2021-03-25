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


class Policy_net
{
    static object Policy_net(object name, object env, object temp = 0.1)
    {
        /*/
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        /*/

        ob_space = env.ObservationSpace;
        act_space = env.ActionSpace;

        using (tf.variable_scope(name))
        {
            this.obs = tf.placeholder(dtype: tf.float32, shape:[null] + list(ob_space.shape), name: "obs");

            using (tf.variable_scope("policy_net"))
            {
                layer_1 = tf.layers.dense(inputs: this.obs, units: 20, activation: tf.tanh);
                layer_2 = tf.layers.dense(inputs: layer_1, units: 20, activation: tf.tanh);
                layer_3 = tf.layers.dense(inputs: layer_2, units: act_space.n, activation: tf.tanh);
                this.act_probs = tf.layers.dense(inputs: tf.divide(layer_3, temp), units: act_space.n, activation: tf.nn.softmax);
            }
            using (tf.variable_scope("value_net"))
            {
                layer_1 = tf.layers.dense(inputs: this.obs, units: 20, activation: tf.tanh);
                layer_2 = tf.layers.dense(inputs: layer_1, units: 20, activation: tf.tanh);
                this.v_preds = tf.layers.dense(inputs: layer_2, units: 1, activation: null);
            }
            this.act_stochastic = tf.multinomial(tf.log(this.act_probs), num_samples = 1);
            this.act_stochastic = tf.reshape(this.act_stochastic, shape:[-1]);

            this.act_deterministic = tf.argmax(this.act_probs, axis: 1);

            this.scope = tf.get_variable_scope().name;
        }
    }
    static object act(object obs, object stochastic = true)
    {
        if (stochastic)
            return tf.get_default_session().run([this.act_stochastic, this.v_preds], feed_dict: { this.obs: obs});
        else
            return tf.get_default_session().run([this.act_deterministic, this.v_preds], feed_dict: { this.obs: obs});

    }
    static object get_action_prob(object obs)
    {
        return tf.get_default_session().run(this.act_probs, feed_dict:{ this.obs: obs});

    }
    static object get_variables()
    {
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, this.scope);

    }
    static object get_trainable_variables()
    {
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, this.scope);
    }
}
