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

namespace PPO_TF
{
    class Policy_net
    {
        private static Tensor act_probs;
        private static Tensor v_preds;
        private static Tensor act_stochastic;
        private static Tensor act_deterministic;
        private static string scope;

        public Policy_net(string name, CartPoleEnv env, double temp = 0.1)
        {
            /*/
            :param name: string
            :param env: gym env
            :param temp: temperature of boltzmann distribution
            /*/

            var ob_space = env.ObservationSpace;
            var act_space = env.ActionSpace;

            using (tf.variable_scope(name))
            {
                var obs = tf.placeholder(dtype: tf.float32, shape: (null) + list(ob_space.shape), name: "obs");

                using (tf.variable_scope("policy_net"))
                {
                    var layer_1 = tf.layers.dense(inputs: obs, units: 20, activation: tf.tanh);
                    var layer_2 = tf.layers.dense(inputs: layer_1, units: 20, activation: tf.tanh);
                    var layer_3 = tf.layers.dense(inputs: layer_2, units: act_space.n, activation: tf.tanh);
                    act_probs = tf.layers.dense(inputs: tf.divide(layer_3, temp), units: act_space.n, activation: tf.nn.softmax);
                }
                using (tf.variable_scope("value_net"))
                {
                    var layer_1 = tf.layers.dense(inputs: obs, units: 20, activation: tf.tanh);
                    var layer_2 = tf.layers.dense(inputs: layer_1, units: 20, activation: tf.tanh);
                    v_preds = tf.layers.dense(inputs: layer_2, units: 1, activation: null);
                }
                act_stochastic = tf.multinomial(tf.log(act_probs), num_samples: 1);
                act_stochastic = tf.reshape(act_stochastic, shape:[-1]);

                act_deterministic = tf.argmax(act_probs, axis: 1);

                scope = tf.get_variable_scope().name;
            }
        }
        static object act(object obs, bool stochastic = true)
        {
            if (stochastic)
                return tf.get_default_session().run((act_stochastic, v_preds), feed_dict: new[] { (obs: obs) });
            else
                return tf.get_default_session().run((act_deterministic, v_preds), feed_dict: new[] { (obs: obs) });
        }
        static object get_action_prob(object obs)
        {
            return tf.get_default_session().run(act_probs, feed_dict: new[] { (obs: obs) });
        }
        List<object> get_variables()
        {
            return tf.get_collection<object>(tf.GraphKeys.GLOBAL_VARIABLES, scope);
        }
        public List<object> get_trainable_variables()
        {
            return tf.get_collection<object>(tf.GraphKeys.TRAINABLE_VARIABLES, scope);
        }
    }
}
