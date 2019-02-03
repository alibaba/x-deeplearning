/*
 * \file activation_test.cc
 * \brief The activation test unit
 */
#include "gtest/gtest.h"

#include "blaze/math/activation.h"
#include "blaze/common/log.h"

namespace blaze {

TEST(TestActivation, Relu) {
  Activation<kRelu> relu_act;
  float input = 2.0;
  float output = 1.0;
  relu_act(1.0, &input, &output);
  EXPECT_FLOAT_EQ(output, 2.0);
}

TEST(TestActivation, InplaceRelu) {
  Activation<kRelu> relu_act;
  float input = -2.0;
  relu_act(1.0, &input, &input);
  EXPECT_FLOAT_EQ(input, -2.0);
}

TEST(TestActivation, PRelu) {
  Activation<kPRelu> prelu_act;
  float input = 2.0;
  double w = 2.0;
  float output = 1.0;
  prelu_act(&input, &w, &output);
  EXPECT_FLOAT_EQ(output, 2.0);

  input = -0.5;
  prelu_act(&input, &w, &output);
  EXPECT_FLOAT_EQ(output, -1.0);
}

TEST(TestActivation, InplacePRelu) {
  Activation<kPRelu> prelu_act;
  float input = 2.0;
  double w = 2.0;
  prelu_act(&input, &w, &input);
  EXPECT_FLOAT_EQ(input, 2.0);

  input = -0.5;
  prelu_act(&input, &w, &input);
  EXPECT_FLOAT_EQ(input, -1.0);
}

TEST(TestActivation, Dice) {
  Activation<kDice> dice_act;
  float input = 2.0;
  double gamma = 1.2;
  double mean = 1.3;
  double avr = 1.1;
  float output;
  dice_act(&input, &gamma, &mean, &avr, &output);
  LOG_INFO("dice output=%f", output);
}

TEST(TestActivation, Tanh) {
  Activation<kTanh> tanh_act;
  float input = 2.0;
  float output;
  tanh_act(&input, &output);
  LOG_INFO("tanh output=%f", output);
}

TEST(TestActivation, Sigmoid) {
  Activation<kSigmoid> sigmoid_act;
  float input = 2.1;
  float output;
  sigmoid_act(&input, &output);
  LOG_INFO("sigmoid output=%f", output);
}

TEST(TestActivation, BN) {
  Activation<kBN> bn_act;
  float input = 3.2;
  double gamma = 3.1;
  double beta = 2.1;
  double mean = 1.6;
  double var = 1.1;
  float output;
  bn_act(&input, &gamma, &beta, &mean, &var, &output);
  LOG_INFO("bn output=%f", output);
}

}  // namespace blaze 

