## ULF模型格式说明

ULF模型格式分为模型网络配置和模型参数两部分,存储格式采用Protobuf，详细定义在[ulf.proto](ulf.md)中描述(ulf.NetParameter定义模型网络配置, ulf.NetWeightsParameter定义模型参数)。从XDL产出的模型，需要在arg_names中配置每个Layer使用的参数名称。

## ULF支持Layer说明

 * <a href="#add_layer">add_layer</a>
 * <a href="#batchdot_layer">batchdot_layer</a>
 * <a href="#bn_layer">bn_layer</a>
 * <a href="#broadcast_to_layer">broadcast_to_layer</a>
 * <a href="#concat_layer">concat_layer</a>
 * <a href="#constant_layer">constant_layer</a>
 * <a href="#dice_layer">dice_layer</a>
 * <a href="#div_layer">div_layer</a>
 * <a href="#embedding_layer">embedding_layer</a>
 * <a href="#fuse_layer">fuse_layer</a>
 * <a href="#gru_layer">gru_layer</a>
 * <a href="#inner_product_layer">inner_product_layer</a>
 * <a href="#inner_product_layer_ex">inner_product_layer_ex</a> 
 * <a href="#multiply_layer">multiply_layer</a>
 * <a href="#prelu_layer">prelu_layer</a>
 * <a href="#relu_layer">relu_layer</a>
 * <a href="#reshape_layer">reshape_layer</a>
 * <a href="#sigmod_layer">sigmoid_layer</a>
 * <a href="#slice_layer">slice_layer</a> 
 * <a href="#softmax_layer">softmax_layer</a> 
 * <a href="#sub_layer">sub_layer</a>
 * <a href="#sum_layer">sum_layer</a>
 * <a href="#tanh_layer">tanh_layer</a>
 * <a href="#where_layer">where_layer</a>


### <a name="add_layer">***add_layer***</a>

支持Elementwise的加法运算层

#### 输入

  <dl>
  <dt><tt>X1</tt> : T</dt>
  <dd>输入张量</dd>
  <dt><tt>X2</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量X1+X2</dd>
  </dl>

#### 例子 

<details>
<summary>add_layer</summary>

```
layer_params {
  name: "add0"
  type: "add_layer"
  bottom: "input0"
  bottom: "input1"
  top: "output0"
}
```
</details>

### <a name="batchdot_layer">***batchdot_layer***</a>

Batchdot运算层

#### 输入

  <dl>
  <dt><tt>X1</tt> : T</dt>
  <dd>输入张量</dd>
  <dt><tt>X2</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 属性

 <dl>
 <dt><tt>transpose_a</tt> : bool </dt>
 <dd>X1是否转置, 默认为False</dd>
 <dt><tt>transpose_b</tt> : bool </dt>
 <dd>X2是否转置, 默认为True</dd>
 </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>batchdot_layer</summary>

```
layer_params {
  name: "batchdot0"
  type: "batchdot_layer"
  bottom: "input0"
  bottom: "input1"
  top: "output0"
  batch_dot_param {
    transpose_a: False
    transpose_b: True
  }
}
```
</details>

### <a name="bn_layer">***bn_layer***</a>

BN运算层, arg_name依次为: beta, gamma, mean, var.

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 属性

 <dl>
 <dt><tt>eps</tt> : bool </dt>
 <dd>避免除0的eps参数, 默认为0.01</dd>
 </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>bn_layer</summary>

```
layer_params {
  name: "bn0"
  type: "bn_layer"
  bottom: "input0"
  top: "output0"
  bn_param {
    eps: 0.01
  }
}
```
</details>

### <a name="broadcast_to_layer">***broadcast_to_layer***</a>

BroadcastTo运算层,如果输入参数等于2, 则将第一个输入张量broadcast为第二个张量的shape；否则，将第一个输入张量broadcast为属性中配置的shape.

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  <dt><tt>X_dst</tt> : T</dt>
  <dd>输入张量,X将Broadcast到该张量的shape, 该输入为Optional</dd>
  </dl>

#### 属性

 <dl>
 <dt><tt>shape</tt> : bool </dt>
 <dd>Broadcast的目标shape</dd>
 </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>broadcast_to_layer</summary>

```
layer_params {
  name: "broadcast0"
  type: "broadcast_to_layer"
  bottom: "input0"
  top: "output0"
  broadcast_to_param {
    shape: 0
    shape: 12
    shape: -1
  }
}
```
</details>

#### <a name="concat_layer">***concat_layer***</a>

Concat层

#### 输入

 <dl>
 <dt><tt>X1</tt> : T </dt>
 <dd>输入张量</dd>
 <dt><tt>X2</tt> : T </dt>
 <dd>输入张量</dd>
 </dl>

#### 属性

 <dl>
 <dt><tt>dim</tt> : int32 </dt>
 <dd>concat维度</dd>
 </dl>

#### 输出

 <dl>
 <dt><tt>Y</tt> : T </dt>
 <dd>输出张量</dd>
 </dl>

<details>
<summary>concat_layer</summary>

```
layer_params {
  name: "concat0"
  type: "concat_layer"
  bottom: "input0"
  top: "output0"
  concat_param {
    dim: 2
  }
}
```
</details>

#### <a name="constant_layer">***constant_layer***</a>

Constant层

#### 属性

 <dl>
 <dt><tt>blob_data.shape</tt> : int32 </dt>
 <dd>shape</dd>
 <dt><tt>blob_data.data</tt> : int32 </dt>
 <dd>数据</dd>
 </dl>

#### 输出

 <dl>
 <dt><tt>Y</tt> : T </dt>
 <dd>输出张量</dd>
 </dl>

<details>
<summary>constant_layer</summary>

```
layer_params {
  name: "constant0"
  type: "constant_layer"
  bottom: "input0"
  top: "output0"
  constant_param {
    blob_data {
      shape: 1
      data: 1.0
    }
  }
}
```
</details>

### <a name="dice_layer">***dice_layer***</a>

Dice激活运算层, arg_names依次为: gamma, mean, var

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 属性

 <dl>
 <dt><tt>eps</tt> : bool </dt>
 <dd>避免除0的eps参数, 默认为0.01</dd>
 </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>dice_layer</summary>

```
layer_params {
  name: "dice0"
  type: "dice_layer"
  bottom: "input0"
  top: "output0"
  dice_param {
    eps: 1e-8
  }
}
```
</details>

### <a name="div_layer">***div_layer***</a>

支持Elementwise的除法运算层

#### 输入

  <dl>
  <dt><tt>X1</tt> : T</dt>
  <dd>输入张量</dd>
  <dt><tt>X2</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量X1/X2</dd>
  </dl>

#### 例子 

<details>
<summary>div_layer</summary>

```
layer_params {
  name: "div0"
  type: "div_layer"
  bottom: "input0"
  top: "output0"
}
```
</details>

### <a name="embedding_layer">***embedding_layer***</a>

Embedding层, embedding配置定义在[embedding.proto](../proto/embedding.proto)中。

#### 输入

  <dl>
  <dt><tt>ID0</tt> : int64</dt>
  <dd>输入张量ID</dd>
  <dt><tt>VALUE0</tt> : T</dt>
  <dd>输入张量VALUE</dd>
  <dt><tt>SEGMENTS0</tt> : int32</dt>
  <dd>输入张量SEGMENTS</dd>
  <dt><tt>...</tt> : ...</dt>
  <dd>...</dd>
  </dl>

#### 属性

 <dl>
 <dt><tt>embedding_conf_path</tt> : string </dt>
 <dd>embedding配置路径</dd>
 <dt><tt>level</tt> : int32</dt>
 <dd>特征层级,一般而言, user特征1, ad特征0</dd>
 </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量Y</dd>
  </dl>

#### 例子 

<details>
<summary>embedding_layer</summary>

```
layer_params {
  name: "embedding0"
  type: "embedding_layer"
  bottom: "input0"
  top: "output0"
  embedding_param {
    embedding_conf_path: "./user_embedding.conf"
    level: 1                   
  }
}
```
</details>

### <a name="fuse_layer">***fuse_layer***</a>

Fuse层,支持elementwise的concat

#### 输入

  <dl>
  <dt><tt>X1</tt> : T</dt>
  <dd>输入张量X1</dd>
  <dt><tt>X2</tt> : T</dt>
  <dd>输入张量X2</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量Y</dd>
  </dl>

#### 例子 

<details>
<summary>fuse_layer</summary>

```
layer_params {
  name: "fuse0"
  type: "fuse_layer"
  bottom: "ad_fea"
  bottom: "user_fea"
  top: "output0"
  fuse_param {
    common_input_index: 1
  }
}
```
</details>

### <a name="gru_layer">***gru_layer***</a>

GRU Layer, arg_names依次为: h2hweight, i2hweight, h2hBias, i2hbias

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

### <a name="inner_product_layer">***inner_product_layer***</a>

FC全联接层, arg_names依次为: weight, bias

#### 输入

 <dl>
 <dt><tt>X</tt> : T </dt>
 <dd>输入张量</dd>
 </dl>

#### 属性

 <dl>
 <dt><tt>transpose</tt> : bool </dt>
 <dd>参数是否转置,默认为False</dd>
 <dt><tt>bias_term</tt> : bool </dt>
 <dd>是否有bias,默认为True</dd>
 </dl>

#### 输出

 <dl>
 <dt><tt>Y</tt> : T </dt>
 <dd>输出张量</dd>
 </dl>

### <a name="inner_product_layer_ex">***inner_product_layer_ex***</a>

GEMM裁剪型FC全联接, 譬如: FC的输入为User特征和Ad特征的叠加. arg_names依次为: weight1, weight2, bias

#### 输入

 <dl>
 <dt><tt>X1</tt> : T </dt>
 <dd>输入张量X[,0:dim]</dd>
 <dt><tt>X2</tt> : T </dt>
 <dd>输入张量X[,dim:k]</dd>
 </dl>

#### 属性

 <dl>
 <dt><tt>transpose</tt> : bool </dt>
 <dd>参数是否转置,默认为False</dd>
 <dt><tt>bias_term</tt> : bool </dt>
 <dd>是否有bias,默认为True</dd>
 </dl>

#### 输出

 <dl>
 <dt><tt>Y</tt> : T </dt>
 <dd>输出张量</dd>
 </dl>

### <a name="multiply_layer">***multiply_layer***</a>

支持Elementwise的乘法运算层

#### 输入

  <dl>
  <dt><tt>X1</tt> : T</dt>
  <dd>输入张量</dd>
  <dt><tt>X2</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量X1*X2</dd>
  </dl>

#### 例子 

<details>
<summary>multiply_layer</summary>

```
layer_params {
  name: "mul0"
  type: "multiply_layer"
  bottom: "input0"
  bottom: "input1"
  top: "output0"
}
```
</details>

### <a name="prelu_layer">***prelu_layer***</a>

Prelu层, arg_names为prelu参数

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>prelu_layer</summary>

```
layer_params {
  name: "prelu0"
  type: "prelu_layer"
  bottom: "input0"
  top: "output0"
}
```
</details>

### <a name="relu_layer">***relu_layer***</a>

Relu层

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>relu_layer</summary>

```
layer_params {
  name: "relu0"
  type: "relu_layer"
  bottom: "input0"
  top: "output0"
}
```
</details>

### <a name="reshape_layer">***reshape_layer***</a>

Reshape层

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 属性

 <dl>
 <dt><tt>shape</tt> : int32 </dt>
 <dd>指定的shape, 0表示沿用上一个shape, -1表示推断得出</dd>
 </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>reshape_layer</summary>

```
layer_params {
  name: "reshape0"
  type: "rehshape_layer"
  bottom: "input0"
  top: "output0"
  reshape_param {
    shape : 1
    shape : -1
    shape : 36
  }
}
```
</details>

### <a name="sigmoid_layer">***sigmoid_layer***</a>

Sigmoid层

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>sigmoid_layer</summary>

```
layer_params {
  name: "sigmod0"
  type: "sigmoid_layer"
  bottom: "input0"
  top: "output0"
}
```
</details>

### <a name="slice_layer">***slice_layer***</a>

多Slice/Concat组合层, 将原始输入Reshape为二维, 将多个dim=1上的Slice结果合并.

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 属性

  <dl>
  <dt><tt>concat_dim</tt> : int32 </dt>
  <dd>Concat的dim, 默认值为0</dd>
  <dt><tt>slices.offset</tt> : int32 </dt>
  <dd>Slice的offset</dd>
  <dt><tt>slices.shape</tt> : int32 </dt>
  <dd>Slice的shape</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>slice_layer</summary>

```
layer_params {
  name: "slice0"
  type: "slice_layer"
  bottom: "input0"
  top: "output0"
  slice_param {
    concat_dim: 2
    slices {
      offset: 0
      shape: 150
      shape: 18
    }
    slices {
      offset: 2700
      shape: 150
      shape: 18
    }
  }
}
```
</details>

### <a name="softmax_layer">***softmax_layer***</a>

Softmax层

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

### <a name="sub_layer">***add_layer***</a>

支持Elementwise的减法运算层

#### 输入

  <dl>
  <dt><tt>X1</tt> : T</dt>
  <dd>输入张量</dd>
  <dt><tt>X2</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量X1-X2</dd>
  </dl>

#### 例子 

<details>
<summary>sub_layer</summary>

```
layer_params {
  name: "sub0"
  type: "sub_layer"
  bottom: "input0"
  bottom: "input1"
  top: "output0"
}
```
</details>

### <a name="sum_layer">***sum_layer***</a>

ReduceSum计算层

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量X</dd>
  </dl>

#### 属性

 <dl>
 <dt><tt>dim</tt> : int32 </dt>
 <dd>ReduceSum的dim</dd>
 </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量Y</dd>
  </dl>

#### 例子 

<details>
<summary>sum_layer</summary>

```
layer_params {
  name: "reduce0"
  type: "reduce_layer"
  bottom: "ad_fea"
  top: "output0"
  sum_param {
    dim: 1
  }
}
```
</details>

### <a name="tanh_layer">***tanh_layer***</a>

Tanh层

#### 输入

  <dl>
  <dt><tt>X</tt> : T</dt>
  <dd>输入张量</dd>
  </dl>

#### 输出
  
  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量</dd>
  </dl>

#### 例子 

<details>
<summary>tanh_layer</summary>

```
layer_params {
  name: "tanh0"
  type: "tanh_layer"
  bottom: "input0"
  top: "output0"
}
```
</details>

### <a name="where_layer">***where_layer***</a>

Where层

#### 输入

  <dl>
  <dt><tt>Condition</tt> : int32 | int64
  <dd>输入张量Condition</dd>
  <dt><tt>Y</tt> : T </dt>
  <dd>输入张量X1</dd>
  <dt><tt>Y</tt> : T </dt>
  <dd>输入张量X2</dd>
  </dl>

#### 输出

  <dl>
  <dt><tt>Y</tt> : T </dt>
  <dd>输出张量Conditon > 0 ? X1 : X2 </dd>
  </dl>

#### 例子

<details>
<summary>where_layer</summary>

```
layer_params {
  name: "where0"
  type: "where_layer"
  bottom: "input0"
  bottom: "input1"
  bottom: "input2"
  top: "output0"
}
```
</details>





