### <a name="Embedding"></a><a name="Embedding">**Embedding**</a>

Embedding.

#### Attributes

<dl>
<dt><tt>algo</tt> : string</dt>
<dd>The algorithm of embedding, which includes "sum/avg/assign", and the default value is "sum".</dd>
<dt><tt>url</tt> : string</dt>
<dd>The embedding data source url.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>feature tensor</tt> : int64</dt>
<dd>The feature tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor of shape (M, N).</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>
