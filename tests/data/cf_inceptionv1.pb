
?
inception_5b/5x5_reduceConvinception_5a/output"
kernel_shape	
?0"
strides
"
use_bias("/
_output_shapes
:?????????0"
pads

        "
group
o
inception_3b/relu_3x3_reduceReluinception_3b/3x3_reduce"0
_output_shapes
:??????????
n
inception_3b/relu_5x5_reduceReluinception_3b/5x5_reduce"/
_output_shapes
:????????? 
?
inception_5b/pool_projConvinception_5b/pool"0
_output_shapes
:??????????"
pads

        "
group"
kernel_shape

??"
strides
"
use_bias(
l
data	DataInput"1
_output_shapes
:???????????"&
shape:???????????
m
inception_4e/relu_pool_projReluinception_4e/pool_proj"0
_output_shapes
:??????????
?
inception_5a/3x3Convinception_5a/relu_3x3_reduce"0
_output_shapes
:??????????"
pads

    "
group"
kernel_shape

??"
strides
"
use_bias(
?
conv2/3x3_reduceConvpool1/3x3_s2"
kernel_shape
@@"
strides
"
use_bias("/
_output_shapes
:?????????88@"
pads

        "
group
?
inception_4a/5x5_reduceConvpool3/3x3_s2"
use_bias("/
_output_shapes
:?????????"
pads

        "
group"
kernel_shape	
?"
strides

?
inception_4a/pool_projConvinception_4a/pool"/
_output_shapes
:?????????@"
pads

        "
group"
kernel_shape	
?@"
strides
"
use_bias(
?
inception_5b/3x3_reduceConvinception_5a/output"
group"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

        
?
inception_3a/5x5_reduceConvpool2/3x3_s2"/
_output_shapes
:?????????"
pads

        "
group"
kernel_shape	
?"
strides
"
use_bias(
?
inception_4c/5x5Convinception_4c/relu_5x5_reduce"
strides
"
use_bias("/
_output_shapes
:?????????@"
pads

    "
group"
kernel_shape
@
a
inception_5b/relu_1x1Reluinception_5b/1x1"0
_output_shapes
:??????????
?
inception_4c/3x3Convinception_4c/relu_3x3_reduce"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

    "
group
l
inception_3b/relu_pool_projReluinception_3b/pool_proj"/
_output_shapes
:?????????@
`
inception_3a/relu_1x1Reluinception_3a/1x1"/
_output_shapes
:?????????@
?
inception_3a/1x1Convpool2/3x3_s2"/
_output_shapes
:?????????@"
pads

        "
group"
kernel_shape	
?@"
strides
"
use_bias(
?
inception_5a/poolPoolpool4/3x3_s2"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

    
n
inception_4a/relu_5x5_reduceReluinception_4a/5x5_reduce"/
_output_shapes
:?????????
?
inception_4e/3x3Convinception_4e/relu_3x3_reduce"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

    "
group
?
pool5/7x7_s1Poolinception_5b/output"
pooling_typeAVG"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

        
?
inception_4b/3x3Convinception_4b/relu_3x3_reduce"
kernel_shape	
p?"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

    "
group
n
inception_5b/relu_5x5_reduceReluinception_5b/5x5_reduce"/
_output_shapes
:?????????0
?
inception_4b/3x3_reduceConvinception_4a/output"
strides
"
use_bias("/
_output_shapes
:?????????p"
pads

        "
group"
kernel_shape	
?p
?
pool3/3x3_s2Poolinception_3b/output"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

      
?
inception_4a/1x1Convpool3/3x3_s2"
group"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

        
?
inception_5a/5x5_reduceConvpool4/3x3_s2"/
_output_shapes
:????????? "
pads

        "
group"
kernel_shape	
? "
strides
"
use_bias(
a
inception_5a/relu_1x1Reluinception_5a/1x1"0
_output_shapes
:??????????
o
inception_4e/relu_3x3_reduceReluinception_4e/3x3_reduce"0
_output_shapes
:??????????
?
inception_3a/poolPoolpool2/3x3_s2"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

    
?
inception_4c/1x1Convinception_4b/output"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

        "
group"
kernel_shape

??
?
inception_5a/pool_projConvinception_5a/pool"0
_output_shapes
:??????????"
pads

        "
group"
kernel_shape

??"
strides
"
use_bias(
o
inception_4c/relu_3x3_reduceReluinception_4c/3x3_reduce"0
_output_shapes
:??????????
?
inception_5a/3x3_reduceConvpool4/3x3_s2"0
_output_shapes
:??????????"
pads

        "
group"
kernel_shape

??"
strides
"
use_bias(
?
inception_4e/5x5_reduceConvinception_4d/output"/
_output_shapes
:????????? "
pads

        "
group"
kernel_shape	
? "
strides
"
use_bias(
`
inception_4c/relu_5x5Reluinception_4c/5x5"/
_output_shapes
:?????????@
?
inception_3b/3x3_reduceConvinception_3a/output"
use_bias("0
_output_shapes
:??????????"
pads

        "
group"
kernel_shape

??"
strides

a
inception_4a/relu_1x1Reluinception_4a/1x1"0
_output_shapes
:??????????
?
inception_4e/poolPoolinception_4d/output"
strides
"0
_output_shapes
:??????????"
pads

    "
pooling_typeMAX"
kernel_shape

?
inception_4a/poolPoolpool3/3x3_s2"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

    
?
inception_3b/5x5Convinception_3b/relu_5x5_reduce"/
_output_shapes
:?????????`"
pads

    "
group"
kernel_shape
 `"
strides
"
use_bias(
a
inception_5b/relu_3x3Reluinception_5b/3x3"0
_output_shapes
:??????????
`
inception_4d/relu_1x1Reluinception_4d/1x1"/
_output_shapes
:?????????p
n
inception_4e/relu_5x5_reduceReluinception_4e/5x5_reduce"/
_output_shapes
:????????? 
?
inception_4b/5x5Convinception_4b/relu_5x5_reduce"
strides
"
use_bias("/
_output_shapes
:?????????@"
pads

    "
group"
kernel_shape
@
?
inception_4c/3x3_reduceConvinception_4b/output"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

        "
group"
kernel_shape

??
?
inception_5a/5x5Convinception_5a/relu_5x5_reduce"
kernel_shape	
 ?"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

    "
group
?
inception_4e/3x3_reduceConvinception_4d/output"0
_output_shapes
:??????????"
pads

        "
group"
kernel_shape

??"
strides
"
use_bias(
n
inception_3a/relu_3x3_reduceReluinception_3a/3x3_reduce"/
_output_shapes
:?????????`
?
inception_4b/outputConcatinception_4b/relu_1x1inception_4b/relu_3x3inception_4b/relu_5x5inception_4b/relu_pool_proj"

axis"0
_output_shapes
:??????????
\
loss3/loss3Softmaxloss3/classifier_1"0
_output_shapes
:??????????
?
inception_4c/5x5_reduceConvinception_4b/output"/
_output_shapes
:?????????"
pads

        "
group"
kernel_shape	
?"
strides
"
use_bias(
?
inception_4a/3x3_reduceConvpool3/3x3_s2"
kernel_shape	
?`"
strides
"
use_bias("/
_output_shapes
:?????????`"
pads

        "
group
?
inception_4b/5x5_reduceConvinception_4a/output"
kernel_shape	
?"
strides
"
use_bias("/
_output_shapes
:?????????"
pads

        "
group
n
inception_5a/relu_5x5_reduceReluinception_5a/5x5_reduce"/
_output_shapes
:????????? 
?
inception_4d/1x1Convinception_4c/output"
pads

        "
group"
kernel_shape	
?p"
strides
"
use_bias("/
_output_shapes
:?????????p
`
inception_3a/relu_5x5Reluinception_3a/5x5"/
_output_shapes
:????????? 
?
inception_4d/3x3Convinception_4d/relu_3x3_reduce"0
_output_shapes
:??????????"
pads

    "
group"
kernel_shape

??"
strides
"
use_bias(
a
inception_3b/relu_1x1Reluinception_3b/1x1"0
_output_shapes
:??????????
l
inception_4c/relu_pool_projReluinception_4c/pool_proj"/
_output_shapes
:?????????@
U
loss3/classifier_0Flattenpool5/7x7_s1"(
_output_shapes
:??????????
a
inception_4b/relu_3x3Reluinception_4b/3x3"0
_output_shapes
:??????????
?
inception_3a/outputConcatinception_3a/relu_1x1inception_3a/relu_3x3inception_3a/relu_5x5inception_3a/relu_pool_proj"

axis"0
_output_shapes
:??????????
?
pool1/3x3_s2Poolconv1/relu_7x7"
pooling_typeMAX"
kernel_shape
"
strides
"/
_output_shapes
:?????????88@"
pads

      
?
inception_3a/3x3Convinception_3a/relu_3x3_reduce"
kernel_shape	
`?"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

    "
group
?
inception_4d/poolPoolinception_4c/output"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

    
a
inception_4d/relu_3x3Reluinception_4d/3x3"0
_output_shapes
:??????????
?
inception_4d/outputConcatinception_4d/relu_1x1inception_4d/relu_3x3inception_4d/relu_5x5inception_4d/relu_pool_proj"0
_output_shapes
:??????????"

axis
?
inception_5b/5x5Convinception_5b/relu_5x5_reduce"
group"
kernel_shape	
0?"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

    
`
inception_4a/relu_5x5Reluinception_4a/5x5"/
_output_shapes
:?????????0
?
inception_3b/1x1Convinception_3a/output"
pads

        "
group"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????
?
inception_4c/outputConcatinception_4c/relu_1x1inception_4c/relu_3x3inception_4c/relu_5x5inception_4c/relu_pool_proj"0
_output_shapes
:??????????"

axis
a
inception_3b/relu_3x3Reluinception_3b/3x3"0
_output_shapes
:??????????
?
pool2/3x3_s2Poolconv2/relu_3x3"
strides
"0
_output_shapes
:??????????"
pads

      "
pooling_typeMAX"
kernel_shape

?
inception_4e/1x1Convinception_4d/output"
group"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

        
?
inception_3a/5x5Convinception_3a/relu_5x5_reduce"/
_output_shapes
:????????? "
pads

    "
group"
kernel_shape
 "
strides
"
use_bias(
n
inception_4b/relu_3x3_reduceReluinception_4b/3x3_reduce"/
_output_shapes
:?????????p
m
inception_5a/relu_pool_projReluinception_5a/pool_proj"0
_output_shapes
:??????????
?
inception_4b/1x1Convinception_4a/output"
pads

        "
group"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????
l
inception_4b/relu_pool_projReluinception_4b/pool_proj"/
_output_shapes
:?????????@
a
inception_5b/relu_5x5Reluinception_5b/5x5"0
_output_shapes
:??????????
a
inception_4e/relu_3x3Reluinception_4e/3x3"0
_output_shapes
:??????????
o
inception_5b/relu_3x3_reduceReluinception_5b/3x3_reduce"0
_output_shapes
:??????????
?
inception_3b/outputConcatinception_3b/relu_1x1inception_3b/relu_3x3inception_3b/relu_5x5inception_3b/relu_pool_proj"

axis"0
_output_shapes
:??????????
?
inception_5b/1x1Convinception_5a/output"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

        "
group
a
inception_4e/relu_5x5Reluinception_4e/5x5"0
_output_shapes
:??????????
?
inception_5b/poolPoolinception_5a/output"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

    
a
inception_4e/relu_1x1Reluinception_4e/1x1"0
_output_shapes
:??????????
a
inception_4c/relu_1x1Reluinception_4c/1x1"0
_output_shapes
:??????????
?
inception_4c/poolPoolinception_4b/output"
pads

    "
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????
`
conv2/relu_3x3_reduceReluconv2/3x3_reduce"/
_output_shapes
:?????????88@
?
inception_5a/outputConcatinception_5a/relu_1x1inception_5a/relu_3x3inception_5a/relu_5x5inception_5a/relu_pool_proj"

axis"0
_output_shapes
:??????????
?
	conv2/3x3Convconv2/relu_3x3_reduce"0
_output_shapes
:?????????88?"
pads

    "
group"
kernel_shape	
@?"
strides
"
use_bias(
?
inception_3a/pool_projConvinception_3a/pool"/
_output_shapes
:????????? "
pads

        "
group"
kernel_shape	
? "
strides
"
use_bias(
`
inception_4d/relu_5x5Reluinception_4d/5x5"/
_output_shapes
:?????????@
l
inception_4a/relu_pool_projReluinception_4a/pool_proj"/
_output_shapes
:?????????@
?
inception_4a/3x3Convinception_4a/relu_3x3_reduce"0
_output_shapes
:??????????"
pads

    "
group"
kernel_shape	
`?"
strides
"
use_bias(
n
inception_4d/relu_5x5_reduceReluinception_4d/5x5_reduce"/
_output_shapes
:????????? 
a
inception_4b/relu_1x1Reluinception_4b/1x1"0
_output_shapes
:??????????
o
inception_5a/relu_3x3_reduceReluinception_5a/3x3_reduce"0
_output_shapes
:??????????
?
inception_4e/pool_projConvinception_4e/pool"
use_bias("0
_output_shapes
:??????????"
pads

        "
group"
kernel_shape

??"
strides

a
inception_4a/relu_3x3Reluinception_4a/3x3"0
_output_shapes
:??????????
?
inception_4b/pool_projConvinception_4b/pool"/
_output_shapes
:?????????@"
pads

        "
group"
kernel_shape	
?@"
strides
"
use_bias(
?
inception_3b/3x3Convinception_3b/relu_3x3_reduce"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

    "
group
n
inception_4a/relu_3x3_reduceReluinception_4a/3x3_reduce"/
_output_shapes
:?????????`
m
inception_5b/relu_pool_projReluinception_5b/pool_proj"0
_output_shapes
:??????????
?
loss3/classifier_1FullyConnectedloss3/classifier_0"(
_output_shapes
:??????????"
use_bias("
units?
?
inception_4e/5x5Convinception_4e/relu_5x5_reduce"0
_output_shapes
:??????????"
pads

    "
group"
kernel_shape	
 ?"
strides
"
use_bias(
a
inception_4c/relu_3x3Reluinception_4c/3x3"0
_output_shapes
:??????????
?
conv1/7x7_s2Convdata"/
_output_shapes
:?????????pp@"
pads

    "
group"
kernel_shape
@"
strides
"
use_bias(
n
inception_4b/relu_5x5_reduceReluinception_4b/5x5_reduce"/
_output_shapes
:?????????
?
inception_5a/1x1Convpool4/3x3_s2"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

        "
group
?
pool4/3x3_s2Poolinception_4e/output"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

      
n
inception_3a/relu_5x5_reduceReluinception_3a/5x5_reduce"/
_output_shapes
:?????????
?
inception_3a/3x3_reduceConvpool2/3x3_s2"/
_output_shapes
:?????????`"
pads

        "
group"
kernel_shape	
?`"
strides
"
use_bias(
l
inception_3a/relu_pool_projReluinception_3a/pool_proj"/
_output_shapes
:????????? 
S
conv2/relu_3x3Relu	conv2/3x3"0
_output_shapes
:?????????88?
a
inception_3a/relu_3x3Reluinception_3a/3x3"0
_output_shapes
:??????????
?
inception_3b/pool_projConvinception_3b/pool"
use_bias("/
_output_shapes
:?????????@"
pads

        "
group"
kernel_shape	
?@"
strides

?
inception_4a/5x5Convinception_4a/relu_5x5_reduce"
group"
kernel_shape
0"
strides
"
use_bias("/
_output_shapes
:?????????0"
pads

    
o
inception_4d/relu_3x3_reduceReluinception_4d/3x3_reduce"0
_output_shapes
:??????????
n
inception_4c/relu_5x5_reduceReluinception_4c/5x5_reduce"/
_output_shapes
:?????????
`
inception_4b/relu_5x5Reluinception_4b/5x5"/
_output_shapes
:?????????@
?
inception_3b/poolPoolinception_3a/output"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

    
?
inception_4e/outputConcatinception_4e/relu_1x1inception_4e/relu_3x3inception_4e/relu_5x5inception_4e/relu_pool_proj"

axis"0
_output_shapes
:??????????
?
inception_5b/outputConcatinception_5b/relu_1x1inception_5b/relu_3x3inception_5b/relu_5x5inception_5b/relu_pool_proj"0
_output_shapes
:??????????"

axis
?
inception_3b/5x5_reduceConvinception_3a/output"
kernel_shape	
? "
strides
"
use_bias("/
_output_shapes
:????????? "
pads

        "
group
a
inception_5a/relu_5x5Reluinception_5a/5x5"0
_output_shapes
:??????????
`
inception_3b/relu_5x5Reluinception_3b/5x5"/
_output_shapes
:?????????`
?
inception_4b/poolPoolinception_4a/output"
pooling_typeMAX"
kernel_shape
"
strides
"0
_output_shapes
:??????????"
pads

    
U
conv1/relu_7x7Reluconv1/7x7_s2"/
_output_shapes
:?????????pp@
?
inception_4c/pool_projConvinception_4c/pool"
kernel_shape	
?@"
strides
"
use_bias("/
_output_shapes
:?????????@"
pads

        "
group
?
inception_4d/3x3_reduceConvinception_4c/output"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:??????????"
pads

        "
group
l
inception_4d/relu_pool_projReluinception_4d/pool_proj"/
_output_shapes
:?????????@
?
inception_4a/outputConcatinception_4a/relu_1x1inception_4a/relu_3x3inception_4a/relu_5x5inception_4a/relu_pool_proj"0
_output_shapes
:??????????"

axis
a
inception_5a/relu_3x3Reluinception_5a/3x3"0
_output_shapes
:??????????
?
inception_5b/3x3Convinception_5b/relu_3x3_reduce"0
_output_shapes
:??????????"
pads

    "
group"
kernel_shape

??"
strides
"
use_bias(
?
inception_4d/5x5_reduceConvinception_4c/output"/
_output_shapes
:????????? "
pads

        "
group"
kernel_shape	
? "
strides
"
use_bias(
?
inception_4d/pool_projConvinception_4d/pool"
strides
"
use_bias("/
_output_shapes
:?????????@"
pads

        "
group"
kernel_shape	
?@
?
inception_4d/5x5Convinception_4d/relu_5x5_reduce"/
_output_shapes
:?????????@"
pads

    "
group"
kernel_shape
 @"
strides
"
use_bias(