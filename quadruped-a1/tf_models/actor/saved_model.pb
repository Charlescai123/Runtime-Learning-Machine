��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
�
actordense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*#
shared_nameactordense1/kernel
z
&actordense1/kernel/Read/ReadVariableOpReadVariableOpactordense1/kernel*
_output_shapes
:	�*
dtype0
y
actordense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameactordense1/bias
r
$actordense1/bias/Read/ReadVariableOpReadVariableOpactordense1/bias*
_output_shapes	
:�*
dtype0
�
actordense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*#
shared_nameactordense2/kernel
{
&actordense2/kernel/Read/ReadVariableOpReadVariableOpactordense2/kernel* 
_output_shapes
:
��*
dtype0
y
actordense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameactordense2/bias
r
$actordense2/bias/Read/ReadVariableOpReadVariableOpactordense2/bias*
_output_shapes	
:�*
dtype0
�
actordense3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*#
shared_nameactordense3/kernel
{
&actordense3/kernel/Read/ReadVariableOpReadVariableOpactordense3/kernel* 
_output_shapes
:
��*
dtype0
y
actordense3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameactordense3/bias
r
$actordense3/bias/Read/ReadVariableOpReadVariableOpactordense3/bias*
_output_shapes	
:�*
dtype0
�
actoroutput/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*#
shared_nameactoroutput/kernel
z
&actoroutput/kernel/Read/ReadVariableOpReadVariableOpactoroutput/kernel*
_output_shapes
:	�*
dtype0
x
actoroutput/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameactoroutput/bias
q
$actoroutput/bias/Read/ReadVariableOpReadVariableOpactoroutput/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
regularization_losses
trainable_variables
		keras_api


signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
�
	variables
#layer_regularization_losses
$layer_metrics
regularization_losses

%layers
trainable_variables
&metrics
'non_trainable_variables
 
^\
VARIABLE_VALUEactordense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEactordense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
(layer_regularization_losses
)layer_metrics
regularization_losses

*layers
trainable_variables
+metrics
,non_trainable_variables
^\
VARIABLE_VALUEactordense2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEactordense2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
-layer_regularization_losses
.layer_metrics
regularization_losses

/layers
trainable_variables
0metrics
1non_trainable_variables
^\
VARIABLE_VALUEactordense3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEactordense3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
2layer_regularization_losses
3layer_metrics
regularization_losses

4layers
trainable_variables
5metrics
6non_trainable_variables
^\
VARIABLE_VALUEactoroutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEactoroutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
7layer_regularization_losses
8layer_metrics
 regularization_losses

9layers
!trainable_variables
:metrics
;non_trainable_variables
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
}
serving_default_actorinputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_actorinputactordense1/kernelactordense1/biasactordense2/kernelactordense2/biasactordense3/kernelactordense3/biasactoroutput/kernelactoroutput/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference_signature_wrapper_795
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&actordense1/kernel/Read/ReadVariableOp$actordense1/bias/Read/ReadVariableOp&actordense2/kernel/Read/ReadVariableOp$actordense2/bias/Read/ReadVariableOp&actordense3/kernel/Read/ReadVariableOp$actordense3/bias/Read/ReadVariableOp&actoroutput/kernel/Read/ReadVariableOp$actoroutput/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_1030
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameactordense1/kernelactordense1/biasactordense2/kernelactordense2/biasactordense3/kernelactordense3/biasactoroutput/kernelactoroutput/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_1064��
�

�
D__inference_actordense2_layer_call_and_return_conditional_losses_534

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_actordense3_layer_call_fn_963

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense3_layer_call_and_return_conditional_losses_5512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
#__inference_actor_layer_call_fn_594

actorinput
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
actorinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_5752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
actorinput
�
�
>__inference_actor_layer_call_and_return_conditional_losses_747

actorinput"
actordense1_726:	�
actordense1_728:	�#
actordense2_731:
��
actordense2_733:	�#
actordense3_736:
��
actordense3_738:	�"
actoroutput_741:	�
actoroutput_743:
identity��#actordense1/StatefulPartitionedCall�#actordense2/StatefulPartitionedCall�#actordense3/StatefulPartitionedCall�#actoroutput/StatefulPartitionedCally
actordense1/CastCast
actorinput*

DstT0*

SrcT0*'
_output_shapes
:���������2
actordense1/Cast�
#actordense1/StatefulPartitionedCallStatefulPartitionedCallactordense1/Cast:y:0actordense1_726actordense1_728*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense1_layer_call_and_return_conditional_losses_5172%
#actordense1/StatefulPartitionedCall�
#actordense2/StatefulPartitionedCallStatefulPartitionedCall,actordense1/StatefulPartitionedCall:output:0actordense2_731actordense2_733*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense2_layer_call_and_return_conditional_losses_5342%
#actordense2/StatefulPartitionedCall�
#actordense3/StatefulPartitionedCallStatefulPartitionedCall,actordense2/StatefulPartitionedCall:output:0actordense3_736actordense3_738*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense3_layer_call_and_return_conditional_losses_5512%
#actordense3/StatefulPartitionedCall�
#actoroutput/StatefulPartitionedCallStatefulPartitionedCall,actordense3/StatefulPartitionedCall:output:0actoroutput_741actoroutput_743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actoroutput_layer_call_and_return_conditional_losses_5682%
#actoroutput/StatefulPartitionedCall�
IdentityIdentity,actoroutput/StatefulPartitionedCall:output:0$^actordense1/StatefulPartitionedCall$^actordense2/StatefulPartitionedCall$^actordense3/StatefulPartitionedCall$^actoroutput/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2J
#actordense1/StatefulPartitionedCall#actordense1/StatefulPartitionedCall2J
#actordense2/StatefulPartitionedCall#actordense2/StatefulPartitionedCall2J
#actordense3/StatefulPartitionedCall#actordense3/StatefulPartitionedCall2J
#actoroutput/StatefulPartitionedCall#actoroutput/StatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
actorinput
�

�
D__inference_actordense2_layer_call_and_return_conditional_losses_934

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
#__inference_actor_layer_call_fn_903

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_6822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
#__inference_actor_layer_call_fn_882

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_5752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference__traced_save_1030
file_prefix1
-savev2_actordense1_kernel_read_readvariableop/
+savev2_actordense1_bias_read_readvariableop1
-savev2_actordense2_kernel_read_readvariableop/
+savev2_actordense2_bias_read_readvariableop1
-savev2_actordense3_kernel_read_readvariableop/
+savev2_actordense3_bias_read_readvariableop1
-savev2_actoroutput_kernel_read_readvariableop/
+savev2_actoroutput_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_actordense1_kernel_read_readvariableop+savev2_actordense1_bias_read_readvariableop-savev2_actordense2_kernel_read_readvariableop+savev2_actordense2_bias_read_readvariableop-savev2_actordense3_kernel_read_readvariableop+savev2_actordense3_bias_read_readvariableop-savev2_actoroutput_kernel_read_readvariableop+savev2_actoroutput_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*`
_input_shapesO
M: :	�:�:
��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::	

_output_shapes
: 
�
�
>__inference_actor_layer_call_and_return_conditional_losses_682

inputs"
actordense1_661:	�
actordense1_663:	�#
actordense2_666:
��
actordense2_668:	�#
actordense3_671:
��
actordense3_673:	�"
actoroutput_676:	�
actoroutput_678:
identity��#actordense1/StatefulPartitionedCall�#actordense2/StatefulPartitionedCall�#actordense3/StatefulPartitionedCall�#actoroutput/StatefulPartitionedCallu
actordense1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
actordense1/Cast�
#actordense1/StatefulPartitionedCallStatefulPartitionedCallactordense1/Cast:y:0actordense1_661actordense1_663*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense1_layer_call_and_return_conditional_losses_5172%
#actordense1/StatefulPartitionedCall�
#actordense2/StatefulPartitionedCallStatefulPartitionedCall,actordense1/StatefulPartitionedCall:output:0actordense2_666actordense2_668*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense2_layer_call_and_return_conditional_losses_5342%
#actordense2/StatefulPartitionedCall�
#actordense3/StatefulPartitionedCallStatefulPartitionedCall,actordense2/StatefulPartitionedCall:output:0actordense3_671actordense3_673*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense3_layer_call_and_return_conditional_losses_5512%
#actordense3/StatefulPartitionedCall�
#actoroutput/StatefulPartitionedCallStatefulPartitionedCall,actordense3/StatefulPartitionedCall:output:0actoroutput_676actoroutput_678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actoroutput_layer_call_and_return_conditional_losses_5682%
#actoroutput/StatefulPartitionedCall�
IdentityIdentity,actoroutput/StatefulPartitionedCall:output:0$^actordense1/StatefulPartitionedCall$^actordense2/StatefulPartitionedCall$^actordense3/StatefulPartitionedCall$^actoroutput/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2J
#actordense1/StatefulPartitionedCall#actordense1/StatefulPartitionedCall2J
#actordense2/StatefulPartitionedCall#actordense2/StatefulPartitionedCall2J
#actordense3/StatefulPartitionedCall#actordense3/StatefulPartitionedCall2J
#actoroutput/StatefulPartitionedCall#actoroutput/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_actoroutput_layer_call_and_return_conditional_losses_974

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
>__inference_actor_layer_call_and_return_conditional_losses_575

inputs"
actordense1_518:	�
actordense1_520:	�#
actordense2_535:
��
actordense2_537:	�#
actordense3_552:
��
actordense3_554:	�"
actoroutput_569:	�
actoroutput_571:
identity��#actordense1/StatefulPartitionedCall�#actordense2/StatefulPartitionedCall�#actordense3/StatefulPartitionedCall�#actoroutput/StatefulPartitionedCallu
actordense1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
actordense1/Cast�
#actordense1/StatefulPartitionedCallStatefulPartitionedCallactordense1/Cast:y:0actordense1_518actordense1_520*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense1_layer_call_and_return_conditional_losses_5172%
#actordense1/StatefulPartitionedCall�
#actordense2/StatefulPartitionedCallStatefulPartitionedCall,actordense1/StatefulPartitionedCall:output:0actordense2_535actordense2_537*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense2_layer_call_and_return_conditional_losses_5342%
#actordense2/StatefulPartitionedCall�
#actordense3/StatefulPartitionedCallStatefulPartitionedCall,actordense2/StatefulPartitionedCall:output:0actordense3_552actordense3_554*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense3_layer_call_and_return_conditional_losses_5512%
#actordense3/StatefulPartitionedCall�
#actoroutput/StatefulPartitionedCallStatefulPartitionedCall,actordense3/StatefulPartitionedCall:output:0actoroutput_569actoroutput_571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actoroutput_layer_call_and_return_conditional_losses_5682%
#actoroutput/StatefulPartitionedCall�
IdentityIdentity,actoroutput/StatefulPartitionedCall:output:0$^actordense1/StatefulPartitionedCall$^actordense2/StatefulPartitionedCall$^actordense3/StatefulPartitionedCall$^actoroutput/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2J
#actordense1/StatefulPartitionedCall#actordense1/StatefulPartitionedCall2J
#actordense2/StatefulPartitionedCall#actordense2/StatefulPartitionedCall2J
#actordense3/StatefulPartitionedCall#actordense3/StatefulPartitionedCall2J
#actoroutput/StatefulPartitionedCall#actoroutput/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_actor_layer_call_and_return_conditional_losses_772

actorinput"
actordense1_751:	�
actordense1_753:	�#
actordense2_756:
��
actordense2_758:	�#
actordense3_761:
��
actordense3_763:	�"
actoroutput_766:	�
actoroutput_768:
identity��#actordense1/StatefulPartitionedCall�#actordense2/StatefulPartitionedCall�#actordense3/StatefulPartitionedCall�#actoroutput/StatefulPartitionedCally
actordense1/CastCast
actorinput*

DstT0*

SrcT0*'
_output_shapes
:���������2
actordense1/Cast�
#actordense1/StatefulPartitionedCallStatefulPartitionedCallactordense1/Cast:y:0actordense1_751actordense1_753*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense1_layer_call_and_return_conditional_losses_5172%
#actordense1/StatefulPartitionedCall�
#actordense2/StatefulPartitionedCallStatefulPartitionedCall,actordense1/StatefulPartitionedCall:output:0actordense2_756actordense2_758*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense2_layer_call_and_return_conditional_losses_5342%
#actordense2/StatefulPartitionedCall�
#actordense3/StatefulPartitionedCallStatefulPartitionedCall,actordense2/StatefulPartitionedCall:output:0actordense3_761actordense3_763*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense3_layer_call_and_return_conditional_losses_5512%
#actordense3/StatefulPartitionedCall�
#actoroutput/StatefulPartitionedCallStatefulPartitionedCall,actordense3/StatefulPartitionedCall:output:0actoroutput_766actoroutput_768*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actoroutput_layer_call_and_return_conditional_losses_5682%
#actoroutput/StatefulPartitionedCall�
IdentityIdentity,actoroutput/StatefulPartitionedCall:output:0$^actordense1/StatefulPartitionedCall$^actordense2/StatefulPartitionedCall$^actordense3/StatefulPartitionedCall$^actoroutput/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2J
#actordense1/StatefulPartitionedCall#actordense1/StatefulPartitionedCall2J
#actordense2/StatefulPartitionedCall#actordense2/StatefulPartitionedCall2J
#actordense3/StatefulPartitionedCall#actordense3/StatefulPartitionedCall2J
#actoroutput/StatefulPartitionedCall#actoroutput/StatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
actorinput
�
�
)__inference_actordense1_layer_call_fn_923

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense1_layer_call_and_return_conditional_losses_5172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_actordense3_layer_call_and_return_conditional_losses_551

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
!__inference_signature_wrapper_795

actorinput
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
actorinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__wrapped_model_4982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
actorinput
�

�
D__inference_actordense3_layer_call_and_return_conditional_losses_954

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_actoroutput_layer_call_and_return_conditional_losses_568

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
 __inference__traced_restore_1064
file_prefix6
#assignvariableop_actordense1_kernel:	�2
#assignvariableop_1_actordense1_bias:	�9
%assignvariableop_2_actordense2_kernel:
��2
#assignvariableop_3_actordense2_bias:	�9
%assignvariableop_4_actordense3_kernel:
��2
#assignvariableop_5_actordense3_bias:	�8
%assignvariableop_6_actoroutput_kernel:	�1
#assignvariableop_7_actoroutput_bias:

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp#assignvariableop_actordense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_actordense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp%assignvariableop_2_actordense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_actordense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_actordense3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_actordense3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_actoroutput_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_actoroutput_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8�

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�+
�
>__inference_actor_layer_call_and_return_conditional_losses_828

inputs=
*actordense1_matmul_readvariableop_resource:	�:
+actordense1_biasadd_readvariableop_resource:	�>
*actordense2_matmul_readvariableop_resource:
��:
+actordense2_biasadd_readvariableop_resource:	�>
*actordense3_matmul_readvariableop_resource:
��:
+actordense3_biasadd_readvariableop_resource:	�=
*actoroutput_matmul_readvariableop_resource:	�9
+actoroutput_biasadd_readvariableop_resource:
identity��"actordense1/BiasAdd/ReadVariableOp�!actordense1/MatMul/ReadVariableOp�"actordense2/BiasAdd/ReadVariableOp�!actordense2/MatMul/ReadVariableOp�"actordense3/BiasAdd/ReadVariableOp�!actordense3/MatMul/ReadVariableOp�"actoroutput/BiasAdd/ReadVariableOp�!actoroutput/MatMul/ReadVariableOpu
actordense1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
actordense1/Cast�
!actordense1/MatMul/ReadVariableOpReadVariableOp*actordense1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!actordense1/MatMul/ReadVariableOp�
actordense1/MatMulMatMulactordense1/Cast:y:0)actordense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense1/MatMul�
"actordense1/BiasAdd/ReadVariableOpReadVariableOp+actordense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"actordense1/BiasAdd/ReadVariableOp�
actordense1/BiasAddBiasAddactordense1/MatMul:product:0*actordense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense1/BiasAdd}
actordense1/ReluReluactordense1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actordense1/Relu�
!actordense2/MatMul/ReadVariableOpReadVariableOp*actordense2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!actordense2/MatMul/ReadVariableOp�
actordense2/MatMulMatMulactordense1/Relu:activations:0)actordense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense2/MatMul�
"actordense2/BiasAdd/ReadVariableOpReadVariableOp+actordense2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"actordense2/BiasAdd/ReadVariableOp�
actordense2/BiasAddBiasAddactordense2/MatMul:product:0*actordense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense2/BiasAdd}
actordense2/ReluReluactordense2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actordense2/Relu�
!actordense3/MatMul/ReadVariableOpReadVariableOp*actordense3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!actordense3/MatMul/ReadVariableOp�
actordense3/MatMulMatMulactordense2/Relu:activations:0)actordense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense3/MatMul�
"actordense3/BiasAdd/ReadVariableOpReadVariableOp+actordense3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"actordense3/BiasAdd/ReadVariableOp�
actordense3/BiasAddBiasAddactordense3/MatMul:product:0*actordense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense3/BiasAdd}
actordense3/ReluReluactordense3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actordense3/Relu�
!actoroutput/MatMul/ReadVariableOpReadVariableOp*actoroutput_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!actoroutput/MatMul/ReadVariableOp�
actoroutput/MatMulMatMulactordense3/Relu:activations:0)actoroutput/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actoroutput/MatMul�
"actoroutput/BiasAdd/ReadVariableOpReadVariableOp+actoroutput_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"actoroutput/BiasAdd/ReadVariableOp�
actoroutput/BiasAddBiasAddactoroutput/MatMul:product:0*actoroutput/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actoroutput/BiasAdd|
actoroutput/TanhTanhactoroutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
actoroutput/Tanh�
IdentityIdentityactoroutput/Tanh:y:0#^actordense1/BiasAdd/ReadVariableOp"^actordense1/MatMul/ReadVariableOp#^actordense2/BiasAdd/ReadVariableOp"^actordense2/MatMul/ReadVariableOp#^actordense3/BiasAdd/ReadVariableOp"^actordense3/MatMul/ReadVariableOp#^actoroutput/BiasAdd/ReadVariableOp"^actoroutput/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2H
"actordense1/BiasAdd/ReadVariableOp"actordense1/BiasAdd/ReadVariableOp2F
!actordense1/MatMul/ReadVariableOp!actordense1/MatMul/ReadVariableOp2H
"actordense2/BiasAdd/ReadVariableOp"actordense2/BiasAdd/ReadVariableOp2F
!actordense2/MatMul/ReadVariableOp!actordense2/MatMul/ReadVariableOp2H
"actordense3/BiasAdd/ReadVariableOp"actordense3/BiasAdd/ReadVariableOp2F
!actordense3/MatMul/ReadVariableOp!actordense3/MatMul/ReadVariableOp2H
"actoroutput/BiasAdd/ReadVariableOp"actoroutput/BiasAdd/ReadVariableOp2F
!actoroutput/MatMul/ReadVariableOp!actoroutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_actordense1_layer_call_and_return_conditional_losses_517

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_actordense1_layer_call_and_return_conditional_losses_914

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
#__inference_actor_layer_call_fn_722

actorinput
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
actorinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_6822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
actorinput
�+
�
>__inference_actor_layer_call_and_return_conditional_losses_861

inputs=
*actordense1_matmul_readvariableop_resource:	�:
+actordense1_biasadd_readvariableop_resource:	�>
*actordense2_matmul_readvariableop_resource:
��:
+actordense2_biasadd_readvariableop_resource:	�>
*actordense3_matmul_readvariableop_resource:
��:
+actordense3_biasadd_readvariableop_resource:	�=
*actoroutput_matmul_readvariableop_resource:	�9
+actoroutput_biasadd_readvariableop_resource:
identity��"actordense1/BiasAdd/ReadVariableOp�!actordense1/MatMul/ReadVariableOp�"actordense2/BiasAdd/ReadVariableOp�!actordense2/MatMul/ReadVariableOp�"actordense3/BiasAdd/ReadVariableOp�!actordense3/MatMul/ReadVariableOp�"actoroutput/BiasAdd/ReadVariableOp�!actoroutput/MatMul/ReadVariableOpu
actordense1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
actordense1/Cast�
!actordense1/MatMul/ReadVariableOpReadVariableOp*actordense1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!actordense1/MatMul/ReadVariableOp�
actordense1/MatMulMatMulactordense1/Cast:y:0)actordense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense1/MatMul�
"actordense1/BiasAdd/ReadVariableOpReadVariableOp+actordense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"actordense1/BiasAdd/ReadVariableOp�
actordense1/BiasAddBiasAddactordense1/MatMul:product:0*actordense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense1/BiasAdd}
actordense1/ReluReluactordense1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actordense1/Relu�
!actordense2/MatMul/ReadVariableOpReadVariableOp*actordense2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!actordense2/MatMul/ReadVariableOp�
actordense2/MatMulMatMulactordense1/Relu:activations:0)actordense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense2/MatMul�
"actordense2/BiasAdd/ReadVariableOpReadVariableOp+actordense2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"actordense2/BiasAdd/ReadVariableOp�
actordense2/BiasAddBiasAddactordense2/MatMul:product:0*actordense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense2/BiasAdd}
actordense2/ReluReluactordense2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actordense2/Relu�
!actordense3/MatMul/ReadVariableOpReadVariableOp*actordense3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!actordense3/MatMul/ReadVariableOp�
actordense3/MatMulMatMulactordense2/Relu:activations:0)actordense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense3/MatMul�
"actordense3/BiasAdd/ReadVariableOpReadVariableOp+actordense3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"actordense3/BiasAdd/ReadVariableOp�
actordense3/BiasAddBiasAddactordense3/MatMul:product:0*actordense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actordense3/BiasAdd}
actordense3/ReluReluactordense3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actordense3/Relu�
!actoroutput/MatMul/ReadVariableOpReadVariableOp*actoroutput_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!actoroutput/MatMul/ReadVariableOp�
actoroutput/MatMulMatMulactordense3/Relu:activations:0)actoroutput/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actoroutput/MatMul�
"actoroutput/BiasAdd/ReadVariableOpReadVariableOp+actoroutput_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"actoroutput/BiasAdd/ReadVariableOp�
actoroutput/BiasAddBiasAddactoroutput/MatMul:product:0*actoroutput/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actoroutput/BiasAdd|
actoroutput/TanhTanhactoroutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
actoroutput/Tanh�
IdentityIdentityactoroutput/Tanh:y:0#^actordense1/BiasAdd/ReadVariableOp"^actordense1/MatMul/ReadVariableOp#^actordense2/BiasAdd/ReadVariableOp"^actordense2/MatMul/ReadVariableOp#^actordense3/BiasAdd/ReadVariableOp"^actordense3/MatMul/ReadVariableOp#^actoroutput/BiasAdd/ReadVariableOp"^actoroutput/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2H
"actordense1/BiasAdd/ReadVariableOp"actordense1/BiasAdd/ReadVariableOp2F
!actordense1/MatMul/ReadVariableOp!actordense1/MatMul/ReadVariableOp2H
"actordense2/BiasAdd/ReadVariableOp"actordense2/BiasAdd/ReadVariableOp2F
!actordense2/MatMul/ReadVariableOp!actordense2/MatMul/ReadVariableOp2H
"actordense3/BiasAdd/ReadVariableOp"actordense3/BiasAdd/ReadVariableOp2F
!actordense3/MatMul/ReadVariableOp!actordense3/MatMul/ReadVariableOp2H
"actoroutput/BiasAdd/ReadVariableOp"actoroutput/BiasAdd/ReadVariableOp2F
!actoroutput/MatMul/ReadVariableOp!actoroutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_actordense2_layer_call_fn_943

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actordense2_layer_call_and_return_conditional_losses_5342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_actoroutput_layer_call_fn_983

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_actoroutput_layer_call_and_return_conditional_losses_5682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�0
�
__inference__wrapped_model_498

actorinputC
0actor_actordense1_matmul_readvariableop_resource:	�@
1actor_actordense1_biasadd_readvariableop_resource:	�D
0actor_actordense2_matmul_readvariableop_resource:
��@
1actor_actordense2_biasadd_readvariableop_resource:	�D
0actor_actordense3_matmul_readvariableop_resource:
��@
1actor_actordense3_biasadd_readvariableop_resource:	�C
0actor_actoroutput_matmul_readvariableop_resource:	�?
1actor_actoroutput_biasadd_readvariableop_resource:
identity��(actor/actordense1/BiasAdd/ReadVariableOp�'actor/actordense1/MatMul/ReadVariableOp�(actor/actordense2/BiasAdd/ReadVariableOp�'actor/actordense2/MatMul/ReadVariableOp�(actor/actordense3/BiasAdd/ReadVariableOp�'actor/actordense3/MatMul/ReadVariableOp�(actor/actoroutput/BiasAdd/ReadVariableOp�'actor/actoroutput/MatMul/ReadVariableOp�
actor/actordense1/CastCast
actorinput*

DstT0*

SrcT0*'
_output_shapes
:���������2
actor/actordense1/Cast�
'actor/actordense1/MatMul/ReadVariableOpReadVariableOp0actor_actordense1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'actor/actordense1/MatMul/ReadVariableOp�
actor/actordense1/MatMulMatMulactor/actordense1/Cast:y:0/actor/actordense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor/actordense1/MatMul�
(actor/actordense1/BiasAdd/ReadVariableOpReadVariableOp1actor_actordense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(actor/actordense1/BiasAdd/ReadVariableOp�
actor/actordense1/BiasAddBiasAdd"actor/actordense1/MatMul:product:00actor/actordense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor/actordense1/BiasAdd�
actor/actordense1/ReluRelu"actor/actordense1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actor/actordense1/Relu�
'actor/actordense2/MatMul/ReadVariableOpReadVariableOp0actor_actordense2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02)
'actor/actordense2/MatMul/ReadVariableOp�
actor/actordense2/MatMulMatMul$actor/actordense1/Relu:activations:0/actor/actordense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor/actordense2/MatMul�
(actor/actordense2/BiasAdd/ReadVariableOpReadVariableOp1actor_actordense2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(actor/actordense2/BiasAdd/ReadVariableOp�
actor/actordense2/BiasAddBiasAdd"actor/actordense2/MatMul:product:00actor/actordense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor/actordense2/BiasAdd�
actor/actordense2/ReluRelu"actor/actordense2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actor/actordense2/Relu�
'actor/actordense3/MatMul/ReadVariableOpReadVariableOp0actor_actordense3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02)
'actor/actordense3/MatMul/ReadVariableOp�
actor/actordense3/MatMulMatMul$actor/actordense2/Relu:activations:0/actor/actordense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor/actordense3/MatMul�
(actor/actordense3/BiasAdd/ReadVariableOpReadVariableOp1actor_actordense3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(actor/actordense3/BiasAdd/ReadVariableOp�
actor/actordense3/BiasAddBiasAdd"actor/actordense3/MatMul:product:00actor/actordense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
actor/actordense3/BiasAdd�
actor/actordense3/ReluRelu"actor/actordense3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
actor/actordense3/Relu�
'actor/actoroutput/MatMul/ReadVariableOpReadVariableOp0actor_actoroutput_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'actor/actoroutput/MatMul/ReadVariableOp�
actor/actoroutput/MatMulMatMul$actor/actordense3/Relu:activations:0/actor/actoroutput/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actor/actoroutput/MatMul�
(actor/actoroutput/BiasAdd/ReadVariableOpReadVariableOp1actor_actoroutput_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(actor/actoroutput/BiasAdd/ReadVariableOp�
actor/actoroutput/BiasAddBiasAdd"actor/actoroutput/MatMul:product:00actor/actoroutput/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
actor/actoroutput/BiasAdd�
actor/actoroutput/TanhTanh"actor/actoroutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
actor/actoroutput/Tanh�
IdentityIdentityactor/actoroutput/Tanh:y:0)^actor/actordense1/BiasAdd/ReadVariableOp(^actor/actordense1/MatMul/ReadVariableOp)^actor/actordense2/BiasAdd/ReadVariableOp(^actor/actordense2/MatMul/ReadVariableOp)^actor/actordense3/BiasAdd/ReadVariableOp(^actor/actordense3/MatMul/ReadVariableOp)^actor/actoroutput/BiasAdd/ReadVariableOp(^actor/actoroutput/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2T
(actor/actordense1/BiasAdd/ReadVariableOp(actor/actordense1/BiasAdd/ReadVariableOp2R
'actor/actordense1/MatMul/ReadVariableOp'actor/actordense1/MatMul/ReadVariableOp2T
(actor/actordense2/BiasAdd/ReadVariableOp(actor/actordense2/BiasAdd/ReadVariableOp2R
'actor/actordense2/MatMul/ReadVariableOp'actor/actordense2/MatMul/ReadVariableOp2T
(actor/actordense3/BiasAdd/ReadVariableOp(actor/actordense3/BiasAdd/ReadVariableOp2R
'actor/actordense3/MatMul/ReadVariableOp'actor/actordense3/MatMul/ReadVariableOp2T
(actor/actoroutput/BiasAdd/ReadVariableOp(actor/actoroutput/BiasAdd/ReadVariableOp2R
'actor/actoroutput/MatMul/ReadVariableOp'actor/actoroutput/MatMul/ReadVariableOp:S O
'
_output_shapes
:���������
$
_user_specified_name
actorinput"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A

actorinput3
serving_default_actorinput:0���������?
actoroutput0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�/
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
regularization_losses
trainable_variables
		keras_api


signatures
*<&call_and_return_all_conditional_losses
=__call__
>_default_save_signature"�-
_tf_keras_network�,{"name": "actor", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "actor", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float16", "sparse": false, "ragged": false, "name": "actorinput"}, "name": "actorinput", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "actordense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "actordense1", "inbound_nodes": [[["actorinput", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "actordense2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "actordense2", "inbound_nodes": [[["actordense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "actordense3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "actordense3", "inbound_nodes": [[["actordense2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "actoroutput", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "actoroutput", "inbound_nodes": [[["actordense3", 0, 0, {}]]]}], "input_layers": [["actorinput", 0, 0]], "output_layers": [["actoroutput", 0, 0]]}, "shared_object_id": 13, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 12]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 12]}, "float16", "actorinput"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "actor", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float16", "sparse": false, "ragged": false, "name": "actorinput"}, "name": "actorinput", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "actordense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "actordense1", "inbound_nodes": [[["actorinput", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "actordense2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "actordense2", "inbound_nodes": [[["actordense1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "actordense3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "actordense3", "inbound_nodes": [[["actordense2", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "actoroutput", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "actoroutput", "inbound_nodes": [[["actordense3", 0, 0, {}]]], "shared_object_id": 12}], "input_layers": [["actorinput", 0, 0]], "output_layers": [["actoroutput", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "actorinput", "dtype": "float16", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float16", "sparse": false, "ragged": false, "name": "actorinput"}}
�	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"�
_tf_keras_layer�{"name": "actordense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "actordense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["actorinput", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
�	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"�
_tf_keras_layer�{"name": "actordense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "actordense2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["actordense1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*C&call_and_return_all_conditional_losses
D__call__"�
_tf_keras_layer�{"name": "actordense3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "actordense3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["actordense2", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�	

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
*E&call_and_return_all_conditional_losses
F__call__"�
_tf_keras_layer�{"name": "actoroutput", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "actoroutput", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["actordense3", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
�
	variables
#layer_regularization_losses
$layer_metrics
regularization_losses

%layers
trainable_variables
&metrics
'non_trainable_variables
=__call__
>_default_save_signature
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
,
Gserving_default"
signature_map
%:#	�2actordense1/kernel
:�2actordense1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
(layer_regularization_losses
)layer_metrics
regularization_losses

*layers
trainable_variables
+metrics
,non_trainable_variables
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
&:$
��2actordense2/kernel
:�2actordense2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
-layer_regularization_losses
.layer_metrics
regularization_losses

/layers
trainable_variables
0metrics
1non_trainable_variables
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
&:$
��2actordense3/kernel
:�2actordense3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
2layer_regularization_losses
3layer_metrics
regularization_losses

4layers
trainable_variables
5metrics
6non_trainable_variables
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
%:#	�2actoroutput/kernel
:2actoroutput/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
7layer_regularization_losses
8layer_metrics
 regularization_losses

9layers
!trainable_variables
:metrics
;non_trainable_variables
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
>__inference_actor_layer_call_and_return_conditional_losses_828
>__inference_actor_layer_call_and_return_conditional_losses_861
>__inference_actor_layer_call_and_return_conditional_losses_747
>__inference_actor_layer_call_and_return_conditional_losses_772�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference_actor_layer_call_fn_594
#__inference_actor_layer_call_fn_882
#__inference_actor_layer_call_fn_903
#__inference_actor_layer_call_fn_722�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_498�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *)�&
$�!

actorinput���������
�2�
D__inference_actordense1_layer_call_and_return_conditional_losses_914�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_actordense1_layer_call_fn_923�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_actordense2_layer_call_and_return_conditional_losses_934�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_actordense2_layer_call_fn_943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_actordense3_layer_call_and_return_conditional_losses_954�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_actordense3_layer_call_fn_963�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_actoroutput_layer_call_and_return_conditional_losses_974�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_actoroutput_layer_call_fn_983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
!__inference_signature_wrapper_795
actorinput"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_498z3�0
)�&
$�!

actorinput���������
� "9�6
4
actoroutput%�"
actoroutput����������
>__inference_actor_layer_call_and_return_conditional_losses_747n;�8
1�.
$�!

actorinput���������
p 

 
� "%�"
�
0���������
� �
>__inference_actor_layer_call_and_return_conditional_losses_772n;�8
1�.
$�!

actorinput���������
p

 
� "%�"
�
0���������
� �
>__inference_actor_layer_call_and_return_conditional_losses_828j7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
>__inference_actor_layer_call_and_return_conditional_losses_861j7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
#__inference_actor_layer_call_fn_594a;�8
1�.
$�!

actorinput���������
p 

 
� "�����������
#__inference_actor_layer_call_fn_722a;�8
1�.
$�!

actorinput���������
p

 
� "�����������
#__inference_actor_layer_call_fn_882]7�4
-�*
 �
inputs���������
p 

 
� "�����������
#__inference_actor_layer_call_fn_903]7�4
-�*
 �
inputs���������
p

 
� "�����������
D__inference_actordense1_layer_call_and_return_conditional_losses_914]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� }
)__inference_actordense1_layer_call_fn_923P/�,
%�"
 �
inputs���������
� "������������
D__inference_actordense2_layer_call_and_return_conditional_losses_934^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_actordense2_layer_call_fn_943Q0�-
&�#
!�
inputs����������
� "������������
D__inference_actordense3_layer_call_and_return_conditional_losses_954^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_actordense3_layer_call_fn_963Q0�-
&�#
!�
inputs����������
� "������������
D__inference_actoroutput_layer_call_and_return_conditional_losses_974]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
)__inference_actoroutput_layer_call_fn_983P0�-
&�#
!�
inputs����������
� "�����������
!__inference_signature_wrapper_795�A�>
� 
7�4
2

actorinput$�!

actorinput���������"9�6
4
actoroutput%�"
actoroutput���������