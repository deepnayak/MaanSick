??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

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
?
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
conv3d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_44/kernel
?
$conv3d_44/kernel/Read/ReadVariableOpReadVariableOpconv3d_44/kernel**
_output_shapes
:*
dtype0
t
conv3d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_44/bias
m
"conv3d_44/bias/Read/ReadVariableOpReadVariableOpconv3d_44/bias*
_output_shapes
:*
dtype0
?
conv3d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_45/kernel
?
$conv3d_45/kernel/Read/ReadVariableOpReadVariableOpconv3d_45/kernel**
_output_shapes
:*
dtype0
t
conv3d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_45/bias
m
"conv3d_45/bias/Read/ReadVariableOpReadVariableOpconv3d_45/bias*
_output_shapes
:*
dtype0
?
batch_normalization_44/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_44/gamma
?
0batch_normalization_44/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_44/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_44/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_44/beta
?
/batch_normalization_44/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_44/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_44/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_44/moving_mean
?
6batch_normalization_44/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_44/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_44/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_44/moving_variance
?
:batch_normalization_44/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_44/moving_variance*
_output_shapes
:*
dtype0
?
conv3d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_46/kernel
?
$conv3d_46/kernel/Read/ReadVariableOpReadVariableOpconv3d_46/kernel**
_output_shapes
:*
dtype0
t
conv3d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_46/bias
m
"conv3d_46/bias/Read/ReadVariableOpReadVariableOpconv3d_46/bias*
_output_shapes
:*
dtype0
?
conv3d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_47/kernel
?
$conv3d_47/kernel/Read/ReadVariableOpReadVariableOpconv3d_47/kernel**
_output_shapes
:*
dtype0
t
conv3d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_47/bias
m
"conv3d_47/bias/Read/ReadVariableOpReadVariableOpconv3d_47/bias*
_output_shapes
:*
dtype0
?
batch_normalization_45/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_45/gamma
?
0batch_normalization_45/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_45/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_45/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_45/beta
?
/batch_normalization_45/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_45/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_45/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_45/moving_mean
?
6batch_normalization_45/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_45/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_45/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_45/moving_variance
?
:batch_normalization_45/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_45/moving_variance*
_output_shapes
:*
dtype0
|
dense_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?-?* 
shared_namedense_66/kernel
u
#dense_66/kernel/Read/ReadVariableOpReadVariableOpdense_66/kernel* 
_output_shapes
:
?-?*
dtype0
s
dense_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_66/bias
l
!dense_66/bias/Read/ReadVariableOpReadVariableOpdense_66/bias*
_output_shapes	
:?*
dtype0
{
dense_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_67/kernel
t
#dense_67/kernel/Read/ReadVariableOpReadVariableOpdense_67/kernel*
_output_shapes
:	?@*
dtype0
r
dense_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_67/bias
k
!dense_67/bias/Read/ReadVariableOpReadVariableOpdense_67/bias*
_output_shapes
:@*
dtype0
z
dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_68/kernel
s
#dense_68/kernel/Read/ReadVariableOpReadVariableOpdense_68/kernel*
_output_shapes

:@*
dtype0
r
dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_68/bias
k
!dense_68/bias/Read/ReadVariableOpReadVariableOpdense_68/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv3d_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_44/kernel/m
?
+Adam/conv3d_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_44/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_44/bias/m
{
)Adam/conv3d_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_44/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_45/kernel/m
?
+Adam/conv3d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_45/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_45/bias/m
{
)Adam/conv3d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_45/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_44/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_44/gamma/m
?
7Adam/batch_normalization_44/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_44/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_44/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_44/beta/m
?
6Adam/batch_normalization_44/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_44/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_46/kernel/m
?
+Adam/conv3d_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_46/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_46/bias/m
{
)Adam/conv3d_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_46/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_47/kernel/m
?
+Adam/conv3d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_47/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_47/bias/m
{
)Adam/conv3d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_47/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_45/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_45/gamma/m
?
7Adam/batch_normalization_45/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_45/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_45/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_45/beta/m
?
6Adam/batch_normalization_45/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_45/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?-?*'
shared_nameAdam/dense_66/kernel/m
?
*Adam/dense_66/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/m* 
_output_shapes
:
?-?*
dtype0
?
Adam/dense_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_66/bias/m
z
(Adam/dense_66/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_67/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_67/kernel/m
?
*Adam/dense_67/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_67/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_67/bias/m
y
(Adam/dense_67/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_68/kernel/m
?
*Adam/dense_68/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_68/bias/m
y
(Adam/dense_68/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_44/kernel/v
?
+Adam/conv3d_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_44/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_44/bias/v
{
)Adam/conv3d_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_44/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_45/kernel/v
?
+Adam/conv3d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_45/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_45/bias/v
{
)Adam/conv3d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_45/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_44/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_44/gamma/v
?
7Adam/batch_normalization_44/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_44/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_44/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_44/beta/v
?
6Adam/batch_normalization_44/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_44/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_46/kernel/v
?
+Adam/conv3d_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_46/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_46/bias/v
{
)Adam/conv3d_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_46/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_47/kernel/v
?
+Adam/conv3d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_47/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_47/bias/v
{
)Adam/conv3d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_47/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_45/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_45/gamma/v
?
7Adam/batch_normalization_45/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_45/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_45/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_45/beta/v
?
6Adam/batch_normalization_45/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_45/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?-?*'
shared_nameAdam/dense_66/kernel/v
?
*Adam/dense_66/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/v* 
_output_shapes
:
?-?*
dtype0
?
Adam/dense_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_66/bias/v
z
(Adam/dense_66/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_67/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_67/kernel/v
?
*Adam/dense_67/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_67/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_67/bias/v
y
(Adam/dense_67/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_68/kernel/v
?
*Adam/dense_68/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_68/bias/v
y
(Adam/dense_68/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?h
value?gB?g B?g
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
 	variables
!regularization_losses
"	keras_api
?
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
h

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
R
8trainable_variables
9	variables
:regularization_losses
;	keras_api
?
<axis
	=gamma
>beta
?moving_mean
@moving_variance
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
R
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
h

Okernel
Pbias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
?
[iter

\beta_1

]beta_2
	^decay
_learning_ratem?m?m?m?$m?%m?,m?-m?2m?3m?=m?>m?Im?Jm?Om?Pm?Um?Vm?v?v?v?v?$v?%v?,v?-v?2v?3v?=v?>v?Iv?Jv?Ov?Pv?Uv?Vv?
?
0
1
2
3
$4
%5
,6
-7
28
39
=10
>11
I12
J13
O14
P15
U16
V17
?
0
1
2
3
$4
%5
&6
'7
,8
-9
210
311
=12
>13
?14
@15
I16
J17
O18
P19
U20
V21
 
?
`non_trainable_variables
trainable_variables
ametrics
blayer_metrics

clayers
	variables
regularization_losses
dlayer_regularization_losses
 
\Z
VARIABLE_VALUEconv3d_44/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_44/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
enon_trainable_variables
trainable_variables
fmetrics
glayer_metrics

hlayers
	variables
regularization_losses
ilayer_regularization_losses
\Z
VARIABLE_VALUEconv3d_45/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_45/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
jnon_trainable_variables
trainable_variables
kmetrics
llayer_metrics

mlayers
	variables
regularization_losses
nlayer_regularization_losses
 
 
 
?
onon_trainable_variables
trainable_variables
pmetrics
qlayer_metrics

rlayers
 	variables
!regularization_losses
slayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_44/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_44/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_44/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_44/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
&2
'3
 
?
tnon_trainable_variables
(trainable_variables
umetrics
vlayer_metrics

wlayers
)	variables
*regularization_losses
xlayer_regularization_losses
\Z
VARIABLE_VALUEconv3d_46/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_46/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
?
ynon_trainable_variables
.trainable_variables
zmetrics
{layer_metrics

|layers
/	variables
0regularization_losses
}layer_regularization_losses
\Z
VARIABLE_VALUEconv3d_47/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_47/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 
?
~non_trainable_variables
4trainable_variables
metrics
?layer_metrics
?layers
5	variables
6regularization_losses
 ?layer_regularization_losses
 
 
 
?
?non_trainable_variables
8trainable_variables
?metrics
?layer_metrics
?layers
9	variables
:regularization_losses
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_45/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_45/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_45/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_45/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
?2
@3
 
?
?non_trainable_variables
Atrainable_variables
?metrics
?layer_metrics
?layers
B	variables
Cregularization_losses
 ?layer_regularization_losses
 
 
 
?
?non_trainable_variables
Etrainable_variables
?metrics
?layer_metrics
?layers
F	variables
Gregularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_66/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_66/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
?
?non_trainable_variables
Ktrainable_variables
?metrics
?layer_metrics
?layers
L	variables
Mregularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_67/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_67/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

O0
P1
 
?
?non_trainable_variables
Qtrainable_variables
?metrics
?layer_metrics
?layers
R	variables
Sregularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_68/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_68/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
?
?non_trainable_variables
Wtrainable_variables
?metrics
?layer_metrics
?layers
X	variables
Yregularization_losses
 ?layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
?2
@3

?0
?1
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
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

&0
'1
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

?0
@1
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
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv3d_44/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_44/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_45/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_45/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_44/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_44/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_46/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_46/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_47/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_47/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_45/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_45/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_66/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_66/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_67/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_67/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_68/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_68/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_44/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_44/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_45/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_45/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_44/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_44/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_46/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_46/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_47/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_47/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_45/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_45/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_66/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_66/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_67/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_67/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_68/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_68/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv3d_44_inputPlaceholder*3
_output_shapes!
:?????????222*
dtype0*(
shape:?????????222
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv3d_44_inputconv3d_44/kernelconv3d_44/biasconv3d_45/kernelconv3d_45/bias&batch_normalization_44/moving_variancebatch_normalization_44/gamma"batch_normalization_44/moving_meanbatch_normalization_44/betaconv3d_46/kernelconv3d_46/biasconv3d_47/kernelconv3d_47/bias&batch_normalization_45/moving_variancebatch_normalization_45/gamma"batch_normalization_45/moving_meanbatch_normalization_45/betadense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_31652
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv3d_44/kernel/Read/ReadVariableOp"conv3d_44/bias/Read/ReadVariableOp$conv3d_45/kernel/Read/ReadVariableOp"conv3d_45/bias/Read/ReadVariableOp0batch_normalization_44/gamma/Read/ReadVariableOp/batch_normalization_44/beta/Read/ReadVariableOp6batch_normalization_44/moving_mean/Read/ReadVariableOp:batch_normalization_44/moving_variance/Read/ReadVariableOp$conv3d_46/kernel/Read/ReadVariableOp"conv3d_46/bias/Read/ReadVariableOp$conv3d_47/kernel/Read/ReadVariableOp"conv3d_47/bias/Read/ReadVariableOp0batch_normalization_45/gamma/Read/ReadVariableOp/batch_normalization_45/beta/Read/ReadVariableOp6batch_normalization_45/moving_mean/Read/ReadVariableOp:batch_normalization_45/moving_variance/Read/ReadVariableOp#dense_66/kernel/Read/ReadVariableOp!dense_66/bias/Read/ReadVariableOp#dense_67/kernel/Read/ReadVariableOp!dense_67/bias/Read/ReadVariableOp#dense_68/kernel/Read/ReadVariableOp!dense_68/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv3d_44/kernel/m/Read/ReadVariableOp)Adam/conv3d_44/bias/m/Read/ReadVariableOp+Adam/conv3d_45/kernel/m/Read/ReadVariableOp)Adam/conv3d_45/bias/m/Read/ReadVariableOp7Adam/batch_normalization_44/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_44/beta/m/Read/ReadVariableOp+Adam/conv3d_46/kernel/m/Read/ReadVariableOp)Adam/conv3d_46/bias/m/Read/ReadVariableOp+Adam/conv3d_47/kernel/m/Read/ReadVariableOp)Adam/conv3d_47/bias/m/Read/ReadVariableOp7Adam/batch_normalization_45/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_45/beta/m/Read/ReadVariableOp*Adam/dense_66/kernel/m/Read/ReadVariableOp(Adam/dense_66/bias/m/Read/ReadVariableOp*Adam/dense_67/kernel/m/Read/ReadVariableOp(Adam/dense_67/bias/m/Read/ReadVariableOp*Adam/dense_68/kernel/m/Read/ReadVariableOp(Adam/dense_68/bias/m/Read/ReadVariableOp+Adam/conv3d_44/kernel/v/Read/ReadVariableOp)Adam/conv3d_44/bias/v/Read/ReadVariableOp+Adam/conv3d_45/kernel/v/Read/ReadVariableOp)Adam/conv3d_45/bias/v/Read/ReadVariableOp7Adam/batch_normalization_44/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_44/beta/v/Read/ReadVariableOp+Adam/conv3d_46/kernel/v/Read/ReadVariableOp)Adam/conv3d_46/bias/v/Read/ReadVariableOp+Adam/conv3d_47/kernel/v/Read/ReadVariableOp)Adam/conv3d_47/bias/v/Read/ReadVariableOp7Adam/batch_normalization_45/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_45/beta/v/Read/ReadVariableOp*Adam/dense_66/kernel/v/Read/ReadVariableOp(Adam/dense_66/bias/v/Read/ReadVariableOp*Adam/dense_67/kernel/v/Read/ReadVariableOp(Adam/dense_67/bias/v/Read/ReadVariableOp*Adam/dense_68/kernel/v/Read/ReadVariableOp(Adam/dense_68/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_32651
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_44/kernelconv3d_44/biasconv3d_45/kernelconv3d_45/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv3d_46/kernelconv3d_46/biasconv3d_47/kernelconv3d_47/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_variancedense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv3d_44/kernel/mAdam/conv3d_44/bias/mAdam/conv3d_45/kernel/mAdam/conv3d_45/bias/m#Adam/batch_normalization_44/gamma/m"Adam/batch_normalization_44/beta/mAdam/conv3d_46/kernel/mAdam/conv3d_46/bias/mAdam/conv3d_47/kernel/mAdam/conv3d_47/bias/m#Adam/batch_normalization_45/gamma/m"Adam/batch_normalization_45/beta/mAdam/dense_66/kernel/mAdam/dense_66/bias/mAdam/dense_67/kernel/mAdam/dense_67/bias/mAdam/dense_68/kernel/mAdam/dense_68/bias/mAdam/conv3d_44/kernel/vAdam/conv3d_44/bias/vAdam/conv3d_45/kernel/vAdam/conv3d_45/bias/v#Adam/batch_normalization_44/gamma/v"Adam/batch_normalization_44/beta/vAdam/conv3d_46/kernel/vAdam/conv3d_46/bias/vAdam/conv3d_47/kernel/vAdam/conv3d_47/bias/v#Adam/batch_normalization_45/gamma/v"Adam/batch_normalization_45/beta/vAdam/dense_66/kernel/vAdam/dense_66/bias/vAdam/dense_67/kernel/vAdam/dense_67/bias/vAdam/dense_68/kernel/vAdam/dense_68/bias/v*O
TinH
F2D*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_32862??
?
L
0__inference_max_pooling3d_44_layer_call_fn_30494

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_44_layer_call_and_return_conditional_losses_304882
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv3d_45_layer_call_and_return_conditional_losses_30865

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...2	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????...2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????...2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????000: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????000
 
_user_specified_nameinputs
?
?
-__inference_sequential_22_layer_call_fn_31475
conv3d_44_input%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
?-?

unknown_16:	?

unknown_17:	?@

unknown_18:@

unknown_19:@

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_44_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_22_layer_call_and_return_conditional_losses_313792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_44_input
?

?
C__inference_dense_68_layer_call_and_return_conditional_losses_32418

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_conv3d_45_layer_call_and_return_conditional_losses_31987

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...2	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????...2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????...2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????000: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????000
 
_user_specified_nameinputs
ȇ
?
__inference__traced_save_32651
file_prefix/
+savev2_conv3d_44_kernel_read_readvariableop-
)savev2_conv3d_44_bias_read_readvariableop/
+savev2_conv3d_45_kernel_read_readvariableop-
)savev2_conv3d_45_bias_read_readvariableop;
7savev2_batch_normalization_44_gamma_read_readvariableop:
6savev2_batch_normalization_44_beta_read_readvariableopA
=savev2_batch_normalization_44_moving_mean_read_readvariableopE
Asavev2_batch_normalization_44_moving_variance_read_readvariableop/
+savev2_conv3d_46_kernel_read_readvariableop-
)savev2_conv3d_46_bias_read_readvariableop/
+savev2_conv3d_47_kernel_read_readvariableop-
)savev2_conv3d_47_bias_read_readvariableop;
7savev2_batch_normalization_45_gamma_read_readvariableop:
6savev2_batch_normalization_45_beta_read_readvariableopA
=savev2_batch_normalization_45_moving_mean_read_readvariableopE
Asavev2_batch_normalization_45_moving_variance_read_readvariableop.
*savev2_dense_66_kernel_read_readvariableop,
(savev2_dense_66_bias_read_readvariableop.
*savev2_dense_67_kernel_read_readvariableop,
(savev2_dense_67_bias_read_readvariableop.
*savev2_dense_68_kernel_read_readvariableop,
(savev2_dense_68_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv3d_44_kernel_m_read_readvariableop4
0savev2_adam_conv3d_44_bias_m_read_readvariableop6
2savev2_adam_conv3d_45_kernel_m_read_readvariableop4
0savev2_adam_conv3d_45_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_44_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_44_beta_m_read_readvariableop6
2savev2_adam_conv3d_46_kernel_m_read_readvariableop4
0savev2_adam_conv3d_46_bias_m_read_readvariableop6
2savev2_adam_conv3d_47_kernel_m_read_readvariableop4
0savev2_adam_conv3d_47_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_45_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_45_beta_m_read_readvariableop5
1savev2_adam_dense_66_kernel_m_read_readvariableop3
/savev2_adam_dense_66_bias_m_read_readvariableop5
1savev2_adam_dense_67_kernel_m_read_readvariableop3
/savev2_adam_dense_67_bias_m_read_readvariableop5
1savev2_adam_dense_68_kernel_m_read_readvariableop3
/savev2_adam_dense_68_bias_m_read_readvariableop6
2savev2_adam_conv3d_44_kernel_v_read_readvariableop4
0savev2_adam_conv3d_44_bias_v_read_readvariableop6
2savev2_adam_conv3d_45_kernel_v_read_readvariableop4
0savev2_adam_conv3d_45_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_44_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_44_beta_v_read_readvariableop6
2savev2_adam_conv3d_46_kernel_v_read_readvariableop4
0savev2_adam_conv3d_46_bias_v_read_readvariableop6
2savev2_adam_conv3d_47_kernel_v_read_readvariableop4
0savev2_adam_conv3d_47_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_45_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_45_beta_v_read_readvariableop5
1savev2_adam_dense_66_kernel_v_read_readvariableop3
/savev2_adam_dense_66_bias_v_read_readvariableop5
1savev2_adam_dense_67_kernel_v_read_readvariableop3
/savev2_adam_dense_67_bias_v_read_readvariableop5
1savev2_adam_dense_68_kernel_v_read_readvariableop3
/savev2_adam_dense_68_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?$
value?$B?$DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv3d_44_kernel_read_readvariableop)savev2_conv3d_44_bias_read_readvariableop+savev2_conv3d_45_kernel_read_readvariableop)savev2_conv3d_45_bias_read_readvariableop7savev2_batch_normalization_44_gamma_read_readvariableop6savev2_batch_normalization_44_beta_read_readvariableop=savev2_batch_normalization_44_moving_mean_read_readvariableopAsavev2_batch_normalization_44_moving_variance_read_readvariableop+savev2_conv3d_46_kernel_read_readvariableop)savev2_conv3d_46_bias_read_readvariableop+savev2_conv3d_47_kernel_read_readvariableop)savev2_conv3d_47_bias_read_readvariableop7savev2_batch_normalization_45_gamma_read_readvariableop6savev2_batch_normalization_45_beta_read_readvariableop=savev2_batch_normalization_45_moving_mean_read_readvariableopAsavev2_batch_normalization_45_moving_variance_read_readvariableop*savev2_dense_66_kernel_read_readvariableop(savev2_dense_66_bias_read_readvariableop*savev2_dense_67_kernel_read_readvariableop(savev2_dense_67_bias_read_readvariableop*savev2_dense_68_kernel_read_readvariableop(savev2_dense_68_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv3d_44_kernel_m_read_readvariableop0savev2_adam_conv3d_44_bias_m_read_readvariableop2savev2_adam_conv3d_45_kernel_m_read_readvariableop0savev2_adam_conv3d_45_bias_m_read_readvariableop>savev2_adam_batch_normalization_44_gamma_m_read_readvariableop=savev2_adam_batch_normalization_44_beta_m_read_readvariableop2savev2_adam_conv3d_46_kernel_m_read_readvariableop0savev2_adam_conv3d_46_bias_m_read_readvariableop2savev2_adam_conv3d_47_kernel_m_read_readvariableop0savev2_adam_conv3d_47_bias_m_read_readvariableop>savev2_adam_batch_normalization_45_gamma_m_read_readvariableop=savev2_adam_batch_normalization_45_beta_m_read_readvariableop1savev2_adam_dense_66_kernel_m_read_readvariableop/savev2_adam_dense_66_bias_m_read_readvariableop1savev2_adam_dense_67_kernel_m_read_readvariableop/savev2_adam_dense_67_bias_m_read_readvariableop1savev2_adam_dense_68_kernel_m_read_readvariableop/savev2_adam_dense_68_bias_m_read_readvariableop2savev2_adam_conv3d_44_kernel_v_read_readvariableop0savev2_adam_conv3d_44_bias_v_read_readvariableop2savev2_adam_conv3d_45_kernel_v_read_readvariableop0savev2_adam_conv3d_45_bias_v_read_readvariableop>savev2_adam_batch_normalization_44_gamma_v_read_readvariableop=savev2_adam_batch_normalization_44_beta_v_read_readvariableop2savev2_adam_conv3d_46_kernel_v_read_readvariableop0savev2_adam_conv3d_46_bias_v_read_readvariableop2savev2_adam_conv3d_47_kernel_v_read_readvariableop0savev2_adam_conv3d_47_bias_v_read_readvariableop>savev2_adam_batch_normalization_45_gamma_v_read_readvariableop=savev2_adam_batch_normalization_45_beta_v_read_readvariableop1savev2_adam_dense_66_kernel_v_read_readvariableop/savev2_adam_dense_66_bias_v_read_readvariableop1savev2_adam_dense_67_kernel_v_read_readvariableop/savev2_adam_dense_67_bias_v_read_readvariableop1savev2_adam_dense_68_kernel_v_read_readvariableop/savev2_adam_dense_68_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::::::::::::
?-?:?:	?@:@:@:: : : : : : : : : :::::::::::::
?-?:?:	?@:@:@::::::::::::::
?-?:?:	?@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0	,
*
_output_shapes
:: 


_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
?-?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0 ,
*
_output_shapes
:: !

_output_shapes
::0",
*
_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
::0&,
*
_output_shapes
:: '

_output_shapes
::0(,
*
_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
::&,"
 
_output_shapes
:
?-?:!-

_output_shapes	
:?:%.!

_output_shapes
:	?@: /

_output_shapes
:@:$0 

_output_shapes

:@: 1

_output_shapes
::02,
*
_output_shapes
:: 3

_output_shapes
::04,
*
_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
::08,
*
_output_shapes
:: 9

_output_shapes
::0:,
*
_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::&>"
 
_output_shapes
:
?-?:!?

_output_shapes	
:?:%@!

_output_shapes
:	?@: A

_output_shapes
:@:$B 

_output_shapes

:@: C

_output_shapes
::D

_output_shapes
: 
?
?
)__inference_conv3d_45_layer_call_fn_31996

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????...*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_45_layer_call_and_return_conditional_losses_308652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????...2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????000: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????000
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_30518

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_67_layer_call_and_return_conditional_losses_32398

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_45_layer_call_fn_32343

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_309552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????			2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????			: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32216

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_44_layer_call_fn_32156

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_312372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_22_layer_call_fn_32367

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_309712
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????-2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????			:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs
?
a
E__inference_flatten_22_layer_call_and_return_conditional_losses_30971

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????-2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????-2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????			:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs
?
a
E__inference_flatten_22_layer_call_and_return_conditional_losses_32362

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????-2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????-2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????			:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs
Ы
?
 __inference__wrapped_model_30482
conv3d_44_inputT
6sequential_22_conv3d_44_conv3d_readvariableop_resource:E
7sequential_22_conv3d_44_biasadd_readvariableop_resource:T
6sequential_22_conv3d_45_conv3d_readvariableop_resource:E
7sequential_22_conv3d_45_biasadd_readvariableop_resource:T
Fsequential_22_batch_normalization_44_batchnorm_readvariableop_resource:X
Jsequential_22_batch_normalization_44_batchnorm_mul_readvariableop_resource:V
Hsequential_22_batch_normalization_44_batchnorm_readvariableop_1_resource:V
Hsequential_22_batch_normalization_44_batchnorm_readvariableop_2_resource:T
6sequential_22_conv3d_46_conv3d_readvariableop_resource:E
7sequential_22_conv3d_46_biasadd_readvariableop_resource:T
6sequential_22_conv3d_47_conv3d_readvariableop_resource:E
7sequential_22_conv3d_47_biasadd_readvariableop_resource:T
Fsequential_22_batch_normalization_45_batchnorm_readvariableop_resource:X
Jsequential_22_batch_normalization_45_batchnorm_mul_readvariableop_resource:V
Hsequential_22_batch_normalization_45_batchnorm_readvariableop_1_resource:V
Hsequential_22_batch_normalization_45_batchnorm_readvariableop_2_resource:I
5sequential_22_dense_66_matmul_readvariableop_resource:
?-?E
6sequential_22_dense_66_biasadd_readvariableop_resource:	?H
5sequential_22_dense_67_matmul_readvariableop_resource:	?@D
6sequential_22_dense_67_biasadd_readvariableop_resource:@G
5sequential_22_dense_68_matmul_readvariableop_resource:@D
6sequential_22_dense_68_biasadd_readvariableop_resource:
identity??=sequential_22/batch_normalization_44/batchnorm/ReadVariableOp??sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_1??sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_2?Asequential_22/batch_normalization_44/batchnorm/mul/ReadVariableOp?=sequential_22/batch_normalization_45/batchnorm/ReadVariableOp??sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_1??sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_2?Asequential_22/batch_normalization_45/batchnorm/mul/ReadVariableOp?.sequential_22/conv3d_44/BiasAdd/ReadVariableOp?-sequential_22/conv3d_44/Conv3D/ReadVariableOp?.sequential_22/conv3d_45/BiasAdd/ReadVariableOp?-sequential_22/conv3d_45/Conv3D/ReadVariableOp?.sequential_22/conv3d_46/BiasAdd/ReadVariableOp?-sequential_22/conv3d_46/Conv3D/ReadVariableOp?.sequential_22/conv3d_47/BiasAdd/ReadVariableOp?-sequential_22/conv3d_47/Conv3D/ReadVariableOp?-sequential_22/dense_66/BiasAdd/ReadVariableOp?,sequential_22/dense_66/MatMul/ReadVariableOp?-sequential_22/dense_67/BiasAdd/ReadVariableOp?,sequential_22/dense_67/MatMul/ReadVariableOp?-sequential_22/dense_68/BiasAdd/ReadVariableOp?,sequential_22/dense_68/MatMul/ReadVariableOp?
-sequential_22/conv3d_44/Conv3D/ReadVariableOpReadVariableOp6sequential_22_conv3d_44_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02/
-sequential_22/conv3d_44/Conv3D/ReadVariableOp?
sequential_22/conv3d_44/Conv3DConv3Dconv3d_44_input5sequential_22/conv3d_44/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000*
paddingVALID*
strides	
2 
sequential_22/conv3d_44/Conv3D?
.sequential_22/conv3d_44/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv3d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_22/conv3d_44/BiasAdd/ReadVariableOp?
sequential_22/conv3d_44/BiasAddBiasAdd'sequential_22/conv3d_44/Conv3D:output:06sequential_22/conv3d_44/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????0002!
sequential_22/conv3d_44/BiasAdd?
sequential_22/conv3d_44/ReluRelu(sequential_22/conv3d_44/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002
sequential_22/conv3d_44/Relu?
-sequential_22/conv3d_45/Conv3D/ReadVariableOpReadVariableOp6sequential_22_conv3d_45_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02/
-sequential_22/conv3d_45/Conv3D/ReadVariableOp?
sequential_22/conv3d_45/Conv3DConv3D*sequential_22/conv3d_44/Relu:activations:05sequential_22/conv3d_45/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...*
paddingVALID*
strides	
2 
sequential_22/conv3d_45/Conv3D?
.sequential_22/conv3d_45/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv3d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_22/conv3d_45/BiasAdd/ReadVariableOp?
sequential_22/conv3d_45/BiasAddBiasAdd'sequential_22/conv3d_45/Conv3D:output:06sequential_22/conv3d_45/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...2!
sequential_22/conv3d_45/BiasAdd?
sequential_22/conv3d_45/SigmoidSigmoid(sequential_22/conv3d_45/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????...2!
sequential_22/conv3d_45/Sigmoid?
(sequential_22/max_pooling3d_44/MaxPool3D	MaxPool3D#sequential_22/conv3d_45/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2*
(sequential_22/max_pooling3d_44/MaxPool3D?
=sequential_22/batch_normalization_44/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_44_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_22/batch_normalization_44/batchnorm/ReadVariableOp?
4sequential_22/batch_normalization_44/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:26
4sequential_22/batch_normalization_44/batchnorm/add/y?
2sequential_22/batch_normalization_44/batchnorm/addAddV2Esequential_22/batch_normalization_44/batchnorm/ReadVariableOp:value:0=sequential_22/batch_normalization_44/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_44/batchnorm/add?
4sequential_22/batch_normalization_44/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_44/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_44/batchnorm/Rsqrt?
Asequential_22/batch_normalization_44/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_44_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_22/batch_normalization_44/batchnorm/mul/ReadVariableOp?
2sequential_22/batch_normalization_44/batchnorm/mulMul8sequential_22/batch_normalization_44/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_44/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_44/batchnorm/mul?
4sequential_22/batch_normalization_44/batchnorm/mul_1Mul1sequential_22/max_pooling3d_44/MaxPool3D:output:06sequential_22/batch_normalization_44/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????26
4sequential_22/batch_normalization_44/batchnorm/mul_1?
?sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_22_batch_normalization_44_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_1?
4sequential_22/batch_normalization_44/batchnorm/mul_2MulGsequential_22/batch_normalization_44/batchnorm/ReadVariableOp_1:value:06sequential_22/batch_normalization_44/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_44/batchnorm/mul_2?
?sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_22_batch_normalization_44_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_2?
2sequential_22/batch_normalization_44/batchnorm/subSubGsequential_22/batch_normalization_44/batchnorm/ReadVariableOp_2:value:08sequential_22/batch_normalization_44/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_44/batchnorm/sub?
4sequential_22/batch_normalization_44/batchnorm/add_1AddV28sequential_22/batch_normalization_44/batchnorm/mul_1:z:06sequential_22/batch_normalization_44/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????26
4sequential_22/batch_normalization_44/batchnorm/add_1?
-sequential_22/conv3d_46/Conv3D/ReadVariableOpReadVariableOp6sequential_22_conv3d_46_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02/
-sequential_22/conv3d_46/Conv3D/ReadVariableOp?
sequential_22/conv3d_46/Conv3DConv3D8sequential_22/batch_normalization_44/batchnorm/add_1:z:05sequential_22/conv3d_46/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2 
sequential_22/conv3d_46/Conv3D?
.sequential_22/conv3d_46/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv3d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_22/conv3d_46/BiasAdd/ReadVariableOp?
sequential_22/conv3d_46/BiasAddBiasAdd'sequential_22/conv3d_46/Conv3D:output:06sequential_22/conv3d_46/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2!
sequential_22/conv3d_46/BiasAdd?
sequential_22/conv3d_46/ReluRelu(sequential_22/conv3d_46/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
sequential_22/conv3d_46/Relu?
-sequential_22/conv3d_47/Conv3D/ReadVariableOpReadVariableOp6sequential_22_conv3d_47_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02/
-sequential_22/conv3d_47/Conv3D/ReadVariableOp?
sequential_22/conv3d_47/Conv3DConv3D*sequential_22/conv3d_46/Relu:activations:05sequential_22/conv3d_47/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2 
sequential_22/conv3d_47/Conv3D?
.sequential_22/conv3d_47/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv3d_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_22/conv3d_47/BiasAdd/ReadVariableOp?
sequential_22/conv3d_47/BiasAddBiasAdd'sequential_22/conv3d_47/Conv3D:output:06sequential_22/conv3d_47/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2!
sequential_22/conv3d_47/BiasAdd?
sequential_22/conv3d_47/SigmoidSigmoid(sequential_22/conv3d_47/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2!
sequential_22/conv3d_47/Sigmoid?
(sequential_22/max_pooling3d_45/MaxPool3D	MaxPool3D#sequential_22/conv3d_47/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????			*
ksize	
*
paddingVALID*
strides	
2*
(sequential_22/max_pooling3d_45/MaxPool3D?
=sequential_22/batch_normalization_45/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_45_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_22/batch_normalization_45/batchnorm/ReadVariableOp?
4sequential_22/batch_normalization_45/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:26
4sequential_22/batch_normalization_45/batchnorm/add/y?
2sequential_22/batch_normalization_45/batchnorm/addAddV2Esequential_22/batch_normalization_45/batchnorm/ReadVariableOp:value:0=sequential_22/batch_normalization_45/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_45/batchnorm/add?
4sequential_22/batch_normalization_45/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_45/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_45/batchnorm/Rsqrt?
Asequential_22/batch_normalization_45/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_45_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_22/batch_normalization_45/batchnorm/mul/ReadVariableOp?
2sequential_22/batch_normalization_45/batchnorm/mulMul8sequential_22/batch_normalization_45/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_45/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_45/batchnorm/mul?
4sequential_22/batch_normalization_45/batchnorm/mul_1Mul1sequential_22/max_pooling3d_45/MaxPool3D:output:06sequential_22/batch_normalization_45/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????			26
4sequential_22/batch_normalization_45/batchnorm/mul_1?
?sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_22_batch_normalization_45_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_1?
4sequential_22/batch_normalization_45/batchnorm/mul_2MulGsequential_22/batch_normalization_45/batchnorm/ReadVariableOp_1:value:06sequential_22/batch_normalization_45/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_45/batchnorm/mul_2?
?sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_22_batch_normalization_45_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_2?
2sequential_22/batch_normalization_45/batchnorm/subSubGsequential_22/batch_normalization_45/batchnorm/ReadVariableOp_2:value:08sequential_22/batch_normalization_45/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_45/batchnorm/sub?
4sequential_22/batch_normalization_45/batchnorm/add_1AddV28sequential_22/batch_normalization_45/batchnorm/mul_1:z:06sequential_22/batch_normalization_45/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????			26
4sequential_22/batch_normalization_45/batchnorm/add_1?
sequential_22/flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2 
sequential_22/flatten_22/Const?
 sequential_22/flatten_22/ReshapeReshape8sequential_22/batch_normalization_45/batchnorm/add_1:z:0'sequential_22/flatten_22/Const:output:0*
T0*(
_output_shapes
:??????????-2"
 sequential_22/flatten_22/Reshape?
,sequential_22/dense_66/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_66_matmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02.
,sequential_22/dense_66/MatMul/ReadVariableOp?
sequential_22/dense_66/MatMulMatMul)sequential_22/flatten_22/Reshape:output:04sequential_22/dense_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_22/dense_66/MatMul?
-sequential_22/dense_66/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_66_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_22/dense_66/BiasAdd/ReadVariableOp?
sequential_22/dense_66/BiasAddBiasAdd'sequential_22/dense_66/MatMul:product:05sequential_22/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_22/dense_66/BiasAdd?
sequential_22/dense_66/SoftmaxSoftmax'sequential_22/dense_66/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_22/dense_66/Softmax?
,sequential_22/dense_67/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_67_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02.
,sequential_22/dense_67/MatMul/ReadVariableOp?
sequential_22/dense_67/MatMulMatMul(sequential_22/dense_66/Softmax:softmax:04sequential_22/dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_22/dense_67/MatMul?
-sequential_22/dense_67/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_67_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_22/dense_67/BiasAdd/ReadVariableOp?
sequential_22/dense_67/BiasAddBiasAdd'sequential_22/dense_67/MatMul:product:05sequential_22/dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_22/dense_67/BiasAdd?
sequential_22/dense_67/ReluRelu'sequential_22/dense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_22/dense_67/Relu?
,sequential_22/dense_68/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_68_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential_22/dense_68/MatMul/ReadVariableOp?
sequential_22/dense_68/MatMulMatMul)sequential_22/dense_67/Relu:activations:04sequential_22/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_22/dense_68/MatMul?
-sequential_22/dense_68/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_22/dense_68/BiasAdd/ReadVariableOp?
sequential_22/dense_68/BiasAddBiasAdd'sequential_22/dense_68/MatMul:product:05sequential_22/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_22/dense_68/BiasAdd?
sequential_22/dense_68/SigmoidSigmoid'sequential_22/dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_22/dense_68/Sigmoid?

IdentityIdentity"sequential_22/dense_68/Sigmoid:y:0>^sequential_22/batch_normalization_44/batchnorm/ReadVariableOp@^sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_1@^sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_2B^sequential_22/batch_normalization_44/batchnorm/mul/ReadVariableOp>^sequential_22/batch_normalization_45/batchnorm/ReadVariableOp@^sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_1@^sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_2B^sequential_22/batch_normalization_45/batchnorm/mul/ReadVariableOp/^sequential_22/conv3d_44/BiasAdd/ReadVariableOp.^sequential_22/conv3d_44/Conv3D/ReadVariableOp/^sequential_22/conv3d_45/BiasAdd/ReadVariableOp.^sequential_22/conv3d_45/Conv3D/ReadVariableOp/^sequential_22/conv3d_46/BiasAdd/ReadVariableOp.^sequential_22/conv3d_46/Conv3D/ReadVariableOp/^sequential_22/conv3d_47/BiasAdd/ReadVariableOp.^sequential_22/conv3d_47/Conv3D/ReadVariableOp.^sequential_22/dense_66/BiasAdd/ReadVariableOp-^sequential_22/dense_66/MatMul/ReadVariableOp.^sequential_22/dense_67/BiasAdd/ReadVariableOp-^sequential_22/dense_67/MatMul/ReadVariableOp.^sequential_22/dense_68/BiasAdd/ReadVariableOp-^sequential_22/dense_68/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 2~
=sequential_22/batch_normalization_44/batchnorm/ReadVariableOp=sequential_22/batch_normalization_44/batchnorm/ReadVariableOp2?
?sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_1?sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_12?
?sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_2?sequential_22/batch_normalization_44/batchnorm/ReadVariableOp_22?
Asequential_22/batch_normalization_44/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_44/batchnorm/mul/ReadVariableOp2~
=sequential_22/batch_normalization_45/batchnorm/ReadVariableOp=sequential_22/batch_normalization_45/batchnorm/ReadVariableOp2?
?sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_1?sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_12?
?sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_2?sequential_22/batch_normalization_45/batchnorm/ReadVariableOp_22?
Asequential_22/batch_normalization_45/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_45/batchnorm/mul/ReadVariableOp2`
.sequential_22/conv3d_44/BiasAdd/ReadVariableOp.sequential_22/conv3d_44/BiasAdd/ReadVariableOp2^
-sequential_22/conv3d_44/Conv3D/ReadVariableOp-sequential_22/conv3d_44/Conv3D/ReadVariableOp2`
.sequential_22/conv3d_45/BiasAdd/ReadVariableOp.sequential_22/conv3d_45/BiasAdd/ReadVariableOp2^
-sequential_22/conv3d_45/Conv3D/ReadVariableOp-sequential_22/conv3d_45/Conv3D/ReadVariableOp2`
.sequential_22/conv3d_46/BiasAdd/ReadVariableOp.sequential_22/conv3d_46/BiasAdd/ReadVariableOp2^
-sequential_22/conv3d_46/Conv3D/ReadVariableOp-sequential_22/conv3d_46/Conv3D/ReadVariableOp2`
.sequential_22/conv3d_47/BiasAdd/ReadVariableOp.sequential_22/conv3d_47/BiasAdd/ReadVariableOp2^
-sequential_22/conv3d_47/Conv3D/ReadVariableOp-sequential_22/conv3d_47/Conv3D/ReadVariableOp2^
-sequential_22/dense_66/BiasAdd/ReadVariableOp-sequential_22/dense_66/BiasAdd/ReadVariableOp2\
,sequential_22/dense_66/MatMul/ReadVariableOp,sequential_22/dense_66/MatMul/ReadVariableOp2^
-sequential_22/dense_67/BiasAdd/ReadVariableOp-sequential_22/dense_67/BiasAdd/ReadVariableOp2\
,sequential_22/dense_67/MatMul/ReadVariableOp,sequential_22/dense_67/MatMul/ReadVariableOp2^
-sequential_22/dense_68/BiasAdd/ReadVariableOp-sequential_22/dense_68/BiasAdd/ReadVariableOp2\
,sequential_22/dense_68/MatMul/ReadVariableOp,sequential_22/dense_68/MatMul/ReadVariableOp:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_44_input
?
?
(__inference_dense_67_layer_call_fn_32407

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_310012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling3d_45_layer_call_and_return_conditional_losses_30662

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_30578

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_31237

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv3d_47_layer_call_and_return_conditional_losses_30929

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_45_layer_call_fn_32356

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_311572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????			2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????			: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs
?

?
C__inference_dense_68_layer_call_and_return_conditional_losses_31018

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_conv3d_47_layer_call_and_return_conditional_losses_32187

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_66_layer_call_and_return_conditional_losses_32378

inputs2
matmul_readvariableop_resource:
?-?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_45_layer_call_fn_32317

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_306922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_22_layer_call_fn_31907

inputs%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
?-?

unknown_16:	?

unknown_17:	?@

unknown_18:@

unknown_19:@

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_22_layer_call_and_return_conditional_losses_310252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
D__inference_conv3d_44_layer_call_and_return_conditional_losses_30848

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????0002	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????0002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????222: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?

?
C__inference_dense_67_layer_call_and_return_conditional_losses_31001

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?@
?

H__inference_sequential_22_layer_call_and_return_conditional_losses_31535
conv3d_44_input-
conv3d_44_31478:
conv3d_44_31480:-
conv3d_45_31483:
conv3d_45_31485:*
batch_normalization_44_31489:*
batch_normalization_44_31491:*
batch_normalization_44_31493:*
batch_normalization_44_31495:-
conv3d_46_31498:
conv3d_46_31500:-
conv3d_47_31503:
conv3d_47_31505:*
batch_normalization_45_31509:*
batch_normalization_45_31511:*
batch_normalization_45_31513:*
batch_normalization_45_31515:"
dense_66_31519:
?-?
dense_66_31521:	?!
dense_67_31524:	?@
dense_67_31526:@ 
dense_68_31529:@
dense_68_31531:
identity??.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?!conv3d_44/StatefulPartitionedCall?!conv3d_45/StatefulPartitionedCall?!conv3d_46/StatefulPartitionedCall?!conv3d_47/StatefulPartitionedCall? dense_66/StatefulPartitionedCall? dense_67/StatefulPartitionedCall? dense_68/StatefulPartitionedCall?
!conv3d_44/StatefulPartitionedCallStatefulPartitionedCallconv3d_44_inputconv3d_44_31478conv3d_44_31480*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????000*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_44_layer_call_and_return_conditional_losses_308482#
!conv3d_44/StatefulPartitionedCall?
!conv3d_45/StatefulPartitionedCallStatefulPartitionedCall*conv3d_44/StatefulPartitionedCall:output:0conv3d_45_31483conv3d_45_31485*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????...*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_45_layer_call_and_return_conditional_losses_308652#
!conv3d_45/StatefulPartitionedCall?
 max_pooling3d_44/PartitionedCallPartitionedCall*conv3d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_44_layer_call_and_return_conditional_losses_304882"
 max_pooling3d_44/PartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_44/PartitionedCall:output:0batch_normalization_44_31489batch_normalization_44_31491batch_normalization_44_31493batch_normalization_44_31495*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_3089120
.batch_normalization_44/StatefulPartitionedCall?
!conv3d_46/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0conv3d_46_31498conv3d_46_31500*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_46_layer_call_and_return_conditional_losses_309122#
!conv3d_46/StatefulPartitionedCall?
!conv3d_47/StatefulPartitionedCallStatefulPartitionedCall*conv3d_46/StatefulPartitionedCall:output:0conv3d_47_31503conv3d_47_31505*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_47_layer_call_and_return_conditional_losses_309292#
!conv3d_47/StatefulPartitionedCall?
 max_pooling3d_45/PartitionedCallPartitionedCall*conv3d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_45_layer_call_and_return_conditional_losses_306622"
 max_pooling3d_45/PartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_45/PartitionedCall:output:0batch_normalization_45_31509batch_normalization_45_31511batch_normalization_45_31513batch_normalization_45_31515*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_3095520
.batch_normalization_45/StatefulPartitionedCall?
flatten_22/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_309712
flatten_22/PartitionedCall?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_66_31519dense_66_31521*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_309842"
 dense_66/StatefulPartitionedCall?
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_31524dense_67_31526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_310012"
 dense_67/StatefulPartitionedCall?
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_31529dense_68_31531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_68_layer_call_and_return_conditional_losses_310182"
 dense_68/StatefulPartitionedCall?
IdentityIdentity)dense_68/StatefulPartitionedCall:output:0/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall"^conv3d_44/StatefulPartitionedCall"^conv3d_45/StatefulPartitionedCall"^conv3d_46/StatefulPartitionedCall"^conv3d_47/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2F
!conv3d_44/StatefulPartitionedCall!conv3d_44/StatefulPartitionedCall2F
!conv3d_45/StatefulPartitionedCall!conv3d_45/StatefulPartitionedCall2F
!conv3d_46/StatefulPartitionedCall!conv3d_46/StatefulPartitionedCall2F
!conv3d_47/StatefulPartitionedCall!conv3d_47/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_44_input
?
?
-__inference_sequential_22_layer_call_fn_31956

inputs%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
?-?

unknown_16:	?

unknown_17:	?@

unknown_18:@

unknown_19:@

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_22_layer_call_and_return_conditional_losses_313792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
D__inference_conv3d_44_layer_call_and_return_conditional_losses_31967

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????0002	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????0002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????222: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32104

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv3d_46_layer_call_and_return_conditional_losses_30912

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_66_layer_call_fn_32387

inputs
unknown:
?-?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_309842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????-: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_44_layer_call_fn_32143

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_308912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_30891

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32250

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_30692

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_30752

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_30955

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????			2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????			2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????			2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????			: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32050

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?@
?

H__inference_sequential_22_layer_call_and_return_conditional_losses_31595
conv3d_44_input-
conv3d_44_31538:
conv3d_44_31540:-
conv3d_45_31543:
conv3d_45_31545:*
batch_normalization_44_31549:*
batch_normalization_44_31551:*
batch_normalization_44_31553:*
batch_normalization_44_31555:-
conv3d_46_31558:
conv3d_46_31560:-
conv3d_47_31563:
conv3d_47_31565:*
batch_normalization_45_31569:*
batch_normalization_45_31571:*
batch_normalization_45_31573:*
batch_normalization_45_31575:"
dense_66_31579:
?-?
dense_66_31581:	?!
dense_67_31584:	?@
dense_67_31586:@ 
dense_68_31589:@
dense_68_31591:
identity??.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?!conv3d_44/StatefulPartitionedCall?!conv3d_45/StatefulPartitionedCall?!conv3d_46/StatefulPartitionedCall?!conv3d_47/StatefulPartitionedCall? dense_66/StatefulPartitionedCall? dense_67/StatefulPartitionedCall? dense_68/StatefulPartitionedCall?
!conv3d_44/StatefulPartitionedCallStatefulPartitionedCallconv3d_44_inputconv3d_44_31538conv3d_44_31540*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????000*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_44_layer_call_and_return_conditional_losses_308482#
!conv3d_44/StatefulPartitionedCall?
!conv3d_45/StatefulPartitionedCallStatefulPartitionedCall*conv3d_44/StatefulPartitionedCall:output:0conv3d_45_31543conv3d_45_31545*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????...*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_45_layer_call_and_return_conditional_losses_308652#
!conv3d_45/StatefulPartitionedCall?
 max_pooling3d_44/PartitionedCallPartitionedCall*conv3d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_44_layer_call_and_return_conditional_losses_304882"
 max_pooling3d_44/PartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_44/PartitionedCall:output:0batch_normalization_44_31549batch_normalization_44_31551batch_normalization_44_31553batch_normalization_44_31555*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_3123720
.batch_normalization_44/StatefulPartitionedCall?
!conv3d_46/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0conv3d_46_31558conv3d_46_31560*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_46_layer_call_and_return_conditional_losses_309122#
!conv3d_46/StatefulPartitionedCall?
!conv3d_47/StatefulPartitionedCallStatefulPartitionedCall*conv3d_46/StatefulPartitionedCall:output:0conv3d_47_31563conv3d_47_31565*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_47_layer_call_and_return_conditional_losses_309292#
!conv3d_47/StatefulPartitionedCall?
 max_pooling3d_45/PartitionedCallPartitionedCall*conv3d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_45_layer_call_and_return_conditional_losses_306622"
 max_pooling3d_45/PartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_45/PartitionedCall:output:0batch_normalization_45_31569batch_normalization_45_31571batch_normalization_45_31573batch_normalization_45_31575*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_3115720
.batch_normalization_45/StatefulPartitionedCall?
flatten_22/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_309712
flatten_22/PartitionedCall?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_66_31579dense_66_31581*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_309842"
 dense_66/StatefulPartitionedCall?
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_31584dense_67_31586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_310012"
 dense_67/StatefulPartitionedCall?
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_31589dense_68_31591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_68_layer_call_and_return_conditional_losses_310182"
 dense_68/StatefulPartitionedCall?
IdentityIdentity)dense_68/StatefulPartitionedCall:output:0/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall"^conv3d_44/StatefulPartitionedCall"^conv3d_45/StatefulPartitionedCall"^conv3d_46/StatefulPartitionedCall"^conv3d_47/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2F
!conv3d_44/StatefulPartitionedCall!conv3d_44/StatefulPartitionedCall2F
!conv3d_45/StatefulPartitionedCall!conv3d_45/StatefulPartitionedCall2F
!conv3d_46/StatefulPartitionedCall!conv3d_46/StatefulPartitionedCall2F
!conv3d_47/StatefulPartitionedCall!conv3d_47/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_44_input
?
?
D__inference_conv3d_46_layer_call_and_return_conditional_losses_32167

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv3d_44_layer_call_fn_31976

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????000*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_44_layer_call_and_return_conditional_losses_308482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????0002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????222: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?@
?

H__inference_sequential_22_layer_call_and_return_conditional_losses_31379

inputs-
conv3d_44_31322:
conv3d_44_31324:-
conv3d_45_31327:
conv3d_45_31329:*
batch_normalization_44_31333:*
batch_normalization_44_31335:*
batch_normalization_44_31337:*
batch_normalization_44_31339:-
conv3d_46_31342:
conv3d_46_31344:-
conv3d_47_31347:
conv3d_47_31349:*
batch_normalization_45_31353:*
batch_normalization_45_31355:*
batch_normalization_45_31357:*
batch_normalization_45_31359:"
dense_66_31363:
?-?
dense_66_31365:	?!
dense_67_31368:	?@
dense_67_31370:@ 
dense_68_31373:@
dense_68_31375:
identity??.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?!conv3d_44/StatefulPartitionedCall?!conv3d_45/StatefulPartitionedCall?!conv3d_46/StatefulPartitionedCall?!conv3d_47/StatefulPartitionedCall? dense_66/StatefulPartitionedCall? dense_67/StatefulPartitionedCall? dense_68/StatefulPartitionedCall?
!conv3d_44/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_44_31322conv3d_44_31324*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????000*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_44_layer_call_and_return_conditional_losses_308482#
!conv3d_44/StatefulPartitionedCall?
!conv3d_45/StatefulPartitionedCallStatefulPartitionedCall*conv3d_44/StatefulPartitionedCall:output:0conv3d_45_31327conv3d_45_31329*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????...*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_45_layer_call_and_return_conditional_losses_308652#
!conv3d_45/StatefulPartitionedCall?
 max_pooling3d_44/PartitionedCallPartitionedCall*conv3d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_44_layer_call_and_return_conditional_losses_304882"
 max_pooling3d_44/PartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_44/PartitionedCall:output:0batch_normalization_44_31333batch_normalization_44_31335batch_normalization_44_31337batch_normalization_44_31339*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_3123720
.batch_normalization_44/StatefulPartitionedCall?
!conv3d_46/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0conv3d_46_31342conv3d_46_31344*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_46_layer_call_and_return_conditional_losses_309122#
!conv3d_46/StatefulPartitionedCall?
!conv3d_47/StatefulPartitionedCallStatefulPartitionedCall*conv3d_46/StatefulPartitionedCall:output:0conv3d_47_31347conv3d_47_31349*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_47_layer_call_and_return_conditional_losses_309292#
!conv3d_47/StatefulPartitionedCall?
 max_pooling3d_45/PartitionedCallPartitionedCall*conv3d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_45_layer_call_and_return_conditional_losses_306622"
 max_pooling3d_45/PartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_45/PartitionedCall:output:0batch_normalization_45_31353batch_normalization_45_31355batch_normalization_45_31357batch_normalization_45_31359*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_3115720
.batch_normalization_45/StatefulPartitionedCall?
flatten_22/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_309712
flatten_22/PartitionedCall?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_66_31363dense_66_31365*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_309842"
 dense_66/StatefulPartitionedCall?
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_31368dense_67_31370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_310012"
 dense_67/StatefulPartitionedCall?
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_31373dense_68_31375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_68_layer_call_and_return_conditional_losses_310182"
 dense_68/StatefulPartitionedCall?
IdentityIdentity)dense_68/StatefulPartitionedCall:output:0/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall"^conv3d_44/StatefulPartitionedCall"^conv3d_45/StatefulPartitionedCall"^conv3d_46/StatefulPartitionedCall"^conv3d_47/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2F
!conv3d_44/StatefulPartitionedCall!conv3d_44/StatefulPartitionedCall2F
!conv3d_45/StatefulPartitionedCall!conv3d_45/StatefulPartitionedCall2F
!conv3d_46/StatefulPartitionedCall!conv3d_46/StatefulPartitionedCall2F
!conv3d_47/StatefulPartitionedCall!conv3d_47/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
-__inference_sequential_22_layer_call_fn_31072
conv3d_44_input%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
?-?

unknown_16:	?

unknown_17:	?@

unknown_18:@

unknown_19:@

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_44_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_22_layer_call_and_return_conditional_losses_310252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_44_input
?	
?
6__inference_batch_normalization_44_layer_call_fn_32130

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_305782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?@
?

H__inference_sequential_22_layer_call_and_return_conditional_losses_31025

inputs-
conv3d_44_30849:
conv3d_44_30851:-
conv3d_45_30866:
conv3d_45_30868:*
batch_normalization_44_30892:*
batch_normalization_44_30894:*
batch_normalization_44_30896:*
batch_normalization_44_30898:-
conv3d_46_30913:
conv3d_46_30915:-
conv3d_47_30930:
conv3d_47_30932:*
batch_normalization_45_30956:*
batch_normalization_45_30958:*
batch_normalization_45_30960:*
batch_normalization_45_30962:"
dense_66_30985:
?-?
dense_66_30987:	?!
dense_67_31002:	?@
dense_67_31004:@ 
dense_68_31019:@
dense_68_31021:
identity??.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?!conv3d_44/StatefulPartitionedCall?!conv3d_45/StatefulPartitionedCall?!conv3d_46/StatefulPartitionedCall?!conv3d_47/StatefulPartitionedCall? dense_66/StatefulPartitionedCall? dense_67/StatefulPartitionedCall? dense_68/StatefulPartitionedCall?
!conv3d_44/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_44_30849conv3d_44_30851*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????000*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_44_layer_call_and_return_conditional_losses_308482#
!conv3d_44/StatefulPartitionedCall?
!conv3d_45/StatefulPartitionedCallStatefulPartitionedCall*conv3d_44/StatefulPartitionedCall:output:0conv3d_45_30866conv3d_45_30868*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????...*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_45_layer_call_and_return_conditional_losses_308652#
!conv3d_45/StatefulPartitionedCall?
 max_pooling3d_44/PartitionedCallPartitionedCall*conv3d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_44_layer_call_and_return_conditional_losses_304882"
 max_pooling3d_44/PartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_44/PartitionedCall:output:0batch_normalization_44_30892batch_normalization_44_30894batch_normalization_44_30896batch_normalization_44_30898*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_3089120
.batch_normalization_44/StatefulPartitionedCall?
!conv3d_46/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0conv3d_46_30913conv3d_46_30915*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_46_layer_call_and_return_conditional_losses_309122#
!conv3d_46/StatefulPartitionedCall?
!conv3d_47/StatefulPartitionedCallStatefulPartitionedCall*conv3d_46/StatefulPartitionedCall:output:0conv3d_47_30930conv3d_47_30932*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_47_layer_call_and_return_conditional_losses_309292#
!conv3d_47/StatefulPartitionedCall?
 max_pooling3d_45/PartitionedCallPartitionedCall*conv3d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_45_layer_call_and_return_conditional_losses_306622"
 max_pooling3d_45/PartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_45/PartitionedCall:output:0batch_normalization_45_30956batch_normalization_45_30958batch_normalization_45_30960batch_normalization_45_30962*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????			*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_3095520
.batch_normalization_45/StatefulPartitionedCall?
flatten_22/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_309712
flatten_22/PartitionedCall?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_66_30985dense_66_30987*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_309842"
 dense_66/StatefulPartitionedCall?
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_31002dense_67_31004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_310012"
 dense_67/StatefulPartitionedCall?
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_31019dense_68_31021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_68_layer_call_and_return_conditional_losses_310182"
 dense_68/StatefulPartitionedCall?
IdentityIdentity)dense_68/StatefulPartitionedCall:output:0/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall"^conv3d_44/StatefulPartitionedCall"^conv3d_45/StatefulPartitionedCall"^conv3d_46/StatefulPartitionedCall"^conv3d_47/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2F
!conv3d_44/StatefulPartitionedCall!conv3d_44/StatefulPartitionedCall2F
!conv3d_45/StatefulPartitionedCall!conv3d_45/StatefulPartitionedCall2F
!conv3d_46/StatefulPartitionedCall!conv3d_46/StatefulPartitionedCall2F
!conv3d_47/StatefulPartitionedCall!conv3d_47/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
)__inference_conv3d_46_layer_call_fn_32176

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_46_layer_call_and_return_conditional_losses_309122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_45_layer_call_fn_32330

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_307522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_31157

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????			2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????			2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????			2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????			2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????			: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs
??
?
H__inference_sequential_22_layer_call_and_return_conditional_losses_31858

inputsF
(conv3d_44_conv3d_readvariableop_resource:7
)conv3d_44_biasadd_readvariableop_resource:F
(conv3d_45_conv3d_readvariableop_resource:7
)conv3d_45_biasadd_readvariableop_resource:L
>batch_normalization_44_assignmovingavg_readvariableop_resource:N
@batch_normalization_44_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_44_batchnorm_mul_readvariableop_resource:F
8batch_normalization_44_batchnorm_readvariableop_resource:F
(conv3d_46_conv3d_readvariableop_resource:7
)conv3d_46_biasadd_readvariableop_resource:F
(conv3d_47_conv3d_readvariableop_resource:7
)conv3d_47_biasadd_readvariableop_resource:L
>batch_normalization_45_assignmovingavg_readvariableop_resource:N
@batch_normalization_45_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_45_batchnorm_mul_readvariableop_resource:F
8batch_normalization_45_batchnorm_readvariableop_resource:;
'dense_66_matmul_readvariableop_resource:
?-?7
(dense_66_biasadd_readvariableop_resource:	?:
'dense_67_matmul_readvariableop_resource:	?@6
(dense_67_biasadd_readvariableop_resource:@9
'dense_68_matmul_readvariableop_resource:@6
(dense_68_biasadd_readvariableop_resource:
identity??&batch_normalization_44/AssignMovingAvg?5batch_normalization_44/AssignMovingAvg/ReadVariableOp?(batch_normalization_44/AssignMovingAvg_1?7batch_normalization_44/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_44/batchnorm/ReadVariableOp?3batch_normalization_44/batchnorm/mul/ReadVariableOp?&batch_normalization_45/AssignMovingAvg?5batch_normalization_45/AssignMovingAvg/ReadVariableOp?(batch_normalization_45/AssignMovingAvg_1?7batch_normalization_45/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_45/batchnorm/ReadVariableOp?3batch_normalization_45/batchnorm/mul/ReadVariableOp? conv3d_44/BiasAdd/ReadVariableOp?conv3d_44/Conv3D/ReadVariableOp? conv3d_45/BiasAdd/ReadVariableOp?conv3d_45/Conv3D/ReadVariableOp? conv3d_46/BiasAdd/ReadVariableOp?conv3d_46/Conv3D/ReadVariableOp? conv3d_47/BiasAdd/ReadVariableOp?conv3d_47/Conv3D/ReadVariableOp?dense_66/BiasAdd/ReadVariableOp?dense_66/MatMul/ReadVariableOp?dense_67/BiasAdd/ReadVariableOp?dense_67/MatMul/ReadVariableOp?dense_68/BiasAdd/ReadVariableOp?dense_68/MatMul/ReadVariableOp?
conv3d_44/Conv3D/ReadVariableOpReadVariableOp(conv3d_44_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_44/Conv3D/ReadVariableOp?
conv3d_44/Conv3DConv3Dinputs'conv3d_44/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000*
paddingVALID*
strides	
2
conv3d_44/Conv3D?
 conv3d_44/BiasAdd/ReadVariableOpReadVariableOp)conv3d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_44/BiasAdd/ReadVariableOp?
conv3d_44/BiasAddBiasAddconv3d_44/Conv3D:output:0(conv3d_44/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????0002
conv3d_44/BiasAdd?
conv3d_44/ReluReluconv3d_44/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002
conv3d_44/Relu?
conv3d_45/Conv3D/ReadVariableOpReadVariableOp(conv3d_45_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_45/Conv3D/ReadVariableOp?
conv3d_45/Conv3DConv3Dconv3d_44/Relu:activations:0'conv3d_45/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...*
paddingVALID*
strides	
2
conv3d_45/Conv3D?
 conv3d_45/BiasAdd/ReadVariableOpReadVariableOp)conv3d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_45/BiasAdd/ReadVariableOp?
conv3d_45/BiasAddBiasAddconv3d_45/Conv3D:output:0(conv3d_45/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...2
conv3d_45/BiasAdd?
conv3d_45/SigmoidSigmoidconv3d_45/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????...2
conv3d_45/Sigmoid?
max_pooling3d_44/MaxPool3D	MaxPool3Dconv3d_45/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_44/MaxPool3D?
5batch_normalization_44/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_44/moments/mean/reduction_indices?
#batch_normalization_44/moments/meanMean#max_pooling3d_44/MaxPool3D:output:0>batch_normalization_44/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2%
#batch_normalization_44/moments/mean?
+batch_normalization_44/moments/StopGradientStopGradient,batch_normalization_44/moments/mean:output:0*
T0**
_output_shapes
:2-
+batch_normalization_44/moments/StopGradient?
0batch_normalization_44/moments/SquaredDifferenceSquaredDifference#max_pooling3d_44/MaxPool3D:output:04batch_normalization_44/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????22
0batch_normalization_44/moments/SquaredDifference?
9batch_normalization_44/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_44/moments/variance/reduction_indices?
'batch_normalization_44/moments/varianceMean4batch_normalization_44/moments/SquaredDifference:z:0Bbatch_normalization_44/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2)
'batch_normalization_44/moments/variance?
&batch_normalization_44/moments/SqueezeSqueeze,batch_normalization_44/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_44/moments/Squeeze?
(batch_normalization_44/moments/Squeeze_1Squeeze0batch_normalization_44/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_44/moments/Squeeze_1?
,batch_normalization_44/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_44/AssignMovingAvg/decay?
5batch_normalization_44/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_44_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_44/AssignMovingAvg/ReadVariableOp?
*batch_normalization_44/AssignMovingAvg/subSub=batch_normalization_44/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_44/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_44/AssignMovingAvg/sub?
*batch_normalization_44/AssignMovingAvg/mulMul.batch_normalization_44/AssignMovingAvg/sub:z:05batch_normalization_44/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2,
*batch_normalization_44/AssignMovingAvg/mul?
&batch_normalization_44/AssignMovingAvgAssignSubVariableOp>batch_normalization_44_assignmovingavg_readvariableop_resource.batch_normalization_44/AssignMovingAvg/mul:z:06^batch_normalization_44/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_44/AssignMovingAvg?
.batch_normalization_44/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_44/AssignMovingAvg_1/decay?
7batch_normalization_44/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_44_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_44/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_44/AssignMovingAvg_1/subSub?batch_normalization_44/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_44/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_44/AssignMovingAvg_1/sub?
,batch_normalization_44/AssignMovingAvg_1/mulMul0batch_normalization_44/AssignMovingAvg_1/sub:z:07batch_normalization_44/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2.
,batch_normalization_44/AssignMovingAvg_1/mul?
(batch_normalization_44/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_44_assignmovingavg_1_readvariableop_resource0batch_normalization_44/AssignMovingAvg_1/mul:z:08^batch_normalization_44/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_44/AssignMovingAvg_1?
&batch_normalization_44/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_44/batchnorm/add/y?
$batch_normalization_44/batchnorm/addAddV21batch_normalization_44/moments/Squeeze_1:output:0/batch_normalization_44/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_44/batchnorm/add?
&batch_normalization_44/batchnorm/RsqrtRsqrt(batch_normalization_44/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_44/batchnorm/Rsqrt?
3batch_normalization_44/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_44_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_44/batchnorm/mul/ReadVariableOp?
$batch_normalization_44/batchnorm/mulMul*batch_normalization_44/batchnorm/Rsqrt:y:0;batch_normalization_44/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_44/batchnorm/mul?
&batch_normalization_44/batchnorm/mul_1Mul#max_pooling3d_44/MaxPool3D:output:0(batch_normalization_44/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_44/batchnorm/mul_1?
&batch_normalization_44/batchnorm/mul_2Mul/batch_normalization_44/moments/Squeeze:output:0(batch_normalization_44/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_44/batchnorm/mul_2?
/batch_normalization_44/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_44_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_44/batchnorm/ReadVariableOp?
$batch_normalization_44/batchnorm/subSub7batch_normalization_44/batchnorm/ReadVariableOp:value:0*batch_normalization_44/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_44/batchnorm/sub?
&batch_normalization_44/batchnorm/add_1AddV2*batch_normalization_44/batchnorm/mul_1:z:0(batch_normalization_44/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_44/batchnorm/add_1?
conv3d_46/Conv3D/ReadVariableOpReadVariableOp(conv3d_46_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_46/Conv3D/ReadVariableOp?
conv3d_46/Conv3DConv3D*batch_normalization_44/batchnorm/add_1:z:0'conv3d_46/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
conv3d_46/Conv3D?
 conv3d_46/BiasAdd/ReadVariableOpReadVariableOp)conv3d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_46/BiasAdd/ReadVariableOp?
conv3d_46/BiasAddBiasAddconv3d_46/Conv3D:output:0(conv3d_46/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2
conv3d_46/BiasAdd?
conv3d_46/ReluReluconv3d_46/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
conv3d_46/Relu?
conv3d_47/Conv3D/ReadVariableOpReadVariableOp(conv3d_47_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_47/Conv3D/ReadVariableOp?
conv3d_47/Conv3DConv3Dconv3d_46/Relu:activations:0'conv3d_47/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
conv3d_47/Conv3D?
 conv3d_47/BiasAdd/ReadVariableOpReadVariableOp)conv3d_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_47/BiasAdd/ReadVariableOp?
conv3d_47/BiasAddBiasAddconv3d_47/Conv3D:output:0(conv3d_47/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2
conv3d_47/BiasAdd?
conv3d_47/SigmoidSigmoidconv3d_47/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
conv3d_47/Sigmoid?
max_pooling3d_45/MaxPool3D	MaxPool3Dconv3d_47/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????			*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_45/MaxPool3D?
5batch_normalization_45/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_45/moments/mean/reduction_indices?
#batch_normalization_45/moments/meanMean#max_pooling3d_45/MaxPool3D:output:0>batch_normalization_45/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2%
#batch_normalization_45/moments/mean?
+batch_normalization_45/moments/StopGradientStopGradient,batch_normalization_45/moments/mean:output:0*
T0**
_output_shapes
:2-
+batch_normalization_45/moments/StopGradient?
0batch_normalization_45/moments/SquaredDifferenceSquaredDifference#max_pooling3d_45/MaxPool3D:output:04batch_normalization_45/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????			22
0batch_normalization_45/moments/SquaredDifference?
9batch_normalization_45/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_45/moments/variance/reduction_indices?
'batch_normalization_45/moments/varianceMean4batch_normalization_45/moments/SquaredDifference:z:0Bbatch_normalization_45/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2)
'batch_normalization_45/moments/variance?
&batch_normalization_45/moments/SqueezeSqueeze,batch_normalization_45/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_45/moments/Squeeze?
(batch_normalization_45/moments/Squeeze_1Squeeze0batch_normalization_45/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_45/moments/Squeeze_1?
,batch_normalization_45/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_45/AssignMovingAvg/decay?
5batch_normalization_45/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_45_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_45/AssignMovingAvg/ReadVariableOp?
*batch_normalization_45/AssignMovingAvg/subSub=batch_normalization_45/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_45/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_45/AssignMovingAvg/sub?
*batch_normalization_45/AssignMovingAvg/mulMul.batch_normalization_45/AssignMovingAvg/sub:z:05batch_normalization_45/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2,
*batch_normalization_45/AssignMovingAvg/mul?
&batch_normalization_45/AssignMovingAvgAssignSubVariableOp>batch_normalization_45_assignmovingavg_readvariableop_resource.batch_normalization_45/AssignMovingAvg/mul:z:06^batch_normalization_45/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_45/AssignMovingAvg?
.batch_normalization_45/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_45/AssignMovingAvg_1/decay?
7batch_normalization_45/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_45_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_45/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_45/AssignMovingAvg_1/subSub?batch_normalization_45/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_45/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_45/AssignMovingAvg_1/sub?
,batch_normalization_45/AssignMovingAvg_1/mulMul0batch_normalization_45/AssignMovingAvg_1/sub:z:07batch_normalization_45/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2.
,batch_normalization_45/AssignMovingAvg_1/mul?
(batch_normalization_45/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_45_assignmovingavg_1_readvariableop_resource0batch_normalization_45/AssignMovingAvg_1/mul:z:08^batch_normalization_45/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_45/AssignMovingAvg_1?
&batch_normalization_45/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_45/batchnorm/add/y?
$batch_normalization_45/batchnorm/addAddV21batch_normalization_45/moments/Squeeze_1:output:0/batch_normalization_45/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_45/batchnorm/add?
&batch_normalization_45/batchnorm/RsqrtRsqrt(batch_normalization_45/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_45/batchnorm/Rsqrt?
3batch_normalization_45/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_45_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_45/batchnorm/mul/ReadVariableOp?
$batch_normalization_45/batchnorm/mulMul*batch_normalization_45/batchnorm/Rsqrt:y:0;batch_normalization_45/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_45/batchnorm/mul?
&batch_normalization_45/batchnorm/mul_1Mul#max_pooling3d_45/MaxPool3D:output:0(batch_normalization_45/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????			2(
&batch_normalization_45/batchnorm/mul_1?
&batch_normalization_45/batchnorm/mul_2Mul/batch_normalization_45/moments/Squeeze:output:0(batch_normalization_45/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_45/batchnorm/mul_2?
/batch_normalization_45/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_45_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_45/batchnorm/ReadVariableOp?
$batch_normalization_45/batchnorm/subSub7batch_normalization_45/batchnorm/ReadVariableOp:value:0*batch_normalization_45/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_45/batchnorm/sub?
&batch_normalization_45/batchnorm/add_1AddV2*batch_normalization_45/batchnorm/mul_1:z:0(batch_normalization_45/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????			2(
&batch_normalization_45/batchnorm/add_1u
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_22/Const?
flatten_22/ReshapeReshape*batch_normalization_45/batchnorm/add_1:z:0flatten_22/Const:output:0*
T0*(
_output_shapes
:??????????-2
flatten_22/Reshape?
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02 
dense_66/MatMul/ReadVariableOp?
dense_66/MatMulMatMulflatten_22/Reshape:output:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_66/MatMul?
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_66/BiasAdd/ReadVariableOp?
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_66/BiasAdd}
dense_66/SoftmaxSoftmaxdense_66/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_66/Softmax?
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_67/MatMul/ReadVariableOp?
dense_67/MatMulMatMuldense_66/Softmax:softmax:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_67/MatMul?
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_67/BiasAdd/ReadVariableOp?
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_67/BiasAdds
dense_67/ReluReludense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_67/Relu?
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_68/MatMul/ReadVariableOp?
dense_68/MatMulMatMuldense_67/Relu:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_68/MatMul?
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_68/BiasAdd/ReadVariableOp?
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_68/BiasAdd|
dense_68/SigmoidSigmoiddense_68/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_68/Sigmoid?	
IdentityIdentitydense_68/Sigmoid:y:0'^batch_normalization_44/AssignMovingAvg6^batch_normalization_44/AssignMovingAvg/ReadVariableOp)^batch_normalization_44/AssignMovingAvg_18^batch_normalization_44/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_44/batchnorm/ReadVariableOp4^batch_normalization_44/batchnorm/mul/ReadVariableOp'^batch_normalization_45/AssignMovingAvg6^batch_normalization_45/AssignMovingAvg/ReadVariableOp)^batch_normalization_45/AssignMovingAvg_18^batch_normalization_45/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_45/batchnorm/ReadVariableOp4^batch_normalization_45/batchnorm/mul/ReadVariableOp!^conv3d_44/BiasAdd/ReadVariableOp ^conv3d_44/Conv3D/ReadVariableOp!^conv3d_45/BiasAdd/ReadVariableOp ^conv3d_45/Conv3D/ReadVariableOp!^conv3d_46/BiasAdd/ReadVariableOp ^conv3d_46/Conv3D/ReadVariableOp!^conv3d_47/BiasAdd/ReadVariableOp ^conv3d_47/Conv3D/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_44/AssignMovingAvg&batch_normalization_44/AssignMovingAvg2n
5batch_normalization_44/AssignMovingAvg/ReadVariableOp5batch_normalization_44/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_44/AssignMovingAvg_1(batch_normalization_44/AssignMovingAvg_12r
7batch_normalization_44/AssignMovingAvg_1/ReadVariableOp7batch_normalization_44/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_44/batchnorm/ReadVariableOp/batch_normalization_44/batchnorm/ReadVariableOp2j
3batch_normalization_44/batchnorm/mul/ReadVariableOp3batch_normalization_44/batchnorm/mul/ReadVariableOp2P
&batch_normalization_45/AssignMovingAvg&batch_normalization_45/AssignMovingAvg2n
5batch_normalization_45/AssignMovingAvg/ReadVariableOp5batch_normalization_45/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_45/AssignMovingAvg_1(batch_normalization_45/AssignMovingAvg_12r
7batch_normalization_45/AssignMovingAvg_1/ReadVariableOp7batch_normalization_45/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_45/batchnorm/ReadVariableOp/batch_normalization_45/batchnorm/ReadVariableOp2j
3batch_normalization_45/batchnorm/mul/ReadVariableOp3batch_normalization_45/batchnorm/mul/ReadVariableOp2D
 conv3d_44/BiasAdd/ReadVariableOp conv3d_44/BiasAdd/ReadVariableOp2B
conv3d_44/Conv3D/ReadVariableOpconv3d_44/Conv3D/ReadVariableOp2D
 conv3d_45/BiasAdd/ReadVariableOp conv3d_45/BiasAdd/ReadVariableOp2B
conv3d_45/Conv3D/ReadVariableOpconv3d_45/Conv3D/ReadVariableOp2D
 conv3d_46/BiasAdd/ReadVariableOp conv3d_46/BiasAdd/ReadVariableOp2B
conv3d_46/Conv3D/ReadVariableOpconv3d_46/Conv3D/ReadVariableOp2D
 conv3d_47/BiasAdd/ReadVariableOp conv3d_47/BiasAdd/ReadVariableOp2B
conv3d_47/Conv3D/ReadVariableOpconv3d_47/Conv3D/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?

?
C__inference_dense_66_layer_call_and_return_conditional_losses_30984

inputs2
matmul_readvariableop_resource:
?-?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
ԡ
?+
!__inference__traced_restore_32862
file_prefix?
!assignvariableop_conv3d_44_kernel:/
!assignvariableop_1_conv3d_44_bias:A
#assignvariableop_2_conv3d_45_kernel:/
!assignvariableop_3_conv3d_45_bias:=
/assignvariableop_4_batch_normalization_44_gamma:<
.assignvariableop_5_batch_normalization_44_beta:C
5assignvariableop_6_batch_normalization_44_moving_mean:G
9assignvariableop_7_batch_normalization_44_moving_variance:A
#assignvariableop_8_conv3d_46_kernel:/
!assignvariableop_9_conv3d_46_bias:B
$assignvariableop_10_conv3d_47_kernel:0
"assignvariableop_11_conv3d_47_bias:>
0assignvariableop_12_batch_normalization_45_gamma:=
/assignvariableop_13_batch_normalization_45_beta:D
6assignvariableop_14_batch_normalization_45_moving_mean:H
:assignvariableop_15_batch_normalization_45_moving_variance:7
#assignvariableop_16_dense_66_kernel:
?-?0
!assignvariableop_17_dense_66_bias:	?6
#assignvariableop_18_dense_67_kernel:	?@/
!assignvariableop_19_dense_67_bias:@5
#assignvariableop_20_dense_68_kernel:@/
!assignvariableop_21_dense_68_bias:'
assignvariableop_22_adam_iter:	 )
assignvariableop_23_adam_beta_1: )
assignvariableop_24_adam_beta_2: (
assignvariableop_25_adam_decay: 0
&assignvariableop_26_adam_learning_rate: #
assignvariableop_27_total: #
assignvariableop_28_count: %
assignvariableop_29_total_1: %
assignvariableop_30_count_1: I
+assignvariableop_31_adam_conv3d_44_kernel_m:7
)assignvariableop_32_adam_conv3d_44_bias_m:I
+assignvariableop_33_adam_conv3d_45_kernel_m:7
)assignvariableop_34_adam_conv3d_45_bias_m:E
7assignvariableop_35_adam_batch_normalization_44_gamma_m:D
6assignvariableop_36_adam_batch_normalization_44_beta_m:I
+assignvariableop_37_adam_conv3d_46_kernel_m:7
)assignvariableop_38_adam_conv3d_46_bias_m:I
+assignvariableop_39_adam_conv3d_47_kernel_m:7
)assignvariableop_40_adam_conv3d_47_bias_m:E
7assignvariableop_41_adam_batch_normalization_45_gamma_m:D
6assignvariableop_42_adam_batch_normalization_45_beta_m:>
*assignvariableop_43_adam_dense_66_kernel_m:
?-?7
(assignvariableop_44_adam_dense_66_bias_m:	?=
*assignvariableop_45_adam_dense_67_kernel_m:	?@6
(assignvariableop_46_adam_dense_67_bias_m:@<
*assignvariableop_47_adam_dense_68_kernel_m:@6
(assignvariableop_48_adam_dense_68_bias_m:I
+assignvariableop_49_adam_conv3d_44_kernel_v:7
)assignvariableop_50_adam_conv3d_44_bias_v:I
+assignvariableop_51_adam_conv3d_45_kernel_v:7
)assignvariableop_52_adam_conv3d_45_bias_v:E
7assignvariableop_53_adam_batch_normalization_44_gamma_v:D
6assignvariableop_54_adam_batch_normalization_44_beta_v:I
+assignvariableop_55_adam_conv3d_46_kernel_v:7
)assignvariableop_56_adam_conv3d_46_bias_v:I
+assignvariableop_57_adam_conv3d_47_kernel_v:7
)assignvariableop_58_adam_conv3d_47_bias_v:E
7assignvariableop_59_adam_batch_normalization_45_gamma_v:D
6assignvariableop_60_adam_batch_normalization_45_beta_v:>
*assignvariableop_61_adam_dense_66_kernel_v:
?-?7
(assignvariableop_62_adam_dense_66_bias_v:	?=
*assignvariableop_63_adam_dense_67_kernel_v:	?@6
(assignvariableop_64_adam_dense_67_bias_v:@<
*assignvariableop_65_adam_dense_68_kernel_v:@6
(assignvariableop_66_adam_dense_68_bias_v:
identity_68??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?$
value?$B?$DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv3d_44_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv3d_44_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv3d_45_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv3d_45_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_44_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_44_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_44_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_44_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv3d_46_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv3d_46_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv3d_47_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv3d_47_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_45_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_45_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_45_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_45_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_66_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_66_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_67_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_67_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_68_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_68_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv3d_44_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv3d_44_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv3d_45_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv3d_45_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_batch_normalization_44_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_batch_normalization_44_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv3d_46_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv3d_46_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv3d_47_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv3d_47_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_45_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_45_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_66_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_66_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_67_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_67_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_68_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_68_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv3d_44_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv3d_44_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv3d_45_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv3d_45_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_44_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_44_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv3d_46_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv3d_46_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv3d_47_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv3d_47_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_45_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_45_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_66_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_66_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_67_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_67_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_68_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_68_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_669
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_67?
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_68"#
identity_68Identity_68:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
g
K__inference_max_pooling3d_44_layer_call_and_return_conditional_losses_30488

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_44_layer_call_fn_32117

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_305182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32016

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32270

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????			2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????			2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????			2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????			: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs
?
?
(__inference_dense_68_layer_call_fn_32427

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_68_layer_call_and_return_conditional_losses_310182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
H__inference_sequential_22_layer_call_and_return_conditional_losses_31741

inputsF
(conv3d_44_conv3d_readvariableop_resource:7
)conv3d_44_biasadd_readvariableop_resource:F
(conv3d_45_conv3d_readvariableop_resource:7
)conv3d_45_biasadd_readvariableop_resource:F
8batch_normalization_44_batchnorm_readvariableop_resource:J
<batch_normalization_44_batchnorm_mul_readvariableop_resource:H
:batch_normalization_44_batchnorm_readvariableop_1_resource:H
:batch_normalization_44_batchnorm_readvariableop_2_resource:F
(conv3d_46_conv3d_readvariableop_resource:7
)conv3d_46_biasadd_readvariableop_resource:F
(conv3d_47_conv3d_readvariableop_resource:7
)conv3d_47_biasadd_readvariableop_resource:F
8batch_normalization_45_batchnorm_readvariableop_resource:J
<batch_normalization_45_batchnorm_mul_readvariableop_resource:H
:batch_normalization_45_batchnorm_readvariableop_1_resource:H
:batch_normalization_45_batchnorm_readvariableop_2_resource:;
'dense_66_matmul_readvariableop_resource:
?-?7
(dense_66_biasadd_readvariableop_resource:	?:
'dense_67_matmul_readvariableop_resource:	?@6
(dense_67_biasadd_readvariableop_resource:@9
'dense_68_matmul_readvariableop_resource:@6
(dense_68_biasadd_readvariableop_resource:
identity??/batch_normalization_44/batchnorm/ReadVariableOp?1batch_normalization_44/batchnorm/ReadVariableOp_1?1batch_normalization_44/batchnorm/ReadVariableOp_2?3batch_normalization_44/batchnorm/mul/ReadVariableOp?/batch_normalization_45/batchnorm/ReadVariableOp?1batch_normalization_45/batchnorm/ReadVariableOp_1?1batch_normalization_45/batchnorm/ReadVariableOp_2?3batch_normalization_45/batchnorm/mul/ReadVariableOp? conv3d_44/BiasAdd/ReadVariableOp?conv3d_44/Conv3D/ReadVariableOp? conv3d_45/BiasAdd/ReadVariableOp?conv3d_45/Conv3D/ReadVariableOp? conv3d_46/BiasAdd/ReadVariableOp?conv3d_46/Conv3D/ReadVariableOp? conv3d_47/BiasAdd/ReadVariableOp?conv3d_47/Conv3D/ReadVariableOp?dense_66/BiasAdd/ReadVariableOp?dense_66/MatMul/ReadVariableOp?dense_67/BiasAdd/ReadVariableOp?dense_67/MatMul/ReadVariableOp?dense_68/BiasAdd/ReadVariableOp?dense_68/MatMul/ReadVariableOp?
conv3d_44/Conv3D/ReadVariableOpReadVariableOp(conv3d_44_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_44/Conv3D/ReadVariableOp?
conv3d_44/Conv3DConv3Dinputs'conv3d_44/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000*
paddingVALID*
strides	
2
conv3d_44/Conv3D?
 conv3d_44/BiasAdd/ReadVariableOpReadVariableOp)conv3d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_44/BiasAdd/ReadVariableOp?
conv3d_44/BiasAddBiasAddconv3d_44/Conv3D:output:0(conv3d_44/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????0002
conv3d_44/BiasAdd?
conv3d_44/ReluReluconv3d_44/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002
conv3d_44/Relu?
conv3d_45/Conv3D/ReadVariableOpReadVariableOp(conv3d_45_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_45/Conv3D/ReadVariableOp?
conv3d_45/Conv3DConv3Dconv3d_44/Relu:activations:0'conv3d_45/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...*
paddingVALID*
strides	
2
conv3d_45/Conv3D?
 conv3d_45/BiasAdd/ReadVariableOpReadVariableOp)conv3d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_45/BiasAdd/ReadVariableOp?
conv3d_45/BiasAddBiasAddconv3d_45/Conv3D:output:0(conv3d_45/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????...2
conv3d_45/BiasAdd?
conv3d_45/SigmoidSigmoidconv3d_45/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????...2
conv3d_45/Sigmoid?
max_pooling3d_44/MaxPool3D	MaxPool3Dconv3d_45/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_44/MaxPool3D?
/batch_normalization_44/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_44_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_44/batchnorm/ReadVariableOp?
&batch_normalization_44/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_44/batchnorm/add/y?
$batch_normalization_44/batchnorm/addAddV27batch_normalization_44/batchnorm/ReadVariableOp:value:0/batch_normalization_44/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_44/batchnorm/add?
&batch_normalization_44/batchnorm/RsqrtRsqrt(batch_normalization_44/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_44/batchnorm/Rsqrt?
3batch_normalization_44/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_44_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_44/batchnorm/mul/ReadVariableOp?
$batch_normalization_44/batchnorm/mulMul*batch_normalization_44/batchnorm/Rsqrt:y:0;batch_normalization_44/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_44/batchnorm/mul?
&batch_normalization_44/batchnorm/mul_1Mul#max_pooling3d_44/MaxPool3D:output:0(batch_normalization_44/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_44/batchnorm/mul_1?
1batch_normalization_44/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_44_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_44/batchnorm/ReadVariableOp_1?
&batch_normalization_44/batchnorm/mul_2Mul9batch_normalization_44/batchnorm/ReadVariableOp_1:value:0(batch_normalization_44/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_44/batchnorm/mul_2?
1batch_normalization_44/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_44_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_44/batchnorm/ReadVariableOp_2?
$batch_normalization_44/batchnorm/subSub9batch_normalization_44/batchnorm/ReadVariableOp_2:value:0*batch_normalization_44/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_44/batchnorm/sub?
&batch_normalization_44/batchnorm/add_1AddV2*batch_normalization_44/batchnorm/mul_1:z:0(batch_normalization_44/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_44/batchnorm/add_1?
conv3d_46/Conv3D/ReadVariableOpReadVariableOp(conv3d_46_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_46/Conv3D/ReadVariableOp?
conv3d_46/Conv3DConv3D*batch_normalization_44/batchnorm/add_1:z:0'conv3d_46/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
conv3d_46/Conv3D?
 conv3d_46/BiasAdd/ReadVariableOpReadVariableOp)conv3d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_46/BiasAdd/ReadVariableOp?
conv3d_46/BiasAddBiasAddconv3d_46/Conv3D:output:0(conv3d_46/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2
conv3d_46/BiasAdd?
conv3d_46/ReluReluconv3d_46/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
conv3d_46/Relu?
conv3d_47/Conv3D/ReadVariableOpReadVariableOp(conv3d_47_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_47/Conv3D/ReadVariableOp?
conv3d_47/Conv3DConv3Dconv3d_46/Relu:activations:0'conv3d_47/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
conv3d_47/Conv3D?
 conv3d_47/BiasAdd/ReadVariableOpReadVariableOp)conv3d_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_47/BiasAdd/ReadVariableOp?
conv3d_47/BiasAddBiasAddconv3d_47/Conv3D:output:0(conv3d_47/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2
conv3d_47/BiasAdd?
conv3d_47/SigmoidSigmoidconv3d_47/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
conv3d_47/Sigmoid?
max_pooling3d_45/MaxPool3D	MaxPool3Dconv3d_47/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????			*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_45/MaxPool3D?
/batch_normalization_45/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_45_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_45/batchnorm/ReadVariableOp?
&batch_normalization_45/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_45/batchnorm/add/y?
$batch_normalization_45/batchnorm/addAddV27batch_normalization_45/batchnorm/ReadVariableOp:value:0/batch_normalization_45/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_45/batchnorm/add?
&batch_normalization_45/batchnorm/RsqrtRsqrt(batch_normalization_45/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_45/batchnorm/Rsqrt?
3batch_normalization_45/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_45_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_45/batchnorm/mul/ReadVariableOp?
$batch_normalization_45/batchnorm/mulMul*batch_normalization_45/batchnorm/Rsqrt:y:0;batch_normalization_45/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_45/batchnorm/mul?
&batch_normalization_45/batchnorm/mul_1Mul#max_pooling3d_45/MaxPool3D:output:0(batch_normalization_45/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????			2(
&batch_normalization_45/batchnorm/mul_1?
1batch_normalization_45/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_45_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_45/batchnorm/ReadVariableOp_1?
&batch_normalization_45/batchnorm/mul_2Mul9batch_normalization_45/batchnorm/ReadVariableOp_1:value:0(batch_normalization_45/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_45/batchnorm/mul_2?
1batch_normalization_45/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_45_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_45/batchnorm/ReadVariableOp_2?
$batch_normalization_45/batchnorm/subSub9batch_normalization_45/batchnorm/ReadVariableOp_2:value:0*batch_normalization_45/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_45/batchnorm/sub?
&batch_normalization_45/batchnorm/add_1AddV2*batch_normalization_45/batchnorm/mul_1:z:0(batch_normalization_45/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????			2(
&batch_normalization_45/batchnorm/add_1u
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_22/Const?
flatten_22/ReshapeReshape*batch_normalization_45/batchnorm/add_1:z:0flatten_22/Const:output:0*
T0*(
_output_shapes
:??????????-2
flatten_22/Reshape?
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02 
dense_66/MatMul/ReadVariableOp?
dense_66/MatMulMatMulflatten_22/Reshape:output:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_66/MatMul?
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_66/BiasAdd/ReadVariableOp?
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_66/BiasAdd}
dense_66/SoftmaxSoftmaxdense_66/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_66/Softmax?
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_67/MatMul/ReadVariableOp?
dense_67/MatMulMatMuldense_66/Softmax:softmax:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_67/MatMul?
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_67/BiasAdd/ReadVariableOp?
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_67/BiasAdds
dense_67/ReluReludense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_67/Relu?
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_68/MatMul/ReadVariableOp?
dense_68/MatMulMatMuldense_67/Relu:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_68/MatMul?
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_68/BiasAdd/ReadVariableOp?
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_68/BiasAdd|
dense_68/SigmoidSigmoiddense_68/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_68/Sigmoid?
IdentityIdentitydense_68/Sigmoid:y:00^batch_normalization_44/batchnorm/ReadVariableOp2^batch_normalization_44/batchnorm/ReadVariableOp_12^batch_normalization_44/batchnorm/ReadVariableOp_24^batch_normalization_44/batchnorm/mul/ReadVariableOp0^batch_normalization_45/batchnorm/ReadVariableOp2^batch_normalization_45/batchnorm/ReadVariableOp_12^batch_normalization_45/batchnorm/ReadVariableOp_24^batch_normalization_45/batchnorm/mul/ReadVariableOp!^conv3d_44/BiasAdd/ReadVariableOp ^conv3d_44/Conv3D/ReadVariableOp!^conv3d_45/BiasAdd/ReadVariableOp ^conv3d_45/Conv3D/ReadVariableOp!^conv3d_46/BiasAdd/ReadVariableOp ^conv3d_46/Conv3D/ReadVariableOp!^conv3d_47/BiasAdd/ReadVariableOp ^conv3d_47/Conv3D/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_44/batchnorm/ReadVariableOp/batch_normalization_44/batchnorm/ReadVariableOp2f
1batch_normalization_44/batchnorm/ReadVariableOp_11batch_normalization_44/batchnorm/ReadVariableOp_12f
1batch_normalization_44/batchnorm/ReadVariableOp_21batch_normalization_44/batchnorm/ReadVariableOp_22j
3batch_normalization_44/batchnorm/mul/ReadVariableOp3batch_normalization_44/batchnorm/mul/ReadVariableOp2b
/batch_normalization_45/batchnorm/ReadVariableOp/batch_normalization_45/batchnorm/ReadVariableOp2f
1batch_normalization_45/batchnorm/ReadVariableOp_11batch_normalization_45/batchnorm/ReadVariableOp_12f
1batch_normalization_45/batchnorm/ReadVariableOp_21batch_normalization_45/batchnorm/ReadVariableOp_22j
3batch_normalization_45/batchnorm/mul/ReadVariableOp3batch_normalization_45/batchnorm/mul/ReadVariableOp2D
 conv3d_44/BiasAdd/ReadVariableOp conv3d_44/BiasAdd/ReadVariableOp2B
conv3d_44/Conv3D/ReadVariableOpconv3d_44/Conv3D/ReadVariableOp2D
 conv3d_45/BiasAdd/ReadVariableOp conv3d_45/BiasAdd/ReadVariableOp2B
conv3d_45/Conv3D/ReadVariableOpconv3d_45/Conv3D/ReadVariableOp2D
 conv3d_46/BiasAdd/ReadVariableOp conv3d_46/BiasAdd/ReadVariableOp2B
conv3d_46/Conv3D/ReadVariableOpconv3d_46/Conv3D/ReadVariableOp2D
 conv3d_47/BiasAdd/ReadVariableOp conv3d_47/BiasAdd/ReadVariableOp2B
conv3d_47/Conv3D/ReadVariableOpconv3d_47/Conv3D/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32070

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_31652
conv3d_44_input%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
?-?

unknown_16:	?

unknown_17:	?@

unknown_18:@

unknown_19:@

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_44_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_304822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????222: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_44_input
?
L
0__inference_max_pooling3d_45_layer_call_fn_30668

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_45_layer_call_and_return_conditional_losses_306622
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv3d_47_layer_call_fn_32196

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_47_layer_call_and_return_conditional_losses_309292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32304

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????			2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????			2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????			2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????			2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????			: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????			
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
W
conv3d_44_inputD
!serving_default_conv3d_44_input:0?????????222<
dense_680
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?u
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?p
_tf_keras_sequential?p{"name": "sequential_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_44_input"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_44", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv3D", "config": {"name": "conv3d_45", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_44", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv3D", "config": {"name": "conv3d_46", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv3D", "config": {"name": "conv3d_47", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 256, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}, "float32", "conv3d_44_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_44_input"}, "shared_object_id": 0}, {"class_name": "Conv3D", "config": {"name": "conv3d_44", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Conv3D", "config": {"name": "conv3d_45", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_44", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 11}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 12}, {"class_name": "Conv3D", "config": {"name": "conv3d_46", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}, {"class_name": "Conv3D", "config": {"name": "conv3d_47", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 19}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 21}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 24}, {"class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 25}, {"class_name": "Dense", "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 256, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31}, {"class_name": "Dense", "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 34}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 37}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.05000000074505806, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv3d_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_44", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_45", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 8}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 48, 8]}}
?
trainable_variables
 	variables
!regularization_losses
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling3d_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_44", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 39}}
?

#axis
	$gamma
%beta
&moving_mean
'moving_variance
(trainable_variables
)	variables
*regularization_losses
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 11}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 8}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 23, 8]}}
?


,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_46", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 8}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 23, 8]}}
?


2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3d_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_47", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 8}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 21, 8]}}
?
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling3d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
?

<axis
	=gamma
>beta
?moving_mean
@moving_variance
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 21}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 8}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 9, 8]}}
?
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 45}}
?

Ikernel
Jbias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_66", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 256, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5832}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5832]}}
?

Okernel
Pbias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_67", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ukernel
Vbias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_68", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
[iter

\beta_1

]beta_2
	^decay
_learning_ratem?m?m?m?$m?%m?,m?-m?2m?3m?=m?>m?Im?Jm?Om?Pm?Um?Vm?v?v?v?v?$v?%v?,v?-v?2v?3v?=v?>v?Iv?Jv?Ov?Pv?Uv?Vv?"
	optimizer
?
0
1
2
3
$4
%5
,6
-7
28
39
=10
>11
I12
J13
O14
P15
U16
V17"
trackable_list_wrapper
?
0
1
2
3
$4
%5
&6
'7
,8
-9
210
311
=12
>13
?14
@15
I16
J17
O18
P19
U20
V21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`non_trainable_variables
trainable_variables
ametrics
blayer_metrics

clayers
	variables
regularization_losses
dlayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
.:,2conv3d_44/kernel
:2conv3d_44/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
enon_trainable_variables
trainable_variables
fmetrics
glayer_metrics

hlayers
	variables
regularization_losses
ilayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_45/kernel
:2conv3d_45/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jnon_trainable_variables
trainable_variables
kmetrics
llayer_metrics

mlayers
	variables
regularization_losses
nlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
onon_trainable_variables
trainable_variables
pmetrics
qlayer_metrics

rlayers
 	variables
!regularization_losses
slayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_44/gamma
):'2batch_normalization_44/beta
2:0 (2"batch_normalization_44/moving_mean
6:4 (2&batch_normalization_44/moving_variance
.
$0
%1"
trackable_list_wrapper
<
$0
%1
&2
'3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tnon_trainable_variables
(trainable_variables
umetrics
vlayer_metrics

wlayers
)	variables
*regularization_losses
xlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_46/kernel
:2conv3d_46/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables
.trainable_variables
zmetrics
{layer_metrics

|layers
/	variables
0regularization_losses
}layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_47/kernel
:2conv3d_47/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
~non_trainable_variables
4trainable_variables
metrics
?layer_metrics
?layers
5	variables
6regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
8trainable_variables
?metrics
?layer_metrics
?layers
9	variables
:regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_45/gamma
):'2batch_normalization_45/beta
2:0 (2"batch_normalization_45/moving_mean
6:4 (2&batch_normalization_45/moving_variance
.
=0
>1"
trackable_list_wrapper
<
=0
>1
?2
@3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Atrainable_variables
?metrics
?layer_metrics
?layers
B	variables
Cregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Etrainable_variables
?metrics
?layer_metrics
?layers
F	variables
Gregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?-?2dense_66/kernel
:?2dense_66/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Ktrainable_variables
?metrics
?layer_metrics
?layers
L	variables
Mregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?@2dense_67/kernel
:@2dense_67/bias
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Qtrainable_variables
?metrics
?layer_metrics
?layers
R	variables
Sregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_68/kernel
:2dense_68/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Wtrainable_variables
?metrics
?layer_metrics
?layers
X	variables
Yregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
&0
'1
?2
@3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
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
.
&0
'1"
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
.
?0
@1"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 37}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
3:12Adam/conv3d_44/kernel/m
!:2Adam/conv3d_44/bias/m
3:12Adam/conv3d_45/kernel/m
!:2Adam/conv3d_45/bias/m
/:-2#Adam/batch_normalization_44/gamma/m
.:,2"Adam/batch_normalization_44/beta/m
3:12Adam/conv3d_46/kernel/m
!:2Adam/conv3d_46/bias/m
3:12Adam/conv3d_47/kernel/m
!:2Adam/conv3d_47/bias/m
/:-2#Adam/batch_normalization_45/gamma/m
.:,2"Adam/batch_normalization_45/beta/m
(:&
?-?2Adam/dense_66/kernel/m
!:?2Adam/dense_66/bias/m
':%	?@2Adam/dense_67/kernel/m
 :@2Adam/dense_67/bias/m
&:$@2Adam/dense_68/kernel/m
 :2Adam/dense_68/bias/m
3:12Adam/conv3d_44/kernel/v
!:2Adam/conv3d_44/bias/v
3:12Adam/conv3d_45/kernel/v
!:2Adam/conv3d_45/bias/v
/:-2#Adam/batch_normalization_44/gamma/v
.:,2"Adam/batch_normalization_44/beta/v
3:12Adam/conv3d_46/kernel/v
!:2Adam/conv3d_46/bias/v
3:12Adam/conv3d_47/kernel/v
!:2Adam/conv3d_47/bias/v
/:-2#Adam/batch_normalization_45/gamma/v
.:,2"Adam/batch_normalization_45/beta/v
(:&
?-?2Adam/dense_66/kernel/v
!:?2Adam/dense_66/bias/v
':%	?@2Adam/dense_67/kernel/v
 :@2Adam/dense_67/bias/v
&:$@2Adam/dense_68/kernel/v
 :2Adam/dense_68/bias/v
?2?
H__inference_sequential_22_layer_call_and_return_conditional_losses_31741
H__inference_sequential_22_layer_call_and_return_conditional_losses_31858
H__inference_sequential_22_layer_call_and_return_conditional_losses_31535
H__inference_sequential_22_layer_call_and_return_conditional_losses_31595?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_22_layer_call_fn_31072
-__inference_sequential_22_layer_call_fn_31907
-__inference_sequential_22_layer_call_fn_31956
-__inference_sequential_22_layer_call_fn_31475?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_30482?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *:?7
5?2
conv3d_44_input?????????222
?2?
D__inference_conv3d_44_layer_call_and_return_conditional_losses_31967?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv3d_44_layer_call_fn_31976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv3d_45_layer_call_and_return_conditional_losses_31987?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv3d_45_layer_call_fn_31996?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling3d_44_layer_call_and_return_conditional_losses_30488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
0__inference_max_pooling3d_44_layer_call_fn_30494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32016
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32050
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32070
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32104?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_44_layer_call_fn_32117
6__inference_batch_normalization_44_layer_call_fn_32130
6__inference_batch_normalization_44_layer_call_fn_32143
6__inference_batch_normalization_44_layer_call_fn_32156?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv3d_46_layer_call_and_return_conditional_losses_32167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv3d_46_layer_call_fn_32176?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv3d_47_layer_call_and_return_conditional_losses_32187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv3d_47_layer_call_fn_32196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling3d_45_layer_call_and_return_conditional_losses_30662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
0__inference_max_pooling3d_45_layer_call_fn_30668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32216
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32250
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32270
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32304?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_45_layer_call_fn_32317
6__inference_batch_normalization_45_layer_call_fn_32330
6__inference_batch_normalization_45_layer_call_fn_32343
6__inference_batch_normalization_45_layer_call_fn_32356?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_flatten_22_layer_call_and_return_conditional_losses_32362?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_22_layer_call_fn_32367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_66_layer_call_and_return_conditional_losses_32378?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_66_layer_call_fn_32387?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_67_layer_call_and_return_conditional_losses_32398?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_67_layer_call_fn_32407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_68_layer_call_and_return_conditional_losses_32418?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_68_layer_call_fn_32427?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_31652conv3d_44_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_30482?'$&%,-23@=?>IJOPUVD?A
:?7
5?2
conv3d_44_input?????????222
? "3?0
.
dense_68"?
dense_68??????????
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32016?'$&%Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32050?&'$%Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32070z'$&%??<
5?2
,?)
inputs?????????
p 
? "1?.
'?$
0?????????
? ?
Q__inference_batch_normalization_44_layer_call_and_return_conditional_losses_32104z&'$%??<
5?2
,?)
inputs?????????
p
? "1?.
'?$
0?????????
? ?
6__inference_batch_normalization_44_layer_call_fn_32117?'$&%Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "??<8?????????????????????????????????????
6__inference_batch_normalization_44_layer_call_fn_32130?&'$%Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "??<8?????????????????????????????????????
6__inference_batch_normalization_44_layer_call_fn_32143m'$&%??<
5?2
,?)
inputs?????????
p 
? "$?!??????????
6__inference_batch_normalization_44_layer_call_fn_32156m&'$%??<
5?2
,?)
inputs?????????
p
? "$?!??????????
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32216?@=?>Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32250??@=>Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32270z@=?>??<
5?2
,?)
inputs?????????			
p 
? "1?.
'?$
0?????????			
? ?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_32304z?@=>??<
5?2
,?)
inputs?????????			
p
? "1?.
'?$
0?????????			
? ?
6__inference_batch_normalization_45_layer_call_fn_32317?@=?>Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "??<8?????????????????????????????????????
6__inference_batch_normalization_45_layer_call_fn_32330??@=>Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "??<8?????????????????????????????????????
6__inference_batch_normalization_45_layer_call_fn_32343m@=?>??<
5?2
,?)
inputs?????????			
p 
? "$?!?????????			?
6__inference_batch_normalization_45_layer_call_fn_32356m?@=>??<
5?2
,?)
inputs?????????			
p
? "$?!?????????			?
D__inference_conv3d_44_layer_call_and_return_conditional_losses_31967t;?8
1?.
,?)
inputs?????????222
? "1?.
'?$
0?????????000
? ?
)__inference_conv3d_44_layer_call_fn_31976g;?8
1?.
,?)
inputs?????????222
? "$?!?????????000?
D__inference_conv3d_45_layer_call_and_return_conditional_losses_31987t;?8
1?.
,?)
inputs?????????000
? "1?.
'?$
0?????????...
? ?
)__inference_conv3d_45_layer_call_fn_31996g;?8
1?.
,?)
inputs?????????000
? "$?!?????????...?
D__inference_conv3d_46_layer_call_and_return_conditional_losses_32167t,-;?8
1?.
,?)
inputs?????????
? "1?.
'?$
0?????????
? ?
)__inference_conv3d_46_layer_call_fn_32176g,-;?8
1?.
,?)
inputs?????????
? "$?!??????????
D__inference_conv3d_47_layer_call_and_return_conditional_losses_32187t23;?8
1?.
,?)
inputs?????????
? "1?.
'?$
0?????????
? ?
)__inference_conv3d_47_layer_call_fn_32196g23;?8
1?.
,?)
inputs?????????
? "$?!??????????
C__inference_dense_66_layer_call_and_return_conditional_losses_32378^IJ0?-
&?#
!?
inputs??????????-
? "&?#
?
0??????????
? }
(__inference_dense_66_layer_call_fn_32387QIJ0?-
&?#
!?
inputs??????????-
? "????????????
C__inference_dense_67_layer_call_and_return_conditional_losses_32398]OP0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_dense_67_layer_call_fn_32407POP0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_68_layer_call_and_return_conditional_losses_32418\UV/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_dense_68_layer_call_fn_32427OUV/?,
%?"
 ?
inputs?????????@
? "???????????
E__inference_flatten_22_layer_call_and_return_conditional_losses_32362e;?8
1?.
,?)
inputs?????????			
? "&?#
?
0??????????-
? ?
*__inference_flatten_22_layer_call_fn_32367X;?8
1?.
,?)
inputs?????????			
? "???????????-?
K__inference_max_pooling3d_44_layer_call_and_return_conditional_losses_30488?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
0__inference_max_pooling3d_44_layer_call_fn_30494?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
K__inference_max_pooling3d_45_layer_call_and_return_conditional_losses_30662?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
0__inference_max_pooling3d_45_layer_call_fn_30668?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
H__inference_sequential_22_layer_call_and_return_conditional_losses_31535?'$&%,-23@=?>IJOPUVL?I
B??
5?2
conv3d_44_input?????????222
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_22_layer_call_and_return_conditional_losses_31595?&'$%,-23?@=>IJOPUVL?I
B??
5?2
conv3d_44_input?????????222
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_22_layer_call_and_return_conditional_losses_31741?'$&%,-23@=?>IJOPUVC?@
9?6
,?)
inputs?????????222
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_22_layer_call_and_return_conditional_losses_31858?&'$%,-23?@=>IJOPUVC?@
9?6
,?)
inputs?????????222
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_22_layer_call_fn_31072?'$&%,-23@=?>IJOPUVL?I
B??
5?2
conv3d_44_input?????????222
p 

 
? "???????????
-__inference_sequential_22_layer_call_fn_31475?&'$%,-23?@=>IJOPUVL?I
B??
5?2
conv3d_44_input?????????222
p

 
? "???????????
-__inference_sequential_22_layer_call_fn_31907w'$&%,-23@=?>IJOPUVC?@
9?6
,?)
inputs?????????222
p 

 
? "???????????
-__inference_sequential_22_layer_call_fn_31956w&'$%,-23?@=>IJOPUVC?@
9?6
,?)
inputs?????????222
p

 
? "???????????
#__inference_signature_wrapper_31652?'$&%,-23@=?>IJOPUVW?T
? 
M?J
H
conv3d_44_input5?2
conv3d_44_input?????????222"3?0
.
dense_68"?
dense_68?????????