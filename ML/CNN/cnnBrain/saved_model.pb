??
??
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
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
conv3d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_42/kernel
?
$conv3d_42/kernel/Read/ReadVariableOpReadVariableOpconv3d_42/kernel**
_output_shapes
:*
dtype0
t
conv3d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_42/bias
m
"conv3d_42/bias/Read/ReadVariableOpReadVariableOpconv3d_42/bias*
_output_shapes
:*
dtype0
?
batch_normalization_42/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_42/gamma
?
0batch_normalization_42/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_42/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_42/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_42/beta
?
/batch_normalization_42/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_42/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_42/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_42/moving_mean
?
6batch_normalization_42/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_42/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_42/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_42/moving_variance
?
:batch_normalization_42/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_42/moving_variance*
_output_shapes
:*
dtype0
?
conv3d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_43/kernel
?
$conv3d_43/kernel/Read/ReadVariableOpReadVariableOpconv3d_43/kernel**
_output_shapes
:*
dtype0
t
conv3d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_43/bias
m
"conv3d_43/bias/Read/ReadVariableOpReadVariableOpconv3d_43/bias*
_output_shapes
:*
dtype0
?
batch_normalization_43/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_43/gamma
?
0batch_normalization_43/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_43/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_43/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_43/beta
?
/batch_normalization_43/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_43/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_43/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_43/moving_mean
?
6batch_normalization_43/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_43/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_43/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_43/moving_variance
?
:batch_normalization_43/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_43/moving_variance*
_output_shapes
:*
dtype0
{
dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?S@* 
shared_namedense_63/kernel
t
#dense_63/kernel/Read/ReadVariableOpReadVariableOpdense_63/kernel*
_output_shapes
:	?S@*
dtype0
r
dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_63/bias
k
!dense_63/bias/Read/ReadVariableOpReadVariableOpdense_63/bias*
_output_shapes
:@*
dtype0
z
dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_64/kernel
s
#dense_64/kernel/Read/ReadVariableOpReadVariableOpdense_64/kernel*
_output_shapes

:@ *
dtype0
r
dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_64/bias
k
!dense_64/bias/Read/ReadVariableOpReadVariableOpdense_64/bias*
_output_shapes
: *
dtype0
z
dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_65/kernel
s
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel*
_output_shapes

: *
dtype0
r
dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_65/bias
k
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
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
Adam/conv3d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_42/kernel/m
?
+Adam/conv3d_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_42/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_42/bias/m
{
)Adam/conv3d_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_42/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_42/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_42/gamma/m
?
7Adam/batch_normalization_42/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_42/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_42/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_42/beta/m
?
6Adam/batch_normalization_42/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_42/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_43/kernel/m
?
+Adam/conv3d_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_43/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_43/bias/m
{
)Adam/conv3d_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_43/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_43/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_43/gamma/m
?
7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_43/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_43/beta/m
?
6Adam/batch_normalization_43/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?S@*'
shared_nameAdam/dense_63/kernel/m
?
*Adam/dense_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/m*
_output_shapes
:	?S@*
dtype0
?
Adam/dense_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_63/bias/m
y
(Adam/dense_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_64/kernel/m
?
*Adam/dense_64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_64/kernel/m*
_output_shapes

:@ *
dtype0
?
Adam/dense_64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_64/bias/m
y
(Adam/dense_64/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_64/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_65/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_65/kernel/m
?
*Adam/dense_65/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_65/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_65/bias/m
y
(Adam/dense_65/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_42/kernel/v
?
+Adam/conv3d_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_42/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_42/bias/v
{
)Adam/conv3d_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_42/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_42/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_42/gamma/v
?
7Adam/batch_normalization_42/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_42/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_42/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_42/beta/v
?
6Adam/batch_normalization_42/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_42/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_43/kernel/v
?
+Adam/conv3d_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_43/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_43/bias/v
{
)Adam/conv3d_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_43/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_43/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_43/gamma/v
?
7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_43/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_43/beta/v
?
6Adam/batch_normalization_43/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?S@*'
shared_nameAdam/dense_63/kernel/v
?
*Adam/dense_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/v*
_output_shapes
:	?S@*
dtype0
?
Adam/dense_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_63/bias/v
y
(Adam/dense_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_64/kernel/v
?
*Adam/dense_64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_64/kernel/v*
_output_shapes

:@ *
dtype0
?
Adam/dense_64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_64/bias/v
y
(Adam/dense_64/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_64/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_65/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_65/kernel/v
?
*Adam/dense_65/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_65/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_65/bias/v
y
(Adam/dense_65/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?V
value?VB?U B?U
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
 trainable_variables
!	variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
?
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3trainable_variables
4	variables
5regularization_losses
6	keras_api
R
7trainable_variables
8	variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
h

Akernel
Bbias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
h

Gkernel
Hbias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
?
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_ratem?m?m?m?$m?%m?/m?0m?;m?<m?Am?Bm?Gm?Hm?v?v?v?v?$v?%v?/v?0v?;v?<v?Av?Bv?Gv?Hv?
f
0
1
2
3
$4
%5
/6
07
;8
<9
A10
B11
G12
H13
?
0
1
2
3
4
5
$6
%7
/8
09
110
211
;12
<13
A14
B15
G16
H17
 
?
Rnon_trainable_variables
trainable_variables
Smetrics
Tlayer_metrics

Ulayers
	variables
regularization_losses
Vlayer_regularization_losses
 
\Z
VARIABLE_VALUEconv3d_42/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_42/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Wnon_trainable_variables
trainable_variables
Xmetrics
Ylayer_metrics

Zlayers
	variables
regularization_losses
[layer_regularization_losses
 
 
 
?
\non_trainable_variables
trainable_variables
]metrics
^layer_metrics

_layers
	variables
regularization_losses
`layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_42/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_42/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_42/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_42/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
2
3
 
?
anon_trainable_variables
 trainable_variables
bmetrics
clayer_metrics

dlayers
!	variables
"regularization_losses
elayer_regularization_losses
\Z
VARIABLE_VALUEconv3d_43/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_43/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
fnon_trainable_variables
&trainable_variables
gmetrics
hlayer_metrics

ilayers
'	variables
(regularization_losses
jlayer_regularization_losses
 
 
 
?
knon_trainable_variables
*trainable_variables
lmetrics
mlayer_metrics

nlayers
+	variables
,regularization_losses
olayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_43/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_43/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_43/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_43/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
12
23
 
?
pnon_trainable_variables
3trainable_variables
qmetrics
rlayer_metrics

slayers
4	variables
5regularization_losses
tlayer_regularization_losses
 
 
 
?
unon_trainable_variables
7trainable_variables
vmetrics
wlayer_metrics

xlayers
8	variables
9regularization_losses
ylayer_regularization_losses
[Y
VARIABLE_VALUEdense_63/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_63/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
?
znon_trainable_variables
=trainable_variables
{metrics
|layer_metrics

}layers
>	variables
?regularization_losses
~layer_regularization_losses
[Y
VARIABLE_VALUEdense_64/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_64/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

A0
B1
 
?
non_trainable_variables
Ctrainable_variables
?metrics
?layer_metrics
?layers
D	variables
Eregularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_65/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_65/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

G0
H1
 
?
?non_trainable_variables
Itrainable_variables
?metrics
?layer_metrics
?layers
J	variables
Kregularization_losses
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

0
1
12
23

?0
?1
 
F
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
0
1
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
10
21
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
VARIABLE_VALUEAdam/conv3d_42/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_42/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_42/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_42/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_43/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_43/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_63/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_63/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_64/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_64/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_65/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_65/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_42/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_42/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_42/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_42/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_43/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_43/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_63/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_63/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_64/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_64/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_65/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_65/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv3d_42_inputPlaceholder*3
_output_shapes!
:?????????222*
dtype0*(
shape:?????????222
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv3d_42_inputconv3d_42/kernelconv3d_42/bias&batch_normalization_42/moving_variancebatch_normalization_42/gamma"batch_normalization_42/moving_meanbatch_normalization_42/betaconv3d_43/kernelconv3d_43/bias&batch_normalization_43/moving_variancebatch_normalization_43/gamma"batch_normalization_43/moving_meanbatch_normalization_43/betadense_63/kerneldense_63/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_25891
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv3d_42/kernel/Read/ReadVariableOp"conv3d_42/bias/Read/ReadVariableOp0batch_normalization_42/gamma/Read/ReadVariableOp/batch_normalization_42/beta/Read/ReadVariableOp6batch_normalization_42/moving_mean/Read/ReadVariableOp:batch_normalization_42/moving_variance/Read/ReadVariableOp$conv3d_43/kernel/Read/ReadVariableOp"conv3d_43/bias/Read/ReadVariableOp0batch_normalization_43/gamma/Read/ReadVariableOp/batch_normalization_43/beta/Read/ReadVariableOp6batch_normalization_43/moving_mean/Read/ReadVariableOp:batch_normalization_43/moving_variance/Read/ReadVariableOp#dense_63/kernel/Read/ReadVariableOp!dense_63/bias/Read/ReadVariableOp#dense_64/kernel/Read/ReadVariableOp!dense_64/bias/Read/ReadVariableOp#dense_65/kernel/Read/ReadVariableOp!dense_65/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv3d_42/kernel/m/Read/ReadVariableOp)Adam/conv3d_42/bias/m/Read/ReadVariableOp7Adam/batch_normalization_42/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_42/beta/m/Read/ReadVariableOp+Adam/conv3d_43/kernel/m/Read/ReadVariableOp)Adam/conv3d_43/bias/m/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_43/beta/m/Read/ReadVariableOp*Adam/dense_63/kernel/m/Read/ReadVariableOp(Adam/dense_63/bias/m/Read/ReadVariableOp*Adam/dense_64/kernel/m/Read/ReadVariableOp(Adam/dense_64/bias/m/Read/ReadVariableOp*Adam/dense_65/kernel/m/Read/ReadVariableOp(Adam/dense_65/bias/m/Read/ReadVariableOp+Adam/conv3d_42/kernel/v/Read/ReadVariableOp)Adam/conv3d_42/bias/v/Read/ReadVariableOp7Adam/batch_normalization_42/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_42/beta/v/Read/ReadVariableOp+Adam/conv3d_43/kernel/v/Read/ReadVariableOp)Adam/conv3d_43/bias/v/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_43/beta/v/Read/ReadVariableOp*Adam/dense_63/kernel/v/Read/ReadVariableOp(Adam/dense_63/bias/v/Read/ReadVariableOp*Adam/dense_64/kernel/v/Read/ReadVariableOp(Adam/dense_64/bias/v/Read/ReadVariableOp*Adam/dense_65/kernel/v/Read/ReadVariableOp(Adam/dense_65/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
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
__inference__traced_save_26770
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_42/kernelconv3d_42/biasbatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_varianceconv3d_43/kernelconv3d_43/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_variancedense_63/kerneldense_63/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv3d_42/kernel/mAdam/conv3d_42/bias/m#Adam/batch_normalization_42/gamma/m"Adam/batch_normalization_42/beta/mAdam/conv3d_43/kernel/mAdam/conv3d_43/bias/m#Adam/batch_normalization_43/gamma/m"Adam/batch_normalization_43/beta/mAdam/dense_63/kernel/mAdam/dense_63/bias/mAdam/dense_64/kernel/mAdam/dense_64/bias/mAdam/dense_65/kernel/mAdam/dense_65/bias/mAdam/conv3d_42/kernel/vAdam/conv3d_42/bias/v#Adam/batch_normalization_42/gamma/v"Adam/batch_normalization_42/beta/vAdam/conv3d_43/kernel/vAdam/conv3d_43/bias/v#Adam/batch_normalization_43/gamma/v"Adam/batch_normalization_43/beta/vAdam/dense_63/kernel/vAdam/dense_63/bias/vAdam/dense_64/kernel/vAdam/dense_64/bias/vAdam/dense_65/kernel/vAdam/dense_65/bias/v*C
Tin<
:28*
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
!__inference__traced_restore_26945??
?
?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26191

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_25115

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
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_25548

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
:?????????2
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
:?????????2
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
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_21_layer_call_fn_26110

inputs%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	?S@

unknown_12:@

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_253542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
)__inference_conv3d_43_layer_call_fn_26351

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_43_layer_call_and_return_conditional_losses_252582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?q
?
__inference__traced_save_26770
file_prefix/
+savev2_conv3d_42_kernel_read_readvariableop-
)savev2_conv3d_42_bias_read_readvariableop;
7savev2_batch_normalization_42_gamma_read_readvariableop:
6savev2_batch_normalization_42_beta_read_readvariableopA
=savev2_batch_normalization_42_moving_mean_read_readvariableopE
Asavev2_batch_normalization_42_moving_variance_read_readvariableop/
+savev2_conv3d_43_kernel_read_readvariableop-
)savev2_conv3d_43_bias_read_readvariableop;
7savev2_batch_normalization_43_gamma_read_readvariableop:
6savev2_batch_normalization_43_beta_read_readvariableopA
=savev2_batch_normalization_43_moving_mean_read_readvariableopE
Asavev2_batch_normalization_43_moving_variance_read_readvariableop.
*savev2_dense_63_kernel_read_readvariableop,
(savev2_dense_63_bias_read_readvariableop.
*savev2_dense_64_kernel_read_readvariableop,
(savev2_dense_64_bias_read_readvariableop.
*savev2_dense_65_kernel_read_readvariableop,
(savev2_dense_65_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv3d_42_kernel_m_read_readvariableop4
0savev2_adam_conv3d_42_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_42_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_42_beta_m_read_readvariableop6
2savev2_adam_conv3d_43_kernel_m_read_readvariableop4
0savev2_adam_conv3d_43_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_m_read_readvariableop5
1savev2_adam_dense_63_kernel_m_read_readvariableop3
/savev2_adam_dense_63_bias_m_read_readvariableop5
1savev2_adam_dense_64_kernel_m_read_readvariableop3
/savev2_adam_dense_64_bias_m_read_readvariableop5
1savev2_adam_dense_65_kernel_m_read_readvariableop3
/savev2_adam_dense_65_bias_m_read_readvariableop6
2savev2_adam_conv3d_42_kernel_v_read_readvariableop4
0savev2_adam_conv3d_42_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_42_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_42_beta_v_read_readvariableop6
2savev2_adam_conv3d_43_kernel_v_read_readvariableop4
0savev2_adam_conv3d_43_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_v_read_readvariableop5
1savev2_adam_dense_63_kernel_v_read_readvariableop3
/savev2_adam_dense_63_bias_v_read_readvariableop5
1savev2_adam_dense_64_kernel_v_read_readvariableop3
/savev2_adam_dense_64_bias_v_read_readvariableop5
1savev2_adam_dense_65_kernel_v_read_readvariableop3
/savev2_adam_dense_65_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv3d_42_kernel_read_readvariableop)savev2_conv3d_42_bias_read_readvariableop7savev2_batch_normalization_42_gamma_read_readvariableop6savev2_batch_normalization_42_beta_read_readvariableop=savev2_batch_normalization_42_moving_mean_read_readvariableopAsavev2_batch_normalization_42_moving_variance_read_readvariableop+savev2_conv3d_43_kernel_read_readvariableop)savev2_conv3d_43_bias_read_readvariableop7savev2_batch_normalization_43_gamma_read_readvariableop6savev2_batch_normalization_43_beta_read_readvariableop=savev2_batch_normalization_43_moving_mean_read_readvariableopAsavev2_batch_normalization_43_moving_variance_read_readvariableop*savev2_dense_63_kernel_read_readvariableop(savev2_dense_63_bias_read_readvariableop*savev2_dense_64_kernel_read_readvariableop(savev2_dense_64_bias_read_readvariableop*savev2_dense_65_kernel_read_readvariableop(savev2_dense_65_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv3d_42_kernel_m_read_readvariableop0savev2_adam_conv3d_42_bias_m_read_readvariableop>savev2_adam_batch_normalization_42_gamma_m_read_readvariableop=savev2_adam_batch_normalization_42_beta_m_read_readvariableop2savev2_adam_conv3d_43_kernel_m_read_readvariableop0savev2_adam_conv3d_43_bias_m_read_readvariableop>savev2_adam_batch_normalization_43_gamma_m_read_readvariableop=savev2_adam_batch_normalization_43_beta_m_read_readvariableop1savev2_adam_dense_63_kernel_m_read_readvariableop/savev2_adam_dense_63_bias_m_read_readvariableop1savev2_adam_dense_64_kernel_m_read_readvariableop/savev2_adam_dense_64_bias_m_read_readvariableop1savev2_adam_dense_65_kernel_m_read_readvariableop/savev2_adam_dense_65_bias_m_read_readvariableop2savev2_adam_conv3d_42_kernel_v_read_readvariableop0savev2_adam_conv3d_42_bias_v_read_readvariableop>savev2_adam_batch_normalization_42_gamma_v_read_readvariableop=savev2_adam_batch_normalization_42_beta_v_read_readvariableop2savev2_adam_conv3d_43_kernel_v_read_readvariableop0savev2_adam_conv3d_43_bias_v_read_readvariableop>savev2_adam_batch_normalization_43_gamma_v_read_readvariableop=savev2_adam_batch_normalization_43_beta_v_read_readvariableop1savev2_adam_dense_63_kernel_v_read_readvariableop/savev2_adam_dense_63_bias_v_read_readvariableop1savev2_adam_dense_64_kernel_v_read_readvariableop/savev2_adam_dense_64_bias_v_read_readvariableop1savev2_adam_dense_65_kernel_v_read_readvariableop/savev2_adam_dense_65_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::::::::	?S@:@:@ : : :: : : : : : : : : :::::::::	?S@:@:@ : : ::::::::::	?S@:@:@ : : :: 2(
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
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?S@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0 ,
*
_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
::%$!

_output_shapes
:	?S@: %

_output_shapes
:@:$& 

_output_shapes

:@ : '

_output_shapes
: :$( 

_output_shapes

: : )

_output_shapes
::0*,
*
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::0.,
*
_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::%2!

_output_shapes
:	?S@: 3

_output_shapes
:@:$4 

_output_shapes

:@ : 5

_output_shapes
: :$6 

_output_shapes

: : 7

_output_shapes
::8

_output_shapes
: 
?
?
-__inference_sequential_21_layer_call_fn_25742
conv3d_42_input%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	?S@

unknown_12:@

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_256622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_42_input
?
?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_24881

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
?
g
K__inference_max_pooling3d_43_layer_call_and_return_conditional_losses_25025

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
?+
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26459

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
:?????????2
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
:?????????2
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
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_64_layer_call_fn_26562

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_253302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

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
?
a
E__inference_flatten_21_layer_call_and_return_conditional_losses_26517

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????)  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????S2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????S2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_21_layer_call_fn_25393
conv3d_42_input%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	?S@

unknown_12:@

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_253542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_42_input
?
?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_25237

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
:?????????2
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
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26405

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
?
?
6__inference_batch_normalization_42_layer_call_fn_26318

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
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_252372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling3d_43_layer_call_fn_25031

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
K__inference_max_pooling3d_43_layer_call_and_return_conditional_losses_250252
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
D__inference_conv3d_43_layer_call_and_return_conditional_losses_26342

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
:?????????*
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
:?????????2	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_43_layer_call_fn_26498

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
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_252842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_42_layer_call_fn_26305

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
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_249412
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
?
?
(__inference_dense_63_layer_call_fn_26542

inputs
unknown:	?S@
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
C__inference_dense_63_layer_call_and_return_conditional_losses_253132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????S: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????S
 
_user_specified_nameinputs
?5
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_25662

inputs-
conv3d_42_25615:
conv3d_42_25617:*
batch_normalization_42_25621:*
batch_normalization_42_25623:*
batch_normalization_42_25625:*
batch_normalization_42_25627:-
conv3d_43_25630:
conv3d_43_25632:*
batch_normalization_43_25636:*
batch_normalization_43_25638:*
batch_normalization_43_25640:*
batch_normalization_43_25642:!
dense_63_25646:	?S@
dense_63_25648:@ 
dense_64_25651:@ 
dense_64_25653:  
dense_65_25656: 
dense_65_25658:
identity??.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?!conv3d_42/StatefulPartitionedCall?!conv3d_43/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall?
!conv3d_42/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_42_25615conv3d_42_25617*
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
D__inference_conv3d_42_layer_call_and_return_conditional_losses_252112#
!conv3d_42/StatefulPartitionedCall?
 max_pooling3d_42/PartitionedCallPartitionedCall*conv3d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_42_layer_call_and_return_conditional_losses_248512"
 max_pooling3d_42/PartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_42/PartitionedCall:output:0batch_normalization_42_25621batch_normalization_42_25623batch_normalization_42_25625batch_normalization_42_25627*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2554820
.batch_normalization_42/StatefulPartitionedCall?
!conv3d_43/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0conv3d_43_25630conv3d_43_25632*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_43_layer_call_and_return_conditional_losses_252582#
!conv3d_43/StatefulPartitionedCall?
 max_pooling3d_43/PartitionedCallPartitionedCall*conv3d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_43_layer_call_and_return_conditional_losses_250252"
 max_pooling3d_43/PartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_43/PartitionedCall:output:0batch_normalization_43_25636batch_normalization_43_25638batch_normalization_43_25640batch_normalization_43_25642*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2547820
.batch_normalization_43/StatefulPartitionedCall?
flatten_21/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????S* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_21_layer_call_and_return_conditional_losses_253002
flatten_21/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0dense_63_25646dense_63_25648*
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
C__inference_dense_63_layer_call_and_return_conditional_losses_253132"
 dense_63/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_25651dense_64_25653*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_253302"
 dense_64/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_25656dense_65_25658*
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
C__inference_dense_65_layer_call_and_return_conditional_losses_253472"
 dense_65/StatefulPartitionedCall?
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall"^conv3d_42/StatefulPartitionedCall"^conv3d_43/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2F
!conv3d_42/StatefulPartitionedCall!conv3d_42/StatefulPartitionedCall2F
!conv3d_43/StatefulPartitionedCall!conv3d_43/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?

?
C__inference_dense_65_layer_call_and_return_conditional_losses_26573

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_25284

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
:?????????2
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
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_43_layer_call_fn_26472

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_250552
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
?
?
6__inference_batch_normalization_43_layer_call_fn_26511

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_254782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_25055

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
?6
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_25792
conv3d_42_input-
conv3d_42_25745:
conv3d_42_25747:*
batch_normalization_42_25751:*
batch_normalization_42_25753:*
batch_normalization_42_25755:*
batch_normalization_42_25757:-
conv3d_43_25760:
conv3d_43_25762:*
batch_normalization_43_25766:*
batch_normalization_43_25768:*
batch_normalization_43_25770:*
batch_normalization_43_25772:!
dense_63_25776:	?S@
dense_63_25778:@ 
dense_64_25781:@ 
dense_64_25783:  
dense_65_25786: 
dense_65_25788:
identity??.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?!conv3d_42/StatefulPartitionedCall?!conv3d_43/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall?
!conv3d_42/StatefulPartitionedCallStatefulPartitionedCallconv3d_42_inputconv3d_42_25745conv3d_42_25747*
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
D__inference_conv3d_42_layer_call_and_return_conditional_losses_252112#
!conv3d_42/StatefulPartitionedCall?
 max_pooling3d_42/PartitionedCallPartitionedCall*conv3d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_42_layer_call_and_return_conditional_losses_248512"
 max_pooling3d_42/PartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_42/PartitionedCall:output:0batch_normalization_42_25751batch_normalization_42_25753batch_normalization_42_25755batch_normalization_42_25757*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2523720
.batch_normalization_42/StatefulPartitionedCall?
!conv3d_43/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0conv3d_43_25760conv3d_43_25762*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_43_layer_call_and_return_conditional_losses_252582#
!conv3d_43/StatefulPartitionedCall?
 max_pooling3d_43/PartitionedCallPartitionedCall*conv3d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_43_layer_call_and_return_conditional_losses_250252"
 max_pooling3d_43/PartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_43/PartitionedCall:output:0batch_normalization_43_25766batch_normalization_43_25768batch_normalization_43_25770batch_normalization_43_25772*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2528420
.batch_normalization_43/StatefulPartitionedCall?
flatten_21/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????S* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_21_layer_call_and_return_conditional_losses_253002
flatten_21/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0dense_63_25776dense_63_25778*
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
C__inference_dense_63_layer_call_and_return_conditional_losses_253132"
 dense_63/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_25781dense_64_25783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_253302"
 dense_64/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_25786dense_65_25788*
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
C__inference_dense_65_layer_call_and_return_conditional_losses_253472"
 dense_65/StatefulPartitionedCall?
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall"^conv3d_42/StatefulPartitionedCall"^conv3d_43/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2F
!conv3d_42/StatefulPartitionedCall!conv3d_42/StatefulPartitionedCall2F
!conv3d_43/StatefulPartitionedCall!conv3d_43/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_42_input
?s
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_25966

inputsF
(conv3d_42_conv3d_readvariableop_resource:7
)conv3d_42_biasadd_readvariableop_resource:F
8batch_normalization_42_batchnorm_readvariableop_resource:J
<batch_normalization_42_batchnorm_mul_readvariableop_resource:H
:batch_normalization_42_batchnorm_readvariableop_1_resource:H
:batch_normalization_42_batchnorm_readvariableop_2_resource:F
(conv3d_43_conv3d_readvariableop_resource:7
)conv3d_43_biasadd_readvariableop_resource:F
8batch_normalization_43_batchnorm_readvariableop_resource:J
<batch_normalization_43_batchnorm_mul_readvariableop_resource:H
:batch_normalization_43_batchnorm_readvariableop_1_resource:H
:batch_normalization_43_batchnorm_readvariableop_2_resource::
'dense_63_matmul_readvariableop_resource:	?S@6
(dense_63_biasadd_readvariableop_resource:@9
'dense_64_matmul_readvariableop_resource:@ 6
(dense_64_biasadd_readvariableop_resource: 9
'dense_65_matmul_readvariableop_resource: 6
(dense_65_biasadd_readvariableop_resource:
identity??/batch_normalization_42/batchnorm/ReadVariableOp?1batch_normalization_42/batchnorm/ReadVariableOp_1?1batch_normalization_42/batchnorm/ReadVariableOp_2?3batch_normalization_42/batchnorm/mul/ReadVariableOp?/batch_normalization_43/batchnorm/ReadVariableOp?1batch_normalization_43/batchnorm/ReadVariableOp_1?1batch_normalization_43/batchnorm/ReadVariableOp_2?3batch_normalization_43/batchnorm/mul/ReadVariableOp? conv3d_42/BiasAdd/ReadVariableOp?conv3d_42/Conv3D/ReadVariableOp? conv3d_43/BiasAdd/ReadVariableOp?conv3d_43/Conv3D/ReadVariableOp?dense_63/BiasAdd/ReadVariableOp?dense_63/MatMul/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?dense_65/BiasAdd/ReadVariableOp?dense_65/MatMul/ReadVariableOp?
conv3d_42/Conv3D/ReadVariableOpReadVariableOp(conv3d_42_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_42/Conv3D/ReadVariableOp?
conv3d_42/Conv3DConv3Dinputs'conv3d_42/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000*
paddingVALID*
strides	
2
conv3d_42/Conv3D?
 conv3d_42/BiasAdd/ReadVariableOpReadVariableOp)conv3d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_42/BiasAdd/ReadVariableOp?
conv3d_42/BiasAddBiasAddconv3d_42/Conv3D:output:0(conv3d_42/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????0002
conv3d_42/BiasAdd?
conv3d_42/SigmoidSigmoidconv3d_42/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002
conv3d_42/Sigmoid?
max_pooling3d_42/MaxPool3D	MaxPool3Dconv3d_42/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_42/MaxPool3D?
/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_42/batchnorm/ReadVariableOp?
&batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_42/batchnorm/add/y?
$batch_normalization_42/batchnorm/addAddV27batch_normalization_42/batchnorm/ReadVariableOp:value:0/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_42/batchnorm/add?
&batch_normalization_42/batchnorm/RsqrtRsqrt(batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_42/batchnorm/Rsqrt?
3batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_42/batchnorm/mul/ReadVariableOp?
$batch_normalization_42/batchnorm/mulMul*batch_normalization_42/batchnorm/Rsqrt:y:0;batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_42/batchnorm/mul?
&batch_normalization_42/batchnorm/mul_1Mul#max_pooling3d_42/MaxPool3D:output:0(batch_normalization_42/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_42/batchnorm/mul_1?
1batch_normalization_42/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_42_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_42/batchnorm/ReadVariableOp_1?
&batch_normalization_42/batchnorm/mul_2Mul9batch_normalization_42/batchnorm/ReadVariableOp_1:value:0(batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_42/batchnorm/mul_2?
1batch_normalization_42/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_42_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_42/batchnorm/ReadVariableOp_2?
$batch_normalization_42/batchnorm/subSub9batch_normalization_42/batchnorm/ReadVariableOp_2:value:0*batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_42/batchnorm/sub?
&batch_normalization_42/batchnorm/add_1AddV2*batch_normalization_42/batchnorm/mul_1:z:0(batch_normalization_42/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_42/batchnorm/add_1?
conv3d_43/Conv3D/ReadVariableOpReadVariableOp(conv3d_43_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_43/Conv3D/ReadVariableOp?
conv3d_43/Conv3DConv3D*batch_normalization_42/batchnorm/add_1:z:0'conv3d_43/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
conv3d_43/Conv3D?
 conv3d_43/BiasAdd/ReadVariableOpReadVariableOp)conv3d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_43/BiasAdd/ReadVariableOp?
conv3d_43/BiasAddBiasAddconv3d_43/Conv3D:output:0(conv3d_43/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2
conv3d_43/BiasAdd?
conv3d_43/SigmoidSigmoidconv3d_43/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
conv3d_43/Sigmoid?
max_pooling3d_43/MaxPool3D	MaxPool3Dconv3d_43/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_43/MaxPool3D?
/batch_normalization_43/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_43_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_43/batchnorm/ReadVariableOp?
&batch_normalization_43/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_43/batchnorm/add/y?
$batch_normalization_43/batchnorm/addAddV27batch_normalization_43/batchnorm/ReadVariableOp:value:0/batch_normalization_43/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_43/batchnorm/add?
&batch_normalization_43/batchnorm/RsqrtRsqrt(batch_normalization_43/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_43/batchnorm/Rsqrt?
3batch_normalization_43/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_43_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_43/batchnorm/mul/ReadVariableOp?
$batch_normalization_43/batchnorm/mulMul*batch_normalization_43/batchnorm/Rsqrt:y:0;batch_normalization_43/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_43/batchnorm/mul?
&batch_normalization_43/batchnorm/mul_1Mul#max_pooling3d_43/MaxPool3D:output:0(batch_normalization_43/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_43/batchnorm/mul_1?
1batch_normalization_43/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_43_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_43/batchnorm/ReadVariableOp_1?
&batch_normalization_43/batchnorm/mul_2Mul9batch_normalization_43/batchnorm/ReadVariableOp_1:value:0(batch_normalization_43/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_43/batchnorm/mul_2?
1batch_normalization_43/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_43_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_43/batchnorm/ReadVariableOp_2?
$batch_normalization_43/batchnorm/subSub9batch_normalization_43/batchnorm/ReadVariableOp_2:value:0*batch_normalization_43/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_43/batchnorm/sub?
&batch_normalization_43/batchnorm/add_1AddV2*batch_normalization_43/batchnorm/mul_1:z:0(batch_normalization_43/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_43/batchnorm/add_1u
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????)  2
flatten_21/Const?
flatten_21/ReshapeReshape*batch_normalization_43/batchnorm/add_1:z:0flatten_21/Const:output:0*
T0*(
_output_shapes
:??????????S2
flatten_21/Reshape?
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes
:	?S@*
dtype02 
dense_63/MatMul/ReadVariableOp?
dense_63/MatMulMatMulflatten_21/Reshape:output:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_63/MatMul?
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_63/BiasAdd/ReadVariableOp?
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_63/BiasAdd|
dense_63/SigmoidSigmoiddense_63/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_63/Sigmoid?
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_64/MatMul/ReadVariableOp?
dense_64/MatMulMatMuldense_63/Sigmoid:y:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_64/MatMul?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_64/BiasAdd|
dense_64/SigmoidSigmoiddense_64/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_64/Sigmoid?
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_65/MatMul/ReadVariableOp?
dense_65/MatMulMatMuldense_64/Sigmoid:y:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_65/MatMul?
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_65/BiasAdd/ReadVariableOp?
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_65/BiasAdd|
dense_65/SigmoidSigmoiddense_65/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_65/Sigmoid?
IdentityIdentitydense_65/Sigmoid:y:00^batch_normalization_42/batchnorm/ReadVariableOp2^batch_normalization_42/batchnorm/ReadVariableOp_12^batch_normalization_42/batchnorm/ReadVariableOp_24^batch_normalization_42/batchnorm/mul/ReadVariableOp0^batch_normalization_43/batchnorm/ReadVariableOp2^batch_normalization_43/batchnorm/ReadVariableOp_12^batch_normalization_43/batchnorm/ReadVariableOp_24^batch_normalization_43/batchnorm/mul/ReadVariableOp!^conv3d_42/BiasAdd/ReadVariableOp ^conv3d_42/Conv3D/ReadVariableOp!^conv3d_43/BiasAdd/ReadVariableOp ^conv3d_43/Conv3D/ReadVariableOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 2b
/batch_normalization_42/batchnorm/ReadVariableOp/batch_normalization_42/batchnorm/ReadVariableOp2f
1batch_normalization_42/batchnorm/ReadVariableOp_11batch_normalization_42/batchnorm/ReadVariableOp_12f
1batch_normalization_42/batchnorm/ReadVariableOp_21batch_normalization_42/batchnorm/ReadVariableOp_22j
3batch_normalization_42/batchnorm/mul/ReadVariableOp3batch_normalization_42/batchnorm/mul/ReadVariableOp2b
/batch_normalization_43/batchnorm/ReadVariableOp/batch_normalization_43/batchnorm/ReadVariableOp2f
1batch_normalization_43/batchnorm/ReadVariableOp_11batch_normalization_43/batchnorm/ReadVariableOp_12f
1batch_normalization_43/batchnorm/ReadVariableOp_21batch_normalization_43/batchnorm/ReadVariableOp_22j
3batch_normalization_43/batchnorm/mul/ReadVariableOp3batch_normalization_43/batchnorm/mul/ReadVariableOp2D
 conv3d_42/BiasAdd/ReadVariableOp conv3d_42/BiasAdd/ReadVariableOp2B
conv3d_42/Conv3D/ReadVariableOpconv3d_42/Conv3D/ReadVariableOp2D
 conv3d_43/BiasAdd/ReadVariableOp conv3d_43/BiasAdd/ReadVariableOp2B
conv3d_43/Conv3D/ReadVariableOpconv3d_43/Conv3D/ReadVariableOp2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?

?
C__inference_dense_65_layer_call_and_return_conditional_losses_25347

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_43_layer_call_fn_26485

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_251152
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
?
?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26245

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
:?????????2
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
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_24845
conv3d_42_inputT
6sequential_21_conv3d_42_conv3d_readvariableop_resource:E
7sequential_21_conv3d_42_biasadd_readvariableop_resource:T
Fsequential_21_batch_normalization_42_batchnorm_readvariableop_resource:X
Jsequential_21_batch_normalization_42_batchnorm_mul_readvariableop_resource:V
Hsequential_21_batch_normalization_42_batchnorm_readvariableop_1_resource:V
Hsequential_21_batch_normalization_42_batchnorm_readvariableop_2_resource:T
6sequential_21_conv3d_43_conv3d_readvariableop_resource:E
7sequential_21_conv3d_43_biasadd_readvariableop_resource:T
Fsequential_21_batch_normalization_43_batchnorm_readvariableop_resource:X
Jsequential_21_batch_normalization_43_batchnorm_mul_readvariableop_resource:V
Hsequential_21_batch_normalization_43_batchnorm_readvariableop_1_resource:V
Hsequential_21_batch_normalization_43_batchnorm_readvariableop_2_resource:H
5sequential_21_dense_63_matmul_readvariableop_resource:	?S@D
6sequential_21_dense_63_biasadd_readvariableop_resource:@G
5sequential_21_dense_64_matmul_readvariableop_resource:@ D
6sequential_21_dense_64_biasadd_readvariableop_resource: G
5sequential_21_dense_65_matmul_readvariableop_resource: D
6sequential_21_dense_65_biasadd_readvariableop_resource:
identity??=sequential_21/batch_normalization_42/batchnorm/ReadVariableOp??sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_1??sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_2?Asequential_21/batch_normalization_42/batchnorm/mul/ReadVariableOp?=sequential_21/batch_normalization_43/batchnorm/ReadVariableOp??sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_1??sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_2?Asequential_21/batch_normalization_43/batchnorm/mul/ReadVariableOp?.sequential_21/conv3d_42/BiasAdd/ReadVariableOp?-sequential_21/conv3d_42/Conv3D/ReadVariableOp?.sequential_21/conv3d_43/BiasAdd/ReadVariableOp?-sequential_21/conv3d_43/Conv3D/ReadVariableOp?-sequential_21/dense_63/BiasAdd/ReadVariableOp?,sequential_21/dense_63/MatMul/ReadVariableOp?-sequential_21/dense_64/BiasAdd/ReadVariableOp?,sequential_21/dense_64/MatMul/ReadVariableOp?-sequential_21/dense_65/BiasAdd/ReadVariableOp?,sequential_21/dense_65/MatMul/ReadVariableOp?
-sequential_21/conv3d_42/Conv3D/ReadVariableOpReadVariableOp6sequential_21_conv3d_42_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02/
-sequential_21/conv3d_42/Conv3D/ReadVariableOp?
sequential_21/conv3d_42/Conv3DConv3Dconv3d_42_input5sequential_21/conv3d_42/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000*
paddingVALID*
strides	
2 
sequential_21/conv3d_42/Conv3D?
.sequential_21/conv3d_42/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_conv3d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_21/conv3d_42/BiasAdd/ReadVariableOp?
sequential_21/conv3d_42/BiasAddBiasAdd'sequential_21/conv3d_42/Conv3D:output:06sequential_21/conv3d_42/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????0002!
sequential_21/conv3d_42/BiasAdd?
sequential_21/conv3d_42/SigmoidSigmoid(sequential_21/conv3d_42/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002!
sequential_21/conv3d_42/Sigmoid?
(sequential_21/max_pooling3d_42/MaxPool3D	MaxPool3D#sequential_21/conv3d_42/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2*
(sequential_21/max_pooling3d_42/MaxPool3D?
=sequential_21/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOpFsequential_21_batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_21/batch_normalization_42/batchnorm/ReadVariableOp?
4sequential_21/batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:26
4sequential_21/batch_normalization_42/batchnorm/add/y?
2sequential_21/batch_normalization_42/batchnorm/addAddV2Esequential_21/batch_normalization_42/batchnorm/ReadVariableOp:value:0=sequential_21/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_21/batch_normalization_42/batchnorm/add?
4sequential_21/batch_normalization_42/batchnorm/RsqrtRsqrt6sequential_21/batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_21/batch_normalization_42/batchnorm/Rsqrt?
Asequential_21/batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_21_batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_21/batch_normalization_42/batchnorm/mul/ReadVariableOp?
2sequential_21/batch_normalization_42/batchnorm/mulMul8sequential_21/batch_normalization_42/batchnorm/Rsqrt:y:0Isequential_21/batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_21/batch_normalization_42/batchnorm/mul?
4sequential_21/batch_normalization_42/batchnorm/mul_1Mul1sequential_21/max_pooling3d_42/MaxPool3D:output:06sequential_21/batch_normalization_42/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????26
4sequential_21/batch_normalization_42/batchnorm/mul_1?
?sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_21_batch_normalization_42_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_1?
4sequential_21/batch_normalization_42/batchnorm/mul_2MulGsequential_21/batch_normalization_42/batchnorm/ReadVariableOp_1:value:06sequential_21/batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_21/batch_normalization_42/batchnorm/mul_2?
?sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_21_batch_normalization_42_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_2?
2sequential_21/batch_normalization_42/batchnorm/subSubGsequential_21/batch_normalization_42/batchnorm/ReadVariableOp_2:value:08sequential_21/batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_21/batch_normalization_42/batchnorm/sub?
4sequential_21/batch_normalization_42/batchnorm/add_1AddV28sequential_21/batch_normalization_42/batchnorm/mul_1:z:06sequential_21/batch_normalization_42/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????26
4sequential_21/batch_normalization_42/batchnorm/add_1?
-sequential_21/conv3d_43/Conv3D/ReadVariableOpReadVariableOp6sequential_21_conv3d_43_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02/
-sequential_21/conv3d_43/Conv3D/ReadVariableOp?
sequential_21/conv3d_43/Conv3DConv3D8sequential_21/batch_normalization_42/batchnorm/add_1:z:05sequential_21/conv3d_43/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2 
sequential_21/conv3d_43/Conv3D?
.sequential_21/conv3d_43/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_conv3d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_21/conv3d_43/BiasAdd/ReadVariableOp?
sequential_21/conv3d_43/BiasAddBiasAdd'sequential_21/conv3d_43/Conv3D:output:06sequential_21/conv3d_43/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2!
sequential_21/conv3d_43/BiasAdd?
sequential_21/conv3d_43/SigmoidSigmoid(sequential_21/conv3d_43/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2!
sequential_21/conv3d_43/Sigmoid?
(sequential_21/max_pooling3d_43/MaxPool3D	MaxPool3D#sequential_21/conv3d_43/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2*
(sequential_21/max_pooling3d_43/MaxPool3D?
=sequential_21/batch_normalization_43/batchnorm/ReadVariableOpReadVariableOpFsequential_21_batch_normalization_43_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_21/batch_normalization_43/batchnorm/ReadVariableOp?
4sequential_21/batch_normalization_43/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:26
4sequential_21/batch_normalization_43/batchnorm/add/y?
2sequential_21/batch_normalization_43/batchnorm/addAddV2Esequential_21/batch_normalization_43/batchnorm/ReadVariableOp:value:0=sequential_21/batch_normalization_43/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_21/batch_normalization_43/batchnorm/add?
4sequential_21/batch_normalization_43/batchnorm/RsqrtRsqrt6sequential_21/batch_normalization_43/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_21/batch_normalization_43/batchnorm/Rsqrt?
Asequential_21/batch_normalization_43/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_21_batch_normalization_43_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_21/batch_normalization_43/batchnorm/mul/ReadVariableOp?
2sequential_21/batch_normalization_43/batchnorm/mulMul8sequential_21/batch_normalization_43/batchnorm/Rsqrt:y:0Isequential_21/batch_normalization_43/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_21/batch_normalization_43/batchnorm/mul?
4sequential_21/batch_normalization_43/batchnorm/mul_1Mul1sequential_21/max_pooling3d_43/MaxPool3D:output:06sequential_21/batch_normalization_43/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????26
4sequential_21/batch_normalization_43/batchnorm/mul_1?
?sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_21_batch_normalization_43_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_1?
4sequential_21/batch_normalization_43/batchnorm/mul_2MulGsequential_21/batch_normalization_43/batchnorm/ReadVariableOp_1:value:06sequential_21/batch_normalization_43/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_21/batch_normalization_43/batchnorm/mul_2?
?sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_21_batch_normalization_43_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_2?
2sequential_21/batch_normalization_43/batchnorm/subSubGsequential_21/batch_normalization_43/batchnorm/ReadVariableOp_2:value:08sequential_21/batch_normalization_43/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_21/batch_normalization_43/batchnorm/sub?
4sequential_21/batch_normalization_43/batchnorm/add_1AddV28sequential_21/batch_normalization_43/batchnorm/mul_1:z:06sequential_21/batch_normalization_43/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????26
4sequential_21/batch_normalization_43/batchnorm/add_1?
sequential_21/flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????)  2 
sequential_21/flatten_21/Const?
 sequential_21/flatten_21/ReshapeReshape8sequential_21/batch_normalization_43/batchnorm/add_1:z:0'sequential_21/flatten_21/Const:output:0*
T0*(
_output_shapes
:??????????S2"
 sequential_21/flatten_21/Reshape?
,sequential_21/dense_63/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_63_matmul_readvariableop_resource*
_output_shapes
:	?S@*
dtype02.
,sequential_21/dense_63/MatMul/ReadVariableOp?
sequential_21/dense_63/MatMulMatMul)sequential_21/flatten_21/Reshape:output:04sequential_21/dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_21/dense_63/MatMul?
-sequential_21/dense_63/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_21/dense_63/BiasAdd/ReadVariableOp?
sequential_21/dense_63/BiasAddBiasAdd'sequential_21/dense_63/MatMul:product:05sequential_21/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_21/dense_63/BiasAdd?
sequential_21/dense_63/SigmoidSigmoid'sequential_21/dense_63/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2 
sequential_21/dense_63/Sigmoid?
,sequential_21/dense_64/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_64_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02.
,sequential_21/dense_64/MatMul/ReadVariableOp?
sequential_21/dense_64/MatMulMatMul"sequential_21/dense_63/Sigmoid:y:04sequential_21/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_21/dense_64/MatMul?
-sequential_21/dense_64/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_21/dense_64/BiasAdd/ReadVariableOp?
sequential_21/dense_64/BiasAddBiasAdd'sequential_21/dense_64/MatMul:product:05sequential_21/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_21/dense_64/BiasAdd?
sequential_21/dense_64/SigmoidSigmoid'sequential_21/dense_64/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2 
sequential_21/dense_64/Sigmoid?
,sequential_21/dense_65/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_65_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_21/dense_65/MatMul/ReadVariableOp?
sequential_21/dense_65/MatMulMatMul"sequential_21/dense_64/Sigmoid:y:04sequential_21/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_21/dense_65/MatMul?
-sequential_21/dense_65/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_21/dense_65/BiasAdd/ReadVariableOp?
sequential_21/dense_65/BiasAddBiasAdd'sequential_21/dense_65/MatMul:product:05sequential_21/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_21/dense_65/BiasAdd?
sequential_21/dense_65/SigmoidSigmoid'sequential_21/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_21/dense_65/Sigmoid?
IdentityIdentity"sequential_21/dense_65/Sigmoid:y:0>^sequential_21/batch_normalization_42/batchnorm/ReadVariableOp@^sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_1@^sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_2B^sequential_21/batch_normalization_42/batchnorm/mul/ReadVariableOp>^sequential_21/batch_normalization_43/batchnorm/ReadVariableOp@^sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_1@^sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_2B^sequential_21/batch_normalization_43/batchnorm/mul/ReadVariableOp/^sequential_21/conv3d_42/BiasAdd/ReadVariableOp.^sequential_21/conv3d_42/Conv3D/ReadVariableOp/^sequential_21/conv3d_43/BiasAdd/ReadVariableOp.^sequential_21/conv3d_43/Conv3D/ReadVariableOp.^sequential_21/dense_63/BiasAdd/ReadVariableOp-^sequential_21/dense_63/MatMul/ReadVariableOp.^sequential_21/dense_64/BiasAdd/ReadVariableOp-^sequential_21/dense_64/MatMul/ReadVariableOp.^sequential_21/dense_65/BiasAdd/ReadVariableOp-^sequential_21/dense_65/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 2~
=sequential_21/batch_normalization_42/batchnorm/ReadVariableOp=sequential_21/batch_normalization_42/batchnorm/ReadVariableOp2?
?sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_1?sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_12?
?sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_2?sequential_21/batch_normalization_42/batchnorm/ReadVariableOp_22?
Asequential_21/batch_normalization_42/batchnorm/mul/ReadVariableOpAsequential_21/batch_normalization_42/batchnorm/mul/ReadVariableOp2~
=sequential_21/batch_normalization_43/batchnorm/ReadVariableOp=sequential_21/batch_normalization_43/batchnorm/ReadVariableOp2?
?sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_1?sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_12?
?sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_2?sequential_21/batch_normalization_43/batchnorm/ReadVariableOp_22?
Asequential_21/batch_normalization_43/batchnorm/mul/ReadVariableOpAsequential_21/batch_normalization_43/batchnorm/mul/ReadVariableOp2`
.sequential_21/conv3d_42/BiasAdd/ReadVariableOp.sequential_21/conv3d_42/BiasAdd/ReadVariableOp2^
-sequential_21/conv3d_42/Conv3D/ReadVariableOp-sequential_21/conv3d_42/Conv3D/ReadVariableOp2`
.sequential_21/conv3d_43/BiasAdd/ReadVariableOp.sequential_21/conv3d_43/BiasAdd/ReadVariableOp2^
-sequential_21/conv3d_43/Conv3D/ReadVariableOp-sequential_21/conv3d_43/Conv3D/ReadVariableOp2^
-sequential_21/dense_63/BiasAdd/ReadVariableOp-sequential_21/dense_63/BiasAdd/ReadVariableOp2\
,sequential_21/dense_63/MatMul/ReadVariableOp,sequential_21/dense_63/MatMul/ReadVariableOp2^
-sequential_21/dense_64/BiasAdd/ReadVariableOp-sequential_21/dense_64/BiasAdd/ReadVariableOp2\
,sequential_21/dense_64/MatMul/ReadVariableOp,sequential_21/dense_64/MatMul/ReadVariableOp2^
-sequential_21/dense_65/BiasAdd/ReadVariableOp-sequential_21/dense_65/BiasAdd/ReadVariableOp2\
,sequential_21/dense_65/MatMul/ReadVariableOp,sequential_21/dense_65/MatMul/ReadVariableOp:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_42_input
?
?
)__inference_conv3d_42_layer_call_fn_26171

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
D__inference_conv3d_42_layer_call_and_return_conditional_losses_252112
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
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26371

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
6__inference_batch_normalization_42_layer_call_fn_26331

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_255482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_24941

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
?
a
E__inference_flatten_21_layer_call_and_return_conditional_losses_25300

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????)  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????S2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????S2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_21_layer_call_fn_26522

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
:??????????S* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_21_layer_call_and_return_conditional_losses_253002
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????S2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv3d_43_layer_call_and_return_conditional_losses_25258

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
:?????????*
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
:?????????2	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv3d_42_layer_call_and_return_conditional_losses_25211

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
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
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
?
?
(__inference_dense_65_layer_call_fn_26582

inputs
unknown: 
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
C__inference_dense_65_layer_call_and_return_conditional_losses_253472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_42_layer_call_fn_26292

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
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_248812
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
?
g
K__inference_max_pooling3d_42_layer_call_and_return_conditional_losses_24851

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
?
L
0__inference_max_pooling3d_42_layer_call_fn_24857

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
K__inference_max_pooling3d_42_layer_call_and_return_conditional_losses_248512
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
?5
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_25842
conv3d_42_input-
conv3d_42_25795:
conv3d_42_25797:*
batch_normalization_42_25801:*
batch_normalization_42_25803:*
batch_normalization_42_25805:*
batch_normalization_42_25807:-
conv3d_43_25810:
conv3d_43_25812:*
batch_normalization_43_25816:*
batch_normalization_43_25818:*
batch_normalization_43_25820:*
batch_normalization_43_25822:!
dense_63_25826:	?S@
dense_63_25828:@ 
dense_64_25831:@ 
dense_64_25833:  
dense_65_25836: 
dense_65_25838:
identity??.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?!conv3d_42/StatefulPartitionedCall?!conv3d_43/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall?
!conv3d_42/StatefulPartitionedCallStatefulPartitionedCallconv3d_42_inputconv3d_42_25795conv3d_42_25797*
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
D__inference_conv3d_42_layer_call_and_return_conditional_losses_252112#
!conv3d_42/StatefulPartitionedCall?
 max_pooling3d_42/PartitionedCallPartitionedCall*conv3d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_42_layer_call_and_return_conditional_losses_248512"
 max_pooling3d_42/PartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_42/PartitionedCall:output:0batch_normalization_42_25801batch_normalization_42_25803batch_normalization_42_25805batch_normalization_42_25807*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2554820
.batch_normalization_42/StatefulPartitionedCall?
!conv3d_43/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0conv3d_43_25810conv3d_43_25812*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_43_layer_call_and_return_conditional_losses_252582#
!conv3d_43/StatefulPartitionedCall?
 max_pooling3d_43/PartitionedCallPartitionedCall*conv3d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_43_layer_call_and_return_conditional_losses_250252"
 max_pooling3d_43/PartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_43/PartitionedCall:output:0batch_normalization_43_25816batch_normalization_43_25818batch_normalization_43_25820batch_normalization_43_25822*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2547820
.batch_normalization_43/StatefulPartitionedCall?
flatten_21/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????S* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_21_layer_call_and_return_conditional_losses_253002
flatten_21/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0dense_63_25826dense_63_25828*
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
C__inference_dense_63_layer_call_and_return_conditional_losses_253132"
 dense_63/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_25831dense_64_25833*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_253302"
 dense_64/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_25836dense_65_25838*
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
C__inference_dense_65_layer_call_and_return_conditional_losses_253472"
 dense_65/StatefulPartitionedCall?
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall"^conv3d_42/StatefulPartitionedCall"^conv3d_43/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2F
!conv3d_42/StatefulPartitionedCall!conv3d_42/StatefulPartitionedCall2F
!conv3d_43/StatefulPartitionedCall!conv3d_43/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_42_input
?5
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_25354

inputs-
conv3d_42_25212:
conv3d_42_25214:*
batch_normalization_42_25238:*
batch_normalization_42_25240:*
batch_normalization_42_25242:*
batch_normalization_42_25244:-
conv3d_43_25259:
conv3d_43_25261:*
batch_normalization_43_25285:*
batch_normalization_43_25287:*
batch_normalization_43_25289:*
batch_normalization_43_25291:!
dense_63_25314:	?S@
dense_63_25316:@ 
dense_64_25331:@ 
dense_64_25333:  
dense_65_25348: 
dense_65_25350:
identity??.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?!conv3d_42/StatefulPartitionedCall?!conv3d_43/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall?
!conv3d_42/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_42_25212conv3d_42_25214*
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
D__inference_conv3d_42_layer_call_and_return_conditional_losses_252112#
!conv3d_42/StatefulPartitionedCall?
 max_pooling3d_42/PartitionedCallPartitionedCall*conv3d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_42_layer_call_and_return_conditional_losses_248512"
 max_pooling3d_42/PartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_42/PartitionedCall:output:0batch_normalization_42_25238batch_normalization_42_25240batch_normalization_42_25242batch_normalization_42_25244*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2523720
.batch_normalization_42/StatefulPartitionedCall?
!conv3d_43/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0conv3d_43_25259conv3d_43_25261*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv3d_43_layer_call_and_return_conditional_losses_252582#
!conv3d_43/StatefulPartitionedCall?
 max_pooling3d_43/PartitionedCallPartitionedCall*conv3d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling3d_43_layer_call_and_return_conditional_losses_250252"
 max_pooling3d_43/PartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_43/PartitionedCall:output:0batch_normalization_43_25285batch_normalization_43_25287batch_normalization_43_25289batch_normalization_43_25291*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2528420
.batch_normalization_43/StatefulPartitionedCall?
flatten_21/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????S* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_21_layer_call_and_return_conditional_losses_253002
flatten_21/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0dense_63_25314dense_63_25316*
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
C__inference_dense_63_layer_call_and_return_conditional_losses_253132"
 dense_63/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_25331dense_64_25333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_253302"
 dense_64/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_25348dense_65_25350*
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
C__inference_dense_65_layer_call_and_return_conditional_losses_253472"
 dense_65/StatefulPartitionedCall?
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall"^conv3d_42/StatefulPartitionedCall"^conv3d_43/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2F
!conv3d_42/StatefulPartitionedCall!conv3d_42/StatefulPartitionedCall2F
!conv3d_43/StatefulPartitionedCall!conv3d_43/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?

?
C__inference_dense_64_layer_call_and_return_conditional_losses_26553

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

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
?
#__inference_signature_wrapper_25891
conv3d_42_input%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	?S@

unknown_12:@

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_248452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????222
)
_user_specified_nameconv3d_42_input
?
?
-__inference_sequential_21_layer_call_fn_26151

inputs%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	?S@

unknown_12:@

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_256622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
D__inference_conv3d_42_layer_call_and_return_conditional_losses_26162

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
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
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
C__inference_dense_63_layer_call_and_return_conditional_losses_25313

inputs1
matmul_readvariableop_resource:	?S@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?S@*
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
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????S: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????S
 
_user_specified_nameinputs
?

?
C__inference_dense_64_layer_call_and_return_conditional_losses_25330

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

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
?,
?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26225

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_25478

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
:?????????2
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
:?????????2
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
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_26069

inputsF
(conv3d_42_conv3d_readvariableop_resource:7
)conv3d_42_biasadd_readvariableop_resource:L
>batch_normalization_42_assignmovingavg_readvariableop_resource:N
@batch_normalization_42_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_42_batchnorm_mul_readvariableop_resource:F
8batch_normalization_42_batchnorm_readvariableop_resource:F
(conv3d_43_conv3d_readvariableop_resource:7
)conv3d_43_biasadd_readvariableop_resource:L
>batch_normalization_43_assignmovingavg_readvariableop_resource:N
@batch_normalization_43_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_43_batchnorm_mul_readvariableop_resource:F
8batch_normalization_43_batchnorm_readvariableop_resource::
'dense_63_matmul_readvariableop_resource:	?S@6
(dense_63_biasadd_readvariableop_resource:@9
'dense_64_matmul_readvariableop_resource:@ 6
(dense_64_biasadd_readvariableop_resource: 9
'dense_65_matmul_readvariableop_resource: 6
(dense_65_biasadd_readvariableop_resource:
identity??&batch_normalization_42/AssignMovingAvg?5batch_normalization_42/AssignMovingAvg/ReadVariableOp?(batch_normalization_42/AssignMovingAvg_1?7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_42/batchnorm/ReadVariableOp?3batch_normalization_42/batchnorm/mul/ReadVariableOp?&batch_normalization_43/AssignMovingAvg?5batch_normalization_43/AssignMovingAvg/ReadVariableOp?(batch_normalization_43/AssignMovingAvg_1?7batch_normalization_43/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_43/batchnorm/ReadVariableOp?3batch_normalization_43/batchnorm/mul/ReadVariableOp? conv3d_42/BiasAdd/ReadVariableOp?conv3d_42/Conv3D/ReadVariableOp? conv3d_43/BiasAdd/ReadVariableOp?conv3d_43/Conv3D/ReadVariableOp?dense_63/BiasAdd/ReadVariableOp?dense_63/MatMul/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?dense_65/BiasAdd/ReadVariableOp?dense_65/MatMul/ReadVariableOp?
conv3d_42/Conv3D/ReadVariableOpReadVariableOp(conv3d_42_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_42/Conv3D/ReadVariableOp?
conv3d_42/Conv3DConv3Dinputs'conv3d_42/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000*
paddingVALID*
strides	
2
conv3d_42/Conv3D?
 conv3d_42/BiasAdd/ReadVariableOpReadVariableOp)conv3d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_42/BiasAdd/ReadVariableOp?
conv3d_42/BiasAddBiasAddconv3d_42/Conv3D:output:0(conv3d_42/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????0002
conv3d_42/BiasAdd?
conv3d_42/SigmoidSigmoidconv3d_42/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????0002
conv3d_42/Sigmoid?
max_pooling3d_42/MaxPool3D	MaxPool3Dconv3d_42/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_42/MaxPool3D?
5batch_normalization_42/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_42/moments/mean/reduction_indices?
#batch_normalization_42/moments/meanMean#max_pooling3d_42/MaxPool3D:output:0>batch_normalization_42/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2%
#batch_normalization_42/moments/mean?
+batch_normalization_42/moments/StopGradientStopGradient,batch_normalization_42/moments/mean:output:0*
T0**
_output_shapes
:2-
+batch_normalization_42/moments/StopGradient?
0batch_normalization_42/moments/SquaredDifferenceSquaredDifference#max_pooling3d_42/MaxPool3D:output:04batch_normalization_42/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????22
0batch_normalization_42/moments/SquaredDifference?
9batch_normalization_42/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_42/moments/variance/reduction_indices?
'batch_normalization_42/moments/varianceMean4batch_normalization_42/moments/SquaredDifference:z:0Bbatch_normalization_42/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2)
'batch_normalization_42/moments/variance?
&batch_normalization_42/moments/SqueezeSqueeze,batch_normalization_42/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_42/moments/Squeeze?
(batch_normalization_42/moments/Squeeze_1Squeeze0batch_normalization_42/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_42/moments/Squeeze_1?
,batch_normalization_42/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_42/AssignMovingAvg/decay?
5batch_normalization_42/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_42_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_42/AssignMovingAvg/ReadVariableOp?
*batch_normalization_42/AssignMovingAvg/subSub=batch_normalization_42/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_42/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_42/AssignMovingAvg/sub?
*batch_normalization_42/AssignMovingAvg/mulMul.batch_normalization_42/AssignMovingAvg/sub:z:05batch_normalization_42/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2,
*batch_normalization_42/AssignMovingAvg/mul?
&batch_normalization_42/AssignMovingAvgAssignSubVariableOp>batch_normalization_42_assignmovingavg_readvariableop_resource.batch_normalization_42/AssignMovingAvg/mul:z:06^batch_normalization_42/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_42/AssignMovingAvg?
.batch_normalization_42/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_42/AssignMovingAvg_1/decay?
7batch_normalization_42/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_42_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_42/AssignMovingAvg_1/subSub?batch_normalization_42/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_42/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_42/AssignMovingAvg_1/sub?
,batch_normalization_42/AssignMovingAvg_1/mulMul0batch_normalization_42/AssignMovingAvg_1/sub:z:07batch_normalization_42/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2.
,batch_normalization_42/AssignMovingAvg_1/mul?
(batch_normalization_42/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_42_assignmovingavg_1_readvariableop_resource0batch_normalization_42/AssignMovingAvg_1/mul:z:08^batch_normalization_42/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_42/AssignMovingAvg_1?
&batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_42/batchnorm/add/y?
$batch_normalization_42/batchnorm/addAddV21batch_normalization_42/moments/Squeeze_1:output:0/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_42/batchnorm/add?
&batch_normalization_42/batchnorm/RsqrtRsqrt(batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_42/batchnorm/Rsqrt?
3batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_42/batchnorm/mul/ReadVariableOp?
$batch_normalization_42/batchnorm/mulMul*batch_normalization_42/batchnorm/Rsqrt:y:0;batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_42/batchnorm/mul?
&batch_normalization_42/batchnorm/mul_1Mul#max_pooling3d_42/MaxPool3D:output:0(batch_normalization_42/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_42/batchnorm/mul_1?
&batch_normalization_42/batchnorm/mul_2Mul/batch_normalization_42/moments/Squeeze:output:0(batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_42/batchnorm/mul_2?
/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_42/batchnorm/ReadVariableOp?
$batch_normalization_42/batchnorm/subSub7batch_normalization_42/batchnorm/ReadVariableOp:value:0*batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_42/batchnorm/sub?
&batch_normalization_42/batchnorm/add_1AddV2*batch_normalization_42/batchnorm/mul_1:z:0(batch_normalization_42/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_42/batchnorm/add_1?
conv3d_43/Conv3D/ReadVariableOpReadVariableOp(conv3d_43_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02!
conv3d_43/Conv3D/ReadVariableOp?
conv3d_43/Conv3DConv3D*batch_normalization_42/batchnorm/add_1:z:0'conv3d_43/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????*
paddingVALID*
strides	
2
conv3d_43/Conv3D?
 conv3d_43/BiasAdd/ReadVariableOpReadVariableOp)conv3d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv3d_43/BiasAdd/ReadVariableOp?
conv3d_43/BiasAddBiasAddconv3d_43/Conv3D:output:0(conv3d_43/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2
conv3d_43/BiasAdd?
conv3d_43/SigmoidSigmoidconv3d_43/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2
conv3d_43/Sigmoid?
max_pooling3d_43/MaxPool3D	MaxPool3Dconv3d_43/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_43/MaxPool3D?
5batch_normalization_43/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_43/moments/mean/reduction_indices?
#batch_normalization_43/moments/meanMean#max_pooling3d_43/MaxPool3D:output:0>batch_normalization_43/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2%
#batch_normalization_43/moments/mean?
+batch_normalization_43/moments/StopGradientStopGradient,batch_normalization_43/moments/mean:output:0*
T0**
_output_shapes
:2-
+batch_normalization_43/moments/StopGradient?
0batch_normalization_43/moments/SquaredDifferenceSquaredDifference#max_pooling3d_43/MaxPool3D:output:04batch_normalization_43/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????22
0batch_normalization_43/moments/SquaredDifference?
9batch_normalization_43/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_43/moments/variance/reduction_indices?
'batch_normalization_43/moments/varianceMean4batch_normalization_43/moments/SquaredDifference:z:0Bbatch_normalization_43/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2)
'batch_normalization_43/moments/variance?
&batch_normalization_43/moments/SqueezeSqueeze,batch_normalization_43/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_43/moments/Squeeze?
(batch_normalization_43/moments/Squeeze_1Squeeze0batch_normalization_43/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_43/moments/Squeeze_1?
,batch_normalization_43/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_43/AssignMovingAvg/decay?
5batch_normalization_43/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_43_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_43/AssignMovingAvg/ReadVariableOp?
*batch_normalization_43/AssignMovingAvg/subSub=batch_normalization_43/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_43/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_43/AssignMovingAvg/sub?
*batch_normalization_43/AssignMovingAvg/mulMul.batch_normalization_43/AssignMovingAvg/sub:z:05batch_normalization_43/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2,
*batch_normalization_43/AssignMovingAvg/mul?
&batch_normalization_43/AssignMovingAvgAssignSubVariableOp>batch_normalization_43_assignmovingavg_readvariableop_resource.batch_normalization_43/AssignMovingAvg/mul:z:06^batch_normalization_43/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_43/AssignMovingAvg?
.batch_normalization_43/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_43/AssignMovingAvg_1/decay?
7batch_normalization_43/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_43_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_43/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_43/AssignMovingAvg_1/subSub?batch_normalization_43/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_43/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_43/AssignMovingAvg_1/sub?
,batch_normalization_43/AssignMovingAvg_1/mulMul0batch_normalization_43/AssignMovingAvg_1/sub:z:07batch_normalization_43/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2.
,batch_normalization_43/AssignMovingAvg_1/mul?
(batch_normalization_43/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_43_assignmovingavg_1_readvariableop_resource0batch_normalization_43/AssignMovingAvg_1/mul:z:08^batch_normalization_43/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_43/AssignMovingAvg_1?
&batch_normalization_43/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_43/batchnorm/add/y?
$batch_normalization_43/batchnorm/addAddV21batch_normalization_43/moments/Squeeze_1:output:0/batch_normalization_43/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_43/batchnorm/add?
&batch_normalization_43/batchnorm/RsqrtRsqrt(batch_normalization_43/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_43/batchnorm/Rsqrt?
3batch_normalization_43/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_43_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_43/batchnorm/mul/ReadVariableOp?
$batch_normalization_43/batchnorm/mulMul*batch_normalization_43/batchnorm/Rsqrt:y:0;batch_normalization_43/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_43/batchnorm/mul?
&batch_normalization_43/batchnorm/mul_1Mul#max_pooling3d_43/MaxPool3D:output:0(batch_normalization_43/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_43/batchnorm/mul_1?
&batch_normalization_43/batchnorm/mul_2Mul/batch_normalization_43/moments/Squeeze:output:0(batch_normalization_43/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_43/batchnorm/mul_2?
/batch_normalization_43/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_43_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_43/batchnorm/ReadVariableOp?
$batch_normalization_43/batchnorm/subSub7batch_normalization_43/batchnorm/ReadVariableOp:value:0*batch_normalization_43/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_43/batchnorm/sub?
&batch_normalization_43/batchnorm/add_1AddV2*batch_normalization_43/batchnorm/mul_1:z:0(batch_normalization_43/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2(
&batch_normalization_43/batchnorm/add_1u
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????)  2
flatten_21/Const?
flatten_21/ReshapeReshape*batch_normalization_43/batchnorm/add_1:z:0flatten_21/Const:output:0*
T0*(
_output_shapes
:??????????S2
flatten_21/Reshape?
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes
:	?S@*
dtype02 
dense_63/MatMul/ReadVariableOp?
dense_63/MatMulMatMulflatten_21/Reshape:output:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_63/MatMul?
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_63/BiasAdd/ReadVariableOp?
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_63/BiasAdd|
dense_63/SigmoidSigmoiddense_63/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_63/Sigmoid?
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_64/MatMul/ReadVariableOp?
dense_64/MatMulMatMuldense_63/Sigmoid:y:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_64/MatMul?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_64/BiasAdd|
dense_64/SigmoidSigmoiddense_64/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_64/Sigmoid?
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_65/MatMul/ReadVariableOp?
dense_65/MatMulMatMuldense_64/Sigmoid:y:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_65/MatMul?
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_65/BiasAdd/ReadVariableOp?
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_65/BiasAdd|
dense_65/SigmoidSigmoiddense_65/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_65/Sigmoid?
IdentityIdentitydense_65/Sigmoid:y:0'^batch_normalization_42/AssignMovingAvg6^batch_normalization_42/AssignMovingAvg/ReadVariableOp)^batch_normalization_42/AssignMovingAvg_18^batch_normalization_42/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_42/batchnorm/ReadVariableOp4^batch_normalization_42/batchnorm/mul/ReadVariableOp'^batch_normalization_43/AssignMovingAvg6^batch_normalization_43/AssignMovingAvg/ReadVariableOp)^batch_normalization_43/AssignMovingAvg_18^batch_normalization_43/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_43/batchnorm/ReadVariableOp4^batch_normalization_43/batchnorm/mul/ReadVariableOp!^conv3d_42/BiasAdd/ReadVariableOp ^conv3d_42/Conv3D/ReadVariableOp!^conv3d_43/BiasAdd/ReadVariableOp ^conv3d_43/Conv3D/ReadVariableOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:?????????222: : : : : : : : : : : : : : : : : : 2P
&batch_normalization_42/AssignMovingAvg&batch_normalization_42/AssignMovingAvg2n
5batch_normalization_42/AssignMovingAvg/ReadVariableOp5batch_normalization_42/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_42/AssignMovingAvg_1(batch_normalization_42/AssignMovingAvg_12r
7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_42/batchnorm/ReadVariableOp/batch_normalization_42/batchnorm/ReadVariableOp2j
3batch_normalization_42/batchnorm/mul/ReadVariableOp3batch_normalization_42/batchnorm/mul/ReadVariableOp2P
&batch_normalization_43/AssignMovingAvg&batch_normalization_43/AssignMovingAvg2n
5batch_normalization_43/AssignMovingAvg/ReadVariableOp5batch_normalization_43/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_43/AssignMovingAvg_1(batch_normalization_43/AssignMovingAvg_12r
7batch_normalization_43/AssignMovingAvg_1/ReadVariableOp7batch_normalization_43/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_43/batchnorm/ReadVariableOp/batch_normalization_43/batchnorm/ReadVariableOp2j
3batch_normalization_43/batchnorm/mul/ReadVariableOp3batch_normalization_43/batchnorm/mul/ReadVariableOp2D
 conv3d_42/BiasAdd/ReadVariableOp conv3d_42/BiasAdd/ReadVariableOp2B
conv3d_42/Conv3D/ReadVariableOpconv3d_42/Conv3D/ReadVariableOp2D
 conv3d_43/BiasAdd/ReadVariableOp conv3d_43/BiasAdd/ReadVariableOp2B
conv3d_43/Conv3D/ReadVariableOpconv3d_43/Conv3D/ReadVariableOp2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26425

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
:?????????2
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
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_63_layer_call_and_return_conditional_losses_26533

inputs1
matmul_readvariableop_resource:	?S@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?S@*
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
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????S: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????S
 
_user_specified_nameinputs
??
?#
!__inference__traced_restore_26945
file_prefix?
!assignvariableop_conv3d_42_kernel:/
!assignvariableop_1_conv3d_42_bias:=
/assignvariableop_2_batch_normalization_42_gamma:<
.assignvariableop_3_batch_normalization_42_beta:C
5assignvariableop_4_batch_normalization_42_moving_mean:G
9assignvariableop_5_batch_normalization_42_moving_variance:A
#assignvariableop_6_conv3d_43_kernel:/
!assignvariableop_7_conv3d_43_bias:=
/assignvariableop_8_batch_normalization_43_gamma:<
.assignvariableop_9_batch_normalization_43_beta:D
6assignvariableop_10_batch_normalization_43_moving_mean:H
:assignvariableop_11_batch_normalization_43_moving_variance:6
#assignvariableop_12_dense_63_kernel:	?S@/
!assignvariableop_13_dense_63_bias:@5
#assignvariableop_14_dense_64_kernel:@ /
!assignvariableop_15_dense_64_bias: 5
#assignvariableop_16_dense_65_kernel: /
!assignvariableop_17_dense_65_bias:'
assignvariableop_18_adam_iter:	 )
assignvariableop_19_adam_beta_1: )
assignvariableop_20_adam_beta_2: (
assignvariableop_21_adam_decay: 0
&assignvariableop_22_adam_learning_rate: #
assignvariableop_23_total: #
assignvariableop_24_count: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: I
+assignvariableop_27_adam_conv3d_42_kernel_m:7
)assignvariableop_28_adam_conv3d_42_bias_m:E
7assignvariableop_29_adam_batch_normalization_42_gamma_m:D
6assignvariableop_30_adam_batch_normalization_42_beta_m:I
+assignvariableop_31_adam_conv3d_43_kernel_m:7
)assignvariableop_32_adam_conv3d_43_bias_m:E
7assignvariableop_33_adam_batch_normalization_43_gamma_m:D
6assignvariableop_34_adam_batch_normalization_43_beta_m:=
*assignvariableop_35_adam_dense_63_kernel_m:	?S@6
(assignvariableop_36_adam_dense_63_bias_m:@<
*assignvariableop_37_adam_dense_64_kernel_m:@ 6
(assignvariableop_38_adam_dense_64_bias_m: <
*assignvariableop_39_adam_dense_65_kernel_m: 6
(assignvariableop_40_adam_dense_65_bias_m:I
+assignvariableop_41_adam_conv3d_42_kernel_v:7
)assignvariableop_42_adam_conv3d_42_bias_v:E
7assignvariableop_43_adam_batch_normalization_42_gamma_v:D
6assignvariableop_44_adam_batch_normalization_42_beta_v:I
+assignvariableop_45_adam_conv3d_43_kernel_v:7
)assignvariableop_46_adam_conv3d_43_bias_v:E
7assignvariableop_47_adam_batch_normalization_43_gamma_v:D
6assignvariableop_48_adam_batch_normalization_43_beta_v:=
*assignvariableop_49_adam_dense_63_kernel_v:	?S@6
(assignvariableop_50_adam_dense_63_bias_v:@<
*assignvariableop_51_adam_dense_64_kernel_v:@ 6
(assignvariableop_52_adam_dense_64_bias_v: <
*assignvariableop_53_adam_dense_65_kernel_v: 6
(assignvariableop_54_adam_dense_65_bias_v:
identity_56??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv3d_42_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv3d_42_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_42_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_42_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_42_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_42_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv3d_43_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv3d_43_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_43_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_43_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_43_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_43_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_63_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_63_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_64_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_64_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_65_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_65_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv3d_42_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv3d_42_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_batch_normalization_42_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_batch_normalization_42_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv3d_43_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv3d_43_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_batch_normalization_43_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_43_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_63_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_63_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_64_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_64_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_65_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_65_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv3d_42_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv3d_42_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_42_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_42_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv3d_43_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv3d_43_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_batch_normalization_43_gamma_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_batch_normalization_43_beta_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_63_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_63_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_64_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_64_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_65_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_65_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55?

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_56"#
identity_56Identity_56:output:0*?
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?+
?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26279

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
:?????????2
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
:?????????2
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
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
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
conv3d_42_inputD
!serving_default_conv3d_42_input:0?????????222<
dense_650
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?^
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?Z
_tf_keras_sequential?Z{"name": "sequential_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_21", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_42_input"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_42", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv3D", "config": {"name": "conv3d_43", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_43", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}, "float32", "conv3d_42_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_21", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_42_input"}, "shared_object_id": 0}, {"class_name": "Conv3D", "config": {"name": "conv3d_42", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 4}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9}, {"class_name": "Conv3D", "config": {"name": "conv3d_43", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_43", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 31}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.05000000074505806, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv3d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_42", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 4]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling3d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 32}}
?

axis
	gamma
beta
moving_mean
moving_variance
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_42", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 8}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 8]}}
?


$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3d_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_43", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 8}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 8]}}
?
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling3d_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_43", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 35}}
?

.axis
	/gamma
0beta
1moving_mean
2moving_variance
3trainable_variables
4	variables
5regularization_losses
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_43", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 8}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 11, 8]}}
?
7trainable_variables
8	variables
9regularization_losses
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 37}}
?

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10648}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10648]}}
?

Akernel
Bbias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

Gkernel
Hbias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_ratem?m?m?m?$m?%m?/m?0m?;m?<m?Am?Bm?Gm?Hm?v?v?v?v?$v?%v?/v?0v?;v?<v?Av?Bv?Gv?Hv?"
	optimizer
?
0
1
2
3
$4
%5
/6
07
;8
<9
A10
B11
G12
H13"
trackable_list_wrapper
?
0
1
2
3
4
5
$6
%7
/8
09
110
211
;12
<13
A14
B15
G16
H17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables
trainable_variables
Smetrics
Tlayer_metrics

Ulayers
	variables
regularization_losses
Vlayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
.:,2conv3d_42/kernel
:2conv3d_42/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables
trainable_variables
Xmetrics
Ylayer_metrics

Zlayers
	variables
regularization_losses
[layer_regularization_losses
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
\non_trainable_variables
trainable_variables
]metrics
^layer_metrics

_layers
	variables
regularization_losses
`layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_42/gamma
):'2batch_normalization_42/beta
2:0 (2"batch_normalization_42/moving_mean
6:4 (2&batch_normalization_42/moving_variance
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables
 trainable_variables
bmetrics
clayer_metrics

dlayers
!	variables
"regularization_losses
elayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_43/kernel
:2conv3d_43/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables
&trainable_variables
gmetrics
hlayer_metrics

ilayers
'	variables
(regularization_losses
jlayer_regularization_losses
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
knon_trainable_variables
*trainable_variables
lmetrics
mlayer_metrics

nlayers
+	variables
,regularization_losses
olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_43/gamma
):'2batch_normalization_43/beta
2:0 (2"batch_normalization_43/moving_mean
6:4 (2&batch_normalization_43/moving_variance
.
/0
01"
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables
3trainable_variables
qmetrics
rlayer_metrics

slayers
4	variables
5regularization_losses
tlayer_regularization_losses
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
unon_trainable_variables
7trainable_variables
vmetrics
wlayer_metrics

xlayers
8	variables
9regularization_losses
ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?S@2dense_63/kernel
:@2dense_63/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables
=trainable_variables
{metrics
|layer_metrics

}layers
>	variables
?regularization_losses
~layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@ 2dense_64/kernel
: 2dense_64/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
Ctrainable_variables
?metrics
?layer_metrics
?layers
D	variables
Eregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_65/kernel
:2dense_65/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Itrainable_variables
?metrics
?layer_metrics
?layers
J	variables
Kregularization_losses
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
0
1
12
23"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
f
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
9"
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
0
1"
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
10
21"
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
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 41}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 31}
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
3:12Adam/conv3d_42/kernel/m
!:2Adam/conv3d_42/bias/m
/:-2#Adam/batch_normalization_42/gamma/m
.:,2"Adam/batch_normalization_42/beta/m
3:12Adam/conv3d_43/kernel/m
!:2Adam/conv3d_43/bias/m
/:-2#Adam/batch_normalization_43/gamma/m
.:,2"Adam/batch_normalization_43/beta/m
':%	?S@2Adam/dense_63/kernel/m
 :@2Adam/dense_63/bias/m
&:$@ 2Adam/dense_64/kernel/m
 : 2Adam/dense_64/bias/m
&:$ 2Adam/dense_65/kernel/m
 :2Adam/dense_65/bias/m
3:12Adam/conv3d_42/kernel/v
!:2Adam/conv3d_42/bias/v
/:-2#Adam/batch_normalization_42/gamma/v
.:,2"Adam/batch_normalization_42/beta/v
3:12Adam/conv3d_43/kernel/v
!:2Adam/conv3d_43/bias/v
/:-2#Adam/batch_normalization_43/gamma/v
.:,2"Adam/batch_normalization_43/beta/v
':%	?S@2Adam/dense_63/kernel/v
 :@2Adam/dense_63/bias/v
&:$@ 2Adam/dense_64/kernel/v
 : 2Adam/dense_64/bias/v
&:$ 2Adam/dense_65/kernel/v
 :2Adam/dense_65/bias/v
?2?
H__inference_sequential_21_layer_call_and_return_conditional_losses_25966
H__inference_sequential_21_layer_call_and_return_conditional_losses_26069
H__inference_sequential_21_layer_call_and_return_conditional_losses_25792
H__inference_sequential_21_layer_call_and_return_conditional_losses_25842?
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
-__inference_sequential_21_layer_call_fn_25393
-__inference_sequential_21_layer_call_fn_26110
-__inference_sequential_21_layer_call_fn_26151
-__inference_sequential_21_layer_call_fn_25742?
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
 __inference__wrapped_model_24845?
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
conv3d_42_input?????????222
?2?
D__inference_conv3d_42_layer_call_and_return_conditional_losses_26162?
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
)__inference_conv3d_42_layer_call_fn_26171?
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
K__inference_max_pooling3d_42_layer_call_and_return_conditional_losses_24851?
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
0__inference_max_pooling3d_42_layer_call_fn_24857?
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
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26191
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26225
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26245
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26279?
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
6__inference_batch_normalization_42_layer_call_fn_26292
6__inference_batch_normalization_42_layer_call_fn_26305
6__inference_batch_normalization_42_layer_call_fn_26318
6__inference_batch_normalization_42_layer_call_fn_26331?
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
D__inference_conv3d_43_layer_call_and_return_conditional_losses_26342?
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
)__inference_conv3d_43_layer_call_fn_26351?
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
K__inference_max_pooling3d_43_layer_call_and_return_conditional_losses_25025?
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
0__inference_max_pooling3d_43_layer_call_fn_25031?
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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26371
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26405
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26425
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26459?
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
6__inference_batch_normalization_43_layer_call_fn_26472
6__inference_batch_normalization_43_layer_call_fn_26485
6__inference_batch_normalization_43_layer_call_fn_26498
6__inference_batch_normalization_43_layer_call_fn_26511?
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
E__inference_flatten_21_layer_call_and_return_conditional_losses_26517?
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
*__inference_flatten_21_layer_call_fn_26522?
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
C__inference_dense_63_layer_call_and_return_conditional_losses_26533?
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
(__inference_dense_63_layer_call_fn_26542?
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
C__inference_dense_64_layer_call_and_return_conditional_losses_26553?
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
(__inference_dense_64_layer_call_fn_26562?
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
C__inference_dense_65_layer_call_and_return_conditional_losses_26573?
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
(__inference_dense_65_layer_call_fn_26582?
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
#__inference_signature_wrapper_25891conv3d_42_input"?
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
 __inference__wrapped_model_24845?$%2/10;<ABGHD?A
:?7
5?2
conv3d_42_input?????????222
? "3?0
.
dense_65"?
dense_65??????????
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26191?Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26225?Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26245z??<
5?2
,?)
inputs?????????
p 
? "1?.
'?$
0?????????
? ?
Q__inference_batch_normalization_42_layer_call_and_return_conditional_losses_26279z??<
5?2
,?)
inputs?????????
p
? "1?.
'?$
0?????????
? ?
6__inference_batch_normalization_42_layer_call_fn_26292?Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "??<8?????????????????????????????????????
6__inference_batch_normalization_42_layer_call_fn_26305?Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "??<8?????????????????????????????????????
6__inference_batch_normalization_42_layer_call_fn_26318m??<
5?2
,?)
inputs?????????
p 
? "$?!??????????
6__inference_batch_normalization_42_layer_call_fn_26331m??<
5?2
,?)
inputs?????????
p
? "$?!??????????
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26371?2/10Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26405?12/0Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26425z2/10??<
5?2
,?)
inputs?????????
p 
? "1?.
'?$
0?????????
? ?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_26459z12/0??<
5?2
,?)
inputs?????????
p
? "1?.
'?$
0?????????
? ?
6__inference_batch_normalization_43_layer_call_fn_26472?2/10Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "??<8?????????????????????????????????????
6__inference_batch_normalization_43_layer_call_fn_26485?12/0Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "??<8?????????????????????????????????????
6__inference_batch_normalization_43_layer_call_fn_26498m2/10??<
5?2
,?)
inputs?????????
p 
? "$?!??????????
6__inference_batch_normalization_43_layer_call_fn_26511m12/0??<
5?2
,?)
inputs?????????
p
? "$?!??????????
D__inference_conv3d_42_layer_call_and_return_conditional_losses_26162t;?8
1?.
,?)
inputs?????????222
? "1?.
'?$
0?????????000
? ?
)__inference_conv3d_42_layer_call_fn_26171g;?8
1?.
,?)
inputs?????????222
? "$?!?????????000?
D__inference_conv3d_43_layer_call_and_return_conditional_losses_26342t$%;?8
1?.
,?)
inputs?????????
? "1?.
'?$
0?????????
? ?
)__inference_conv3d_43_layer_call_fn_26351g$%;?8
1?.
,?)
inputs?????????
? "$?!??????????
C__inference_dense_63_layer_call_and_return_conditional_losses_26533];<0?-
&?#
!?
inputs??????????S
? "%?"
?
0?????????@
? |
(__inference_dense_63_layer_call_fn_26542P;<0?-
&?#
!?
inputs??????????S
? "??????????@?
C__inference_dense_64_layer_call_and_return_conditional_losses_26553\AB/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? {
(__inference_dense_64_layer_call_fn_26562OAB/?,
%?"
 ?
inputs?????????@
? "?????????? ?
C__inference_dense_65_layer_call_and_return_conditional_losses_26573\GH/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_65_layer_call_fn_26582OGH/?,
%?"
 ?
inputs????????? 
? "???????????
E__inference_flatten_21_layer_call_and_return_conditional_losses_26517e;?8
1?.
,?)
inputs?????????
? "&?#
?
0??????????S
? ?
*__inference_flatten_21_layer_call_fn_26522X;?8
1?.
,?)
inputs?????????
? "???????????S?
K__inference_max_pooling3d_42_layer_call_and_return_conditional_losses_24851?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
0__inference_max_pooling3d_42_layer_call_fn_24857?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
K__inference_max_pooling3d_43_layer_call_and_return_conditional_losses_25025?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
0__inference_max_pooling3d_43_layer_call_fn_25031?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
H__inference_sequential_21_layer_call_and_return_conditional_losses_25792?$%2/10;<ABGHL?I
B??
5?2
conv3d_42_input?????????222
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_21_layer_call_and_return_conditional_losses_25842?$%12/0;<ABGHL?I
B??
5?2
conv3d_42_input?????????222
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_21_layer_call_and_return_conditional_losses_25966?$%2/10;<ABGHC?@
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
H__inference_sequential_21_layer_call_and_return_conditional_losses_26069?$%12/0;<ABGHC?@
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
-__inference_sequential_21_layer_call_fn_25393|$%2/10;<ABGHL?I
B??
5?2
conv3d_42_input?????????222
p 

 
? "???????????
-__inference_sequential_21_layer_call_fn_25742|$%12/0;<ABGHL?I
B??
5?2
conv3d_42_input?????????222
p

 
? "???????????
-__inference_sequential_21_layer_call_fn_26110s$%2/10;<ABGHC?@
9?6
,?)
inputs?????????222
p 

 
? "???????????
-__inference_sequential_21_layer_call_fn_26151s$%12/0;<ABGHC?@
9?6
,?)
inputs?????????222
p

 
? "???????????
#__inference_signature_wrapper_25891?$%2/10;<ABGHW?T
? 
M?J
H
conv3d_42_input5?2
conv3d_42_input?????????222"3?0
.
dense_65"?
dense_65?????????