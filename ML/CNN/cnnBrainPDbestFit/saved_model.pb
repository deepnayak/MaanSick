??
??
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
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
conv3d_106/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv3d_106/kernel
?
%conv3d_106/kernel/Read/ReadVariableOpReadVariableOpconv3d_106/kernel**
_output_shapes
:*
dtype0
v
conv3d_106/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_106/bias
o
#conv3d_106/bias/Read/ReadVariableOpReadVariableOpconv3d_106/bias*
_output_shapes
:*
dtype0
?
batch_normalization_49/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_49/gamma
?
0batch_normalization_49/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_49/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_49/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_49/beta
?
/batch_normalization_49/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_49/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_49/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_49/moving_mean
?
6batch_normalization_49/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_49/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_49/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_49/moving_variance
?
:batch_normalization_49/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_49/moving_variance*
_output_shapes
:*
dtype0
?
conv3d_107/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv3d_107/kernel
?
%conv3d_107/kernel/Read/ReadVariableOpReadVariableOpconv3d_107/kernel**
_output_shapes
:*
dtype0
v
conv3d_107/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_107/bias
o
#conv3d_107/bias/Read/ReadVariableOpReadVariableOpconv3d_107/bias*
_output_shapes
:*
dtype0
?
conv3d_108/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv3d_108/kernel
?
%conv3d_108/kernel/Read/ReadVariableOpReadVariableOpconv3d_108/kernel**
_output_shapes
:*
dtype0
v
conv3d_108/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_108/bias
o
#conv3d_108/bias/Read/ReadVariableOpReadVariableOpconv3d_108/bias*
_output_shapes
:*
dtype0
?
batch_normalization_50/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_50/gamma
?
0batch_normalization_50/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_50/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_50/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_50/beta
?
/batch_normalization_50/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_50/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_50/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_50/moving_mean
?
6batch_normalization_50/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_50/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_50/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_50/moving_variance
?
:batch_normalization_50/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_50/moving_variance*
_output_shapes
:*
dtype0
?
conv3d_109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv3d_109/kernel
?
%conv3d_109/kernel/Read/ReadVariableOpReadVariableOpconv3d_109/kernel**
_output_shapes
:*
dtype0
v
conv3d_109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_109/bias
o
#conv3d_109/bias/Read/ReadVariableOpReadVariableOpconv3d_109/bias*
_output_shapes
:*
dtype0
?
batch_normalization_51/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_51/gamma
?
0batch_normalization_51/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_51/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_51/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_51/beta
?
/batch_normalization_51/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_51/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_51/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_51/moving_mean
?
6batch_normalization_51/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_51/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_51/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_51/moving_variance
?
:batch_normalization_51/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_51/moving_variance*
_output_shapes
:*
dtype0
?
conv3d_110/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv3d_110/kernel
?
%conv3d_110/kernel/Read/ReadVariableOpReadVariableOpconv3d_110/kernel**
_output_shapes
:*
dtype0
v
conv3d_110/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_110/bias
o
#conv3d_110/bias/Read/ReadVariableOpReadVariableOpconv3d_110/bias*
_output_shapes
:*
dtype0
|
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@* 
shared_namedense_61/kernel
u
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel* 
_output_shapes
:
??@*
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes
:@*
dtype0
z
dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_62/kernel
s
#dense_62/kernel/Read/ReadVariableOpReadVariableOpdense_62/kernel*
_output_shapes

:@*
dtype0
r
dense_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_62/bias
k
!dense_62/bias/Read/ReadVariableOpReadVariableOpdense_62/bias*
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
Adam/conv3d_106/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_106/kernel/m
?
,Adam/conv3d_106/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_106/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_106/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_106/bias/m
}
*Adam/conv3d_106/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_106/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_49/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_49/gamma/m
?
7Adam/batch_normalization_49/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_49/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_49/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_49/beta/m
?
6Adam/batch_normalization_49/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_49/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_107/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_107/kernel/m
?
,Adam/conv3d_107/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_107/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_107/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_107/bias/m
}
*Adam/conv3d_107/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_107/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_108/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_108/kernel/m
?
,Adam/conv3d_108/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_108/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_108/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_108/bias/m
}
*Adam/conv3d_108/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_108/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_50/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_50/gamma/m
?
7Adam/batch_normalization_50/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_50/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_50/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_50/beta/m
?
6Adam/batch_normalization_50/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_50/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_109/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_109/kernel/m
?
,Adam/conv3d_109/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_109/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_109/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_109/bias/m
}
*Adam/conv3d_109/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_109/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_51/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_51/gamma/m
?
7Adam/batch_normalization_51/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_51/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_51/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_51/beta/m
?
6Adam/batch_normalization_51/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_51/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_110/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_110/kernel/m
?
,Adam/conv3d_110/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_110/kernel/m**
_output_shapes
:*
dtype0
?
Adam/conv3d_110/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_110/bias/m
}
*Adam/conv3d_110/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_110/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*'
shared_nameAdam/dense_61/kernel/m
?
*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m* 
_output_shapes
:
??@*
dtype0
?
Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_61/bias/m
y
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_62/kernel/m
?
*Adam/dense_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_62/bias/m
y
(Adam/dense_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_106/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_106/kernel/v
?
,Adam/conv3d_106/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_106/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_106/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_106/bias/v
}
*Adam/conv3d_106/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_106/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_49/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_49/gamma/v
?
7Adam/batch_normalization_49/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_49/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_49/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_49/beta/v
?
6Adam/batch_normalization_49/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_49/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_107/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_107/kernel/v
?
,Adam/conv3d_107/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_107/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_107/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_107/bias/v
}
*Adam/conv3d_107/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_107/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_108/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_108/kernel/v
?
,Adam/conv3d_108/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_108/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_108/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_108/bias/v
}
*Adam/conv3d_108/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_108/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_50/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_50/gamma/v
?
7Adam/batch_normalization_50/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_50/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_50/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_50/beta/v
?
6Adam/batch_normalization_50/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_50/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_109/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_109/kernel/v
?
,Adam/conv3d_109/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_109/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_109/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_109/bias/v
}
*Adam/conv3d_109/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_109/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_51/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_51/gamma/v
?
7Adam/batch_normalization_51/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_51/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_51/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_51/beta/v
?
6Adam/batch_normalization_51/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_51/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_110/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv3d_110/kernel/v
?
,Adam/conv3d_110/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_110/kernel/v**
_output_shapes
:*
dtype0
?
Adam/conv3d_110/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_110/bias/v
}
*Adam/conv3d_110/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_110/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*'
shared_nameAdam/dense_61/kernel/v
?
*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v* 
_output_shapes
:
??@*
dtype0
?
Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_61/bias/v
y
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_62/kernel/v
?
*Adam/dense_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_62/bias/v
y
(Adam/dense_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?w
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?w
value?wB?w B?w
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
R
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:regularization_losses
;trainable_variables
<	keras_api
R
=	variables
>regularization_losses
?trainable_variables
@	keras_api
h

Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
?
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
R
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
h

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
R
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
h

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
h

dkernel
ebias
f	variables
gregularization_losses
htrainable_variables
i	keras_api
?
jiter

kbeta_1

lbeta_2
	mdecay
nlearning_ratem?m?m?m?$m?%m?.m?/m?5m?6m?Am?Bm?Hm?Im?Tm?Um?^m?_m?dm?em?v?v?v?v?$v?%v?.v?/v?5v?6v?Av?Bv?Hv?Iv?Tv?Uv?^v?_v?dv?ev?
?
0
1
2
3
4
5
$6
%7
.8
/9
510
611
712
813
A14
B15
H16
I17
J18
K19
T20
U21
^22
_23
d24
e25
 
?
0
1
2
3
$4
%5
.6
/7
58
69
A10
B11
H12
I13
T14
U15
^16
_17
d18
e19
?
olayer_regularization_losses
player_metrics
qnon_trainable_variables
	variables
regularization_losses

rlayers
trainable_variables
smetrics
 
][
VARIABLE_VALUEconv3d_106/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_106/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
tlayer_regularization_losses
ulayer_metrics
vnon_trainable_variables
	variables
regularization_losses

wlayers
trainable_variables
xmetrics
 
ge
VARIABLE_VALUEbatch_normalization_49/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_49/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_49/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_49/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
?
ylayer_regularization_losses
zlayer_metrics
{non_trainable_variables
 	variables
!regularization_losses

|layers
"trainable_variables
}metrics
][
VARIABLE_VALUEconv3d_107/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_107/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
?
~layer_regularization_losses
layer_metrics
?non_trainable_variables
&	variables
'regularization_losses
?layers
(trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
*	variables
+regularization_losses
?layers
,trainable_variables
?metrics
][
VARIABLE_VALUEconv3d_108/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_108/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
0	variables
1regularization_losses
?layers
2trainable_variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_50/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_50/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_50/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_50/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

50
61
72
83
 

50
61
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
9	variables
:regularization_losses
?layers
;trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
=	variables
>regularization_losses
?layers
?trainable_variables
?metrics
][
VARIABLE_VALUEconv3d_109/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_109/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
 

A0
B1
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
C	variables
Dregularization_losses
?layers
Etrainable_variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_51/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_51/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_51/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_51/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
J2
K3
 

H0
I1
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
L	variables
Mregularization_losses
?layers
Ntrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
P	variables
Qregularization_losses
?layers
Rtrainable_variables
?metrics
][
VARIABLE_VALUEconv3d_110/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_110/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
 

T0
U1
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
V	variables
Wregularization_losses
?layers
Xtrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Z	variables
[regularization_losses
?layers
\trainable_variables
?metrics
[Y
VARIABLE_VALUEdense_61/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_61/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
`	variables
aregularization_losses
?layers
btrainable_variables
?metrics
[Y
VARIABLE_VALUEdense_62/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_62/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

d0
e1
 

d0
e1
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
f	variables
gregularization_losses
?layers
htrainable_variables
?metrics
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
 
 
*
0
1
72
83
J4
K5
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
9
10
11
12
13

?0
?1
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
 
 
 
 
 

70
81
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
J0
K1
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
?~
VARIABLE_VALUEAdam/conv3d_106/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_106/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_49/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_49/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_107/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_107/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_108/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_108/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_50/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_50/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_109/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_109/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_51/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_51/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_110/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_110/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_61/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_61/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_62/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_62/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_106/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_106/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_49/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_49/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_107/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_107/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_108/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_108/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_50/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_50/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_109/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_109/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_51/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_51/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_110/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_110/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_61/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_61/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_62/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_62/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
 serving_default_conv3d_106_inputPlaceholder*3
_output_shapes!
:?????????222*
dtype0*(
shape:?????????222
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_conv3d_106_inputconv3d_106/kernelconv3d_106/bias&batch_normalization_49/moving_variancebatch_normalization_49/gamma"batch_normalization_49/moving_meanbatch_normalization_49/betaconv3d_107/kernelconv3d_107/biasconv3d_108/kernelconv3d_108/bias&batch_normalization_50/moving_variancebatch_normalization_50/gamma"batch_normalization_50/moving_meanbatch_normalization_50/betaconv3d_109/kernelconv3d_109/bias&batch_normalization_51/moving_variancebatch_normalization_51/gamma"batch_normalization_51/moving_meanbatch_normalization_51/betaconv3d_110/kernelconv3d_110/biasdense_61/kerneldense_61/biasdense_62/kerneldense_62/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_49278
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv3d_106/kernel/Read/ReadVariableOp#conv3d_106/bias/Read/ReadVariableOp0batch_normalization_49/gamma/Read/ReadVariableOp/batch_normalization_49/beta/Read/ReadVariableOp6batch_normalization_49/moving_mean/Read/ReadVariableOp:batch_normalization_49/moving_variance/Read/ReadVariableOp%conv3d_107/kernel/Read/ReadVariableOp#conv3d_107/bias/Read/ReadVariableOp%conv3d_108/kernel/Read/ReadVariableOp#conv3d_108/bias/Read/ReadVariableOp0batch_normalization_50/gamma/Read/ReadVariableOp/batch_normalization_50/beta/Read/ReadVariableOp6batch_normalization_50/moving_mean/Read/ReadVariableOp:batch_normalization_50/moving_variance/Read/ReadVariableOp%conv3d_109/kernel/Read/ReadVariableOp#conv3d_109/bias/Read/ReadVariableOp0batch_normalization_51/gamma/Read/ReadVariableOp/batch_normalization_51/beta/Read/ReadVariableOp6batch_normalization_51/moving_mean/Read/ReadVariableOp:batch_normalization_51/moving_variance/Read/ReadVariableOp%conv3d_110/kernel/Read/ReadVariableOp#conv3d_110/bias/Read/ReadVariableOp#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOp#dense_62/kernel/Read/ReadVariableOp!dense_62/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv3d_106/kernel/m/Read/ReadVariableOp*Adam/conv3d_106/bias/m/Read/ReadVariableOp7Adam/batch_normalization_49/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_49/beta/m/Read/ReadVariableOp,Adam/conv3d_107/kernel/m/Read/ReadVariableOp*Adam/conv3d_107/bias/m/Read/ReadVariableOp,Adam/conv3d_108/kernel/m/Read/ReadVariableOp*Adam/conv3d_108/bias/m/Read/ReadVariableOp7Adam/batch_normalization_50/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_50/beta/m/Read/ReadVariableOp,Adam/conv3d_109/kernel/m/Read/ReadVariableOp*Adam/conv3d_109/bias/m/Read/ReadVariableOp7Adam/batch_normalization_51/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_51/beta/m/Read/ReadVariableOp,Adam/conv3d_110/kernel/m/Read/ReadVariableOp*Adam/conv3d_110/bias/m/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp*Adam/dense_62/kernel/m/Read/ReadVariableOp(Adam/dense_62/bias/m/Read/ReadVariableOp,Adam/conv3d_106/kernel/v/Read/ReadVariableOp*Adam/conv3d_106/bias/v/Read/ReadVariableOp7Adam/batch_normalization_49/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_49/beta/v/Read/ReadVariableOp,Adam/conv3d_107/kernel/v/Read/ReadVariableOp*Adam/conv3d_107/bias/v/Read/ReadVariableOp,Adam/conv3d_108/kernel/v/Read/ReadVariableOp*Adam/conv3d_108/bias/v/Read/ReadVariableOp7Adam/batch_normalization_50/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_50/beta/v/Read/ReadVariableOp,Adam/conv3d_109/kernel/v/Read/ReadVariableOp*Adam/conv3d_109/bias/v/Read/ReadVariableOp7Adam/batch_normalization_51/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_51/beta/v/Read/ReadVariableOp,Adam/conv3d_110/kernel/v/Read/ReadVariableOp*Adam/conv3d_110/bias/v/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOp*Adam/dense_62/kernel/v/Read/ReadVariableOp(Adam/dense_62/bias/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
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
__inference__traced_save_50627
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_106/kernelconv3d_106/biasbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_varianceconv3d_107/kernelconv3d_107/biasconv3d_108/kernelconv3d_108/biasbatch_normalization_50/gammabatch_normalization_50/beta"batch_normalization_50/moving_mean&batch_normalization_50/moving_varianceconv3d_109/kernelconv3d_109/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv3d_110/kernelconv3d_110/biasdense_61/kerneldense_61/biasdense_62/kerneldense_62/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv3d_106/kernel/mAdam/conv3d_106/bias/m#Adam/batch_normalization_49/gamma/m"Adam/batch_normalization_49/beta/mAdam/conv3d_107/kernel/mAdam/conv3d_107/bias/mAdam/conv3d_108/kernel/mAdam/conv3d_108/bias/m#Adam/batch_normalization_50/gamma/m"Adam/batch_normalization_50/beta/mAdam/conv3d_109/kernel/mAdam/conv3d_109/bias/m#Adam/batch_normalization_51/gamma/m"Adam/batch_normalization_51/beta/mAdam/conv3d_110/kernel/mAdam/conv3d_110/bias/mAdam/dense_61/kernel/mAdam/dense_61/bias/mAdam/dense_62/kernel/mAdam/dense_62/bias/mAdam/conv3d_106/kernel/vAdam/conv3d_106/bias/v#Adam/batch_normalization_49/gamma/v"Adam/batch_normalization_49/beta/vAdam/conv3d_107/kernel/vAdam/conv3d_107/bias/vAdam/conv3d_108/kernel/vAdam/conv3d_108/bias/v#Adam/batch_normalization_50/gamma/v"Adam/batch_normalization_50/beta/vAdam/conv3d_109/kernel/vAdam/conv3d_109/bias/v#Adam/batch_normalization_51/gamma/v"Adam/batch_normalization_51/beta/vAdam/conv3d_110/kernel/vAdam/conv3d_110/bias/vAdam/dense_61/kernel/vAdam/dense_61/bias/vAdam/dense_62/kernel/vAdam/dense_62/bias/v*W
TinP
N2L*
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
!__inference__traced_restore_50862??
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50141

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_22_layer_call_and_return_conditional_losses_48415

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_50_layer_call_fn_50035

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_479092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_61_layer_call_and_return_conditional_losses_48428

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
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
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_51_layer_call_fn_50281

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_486152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?K
?
H__inference_sequential_22_layer_call_and_return_conditional_losses_48452

inputs.
conv3d_106_48228:
conv3d_106_48230:*
batch_normalization_49_48253:*
batch_normalization_49_48255:*
batch_normalization_49_48257:*
batch_normalization_49_48259:.
conv3d_107_48274:
conv3d_107_48276:.
conv3d_108_48298:
conv3d_108_48300:*
batch_normalization_50_48323:*
batch_normalization_50_48325:*
batch_normalization_50_48327:*
batch_normalization_50_48329:.
conv3d_109_48351:
conv3d_109_48353:*
batch_normalization_51_48376:*
batch_normalization_51_48378:*
batch_normalization_51_48380:*
batch_normalization_51_48382:.
conv3d_110_48404:
conv3d_110_48406:"
dense_61_48429:
??@
dense_61_48431:@ 
dense_62_48446:@
dense_62_48448:
identity??.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_50/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?"conv3d_106/StatefulPartitionedCall?"conv3d_107/StatefulPartitionedCall?"conv3d_108/StatefulPartitionedCall?"conv3d_109/StatefulPartitionedCall?"conv3d_110/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall?
"conv3d_106/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_106_48228conv3d_106_48230*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_106_layer_call_and_return_conditional_losses_482272$
"conv3d_106/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall+conv3d_106/StatefulPartitionedCall:output:0batch_normalization_49_48253batch_normalization_49_48255batch_normalization_49_48257batch_normalization_49_48259*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_4825220
.batch_normalization_49/StatefulPartitionedCall?
"conv3d_107/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0conv3d_107_48274conv3d_107_48276*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_107_layer_call_and_return_conditional_losses_482732$
"conv3d_107/StatefulPartitionedCall?
dropout_42/PartitionedCallPartitionedCall+conv3d_107/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_482842
dropout_42/PartitionedCall?
"conv3d_108/StatefulPartitionedCallStatefulPartitionedCall#dropout_42/PartitionedCall:output:0conv3d_108_48298conv3d_108_48300*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_108_layer_call_and_return_conditional_losses_482972$
"conv3d_108/StatefulPartitionedCall?
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall+conv3d_108/StatefulPartitionedCall:output:0batch_normalization_50_48323batch_normalization_50_48325batch_normalization_50_48327batch_normalization_50_48329*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_4832220
.batch_normalization_50/StatefulPartitionedCall?
dropout_43/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_43_layer_call_and_return_conditional_losses_483372
dropout_43/PartitionedCall?
"conv3d_109/StatefulPartitionedCallStatefulPartitionedCall#dropout_43/PartitionedCall:output:0conv3d_109_48351conv3d_109_48353*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_109_layer_call_and_return_conditional_losses_483502$
"conv3d_109/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall+conv3d_109/StatefulPartitionedCall:output:0batch_normalization_51_48376batch_normalization_51_48378batch_normalization_51_48380batch_normalization_51_48382*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_4837520
.batch_normalization_51/StatefulPartitionedCall?
dropout_44/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_483902
dropout_44/PartitionedCall?
"conv3d_110/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0conv3d_110_48404conv3d_110_48406*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_110_layer_call_and_return_conditional_losses_484032$
"conv3d_110/StatefulPartitionedCall?
flatten_22/PartitionedCallPartitionedCall+conv3d_110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_484152
flatten_22/PartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_61_48429dense_61_48431*
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
C__inference_dense_61_layer_call_and_return_conditional_losses_484282"
 dense_61/StatefulPartitionedCall?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_48446dense_62_48448*
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
C__inference_dense_62_layer_call_and_return_conditional_losses_484452"
 dense_62/StatefulPartitionedCall?
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall#^conv3d_106/StatefulPartitionedCall#^conv3d_107/StatefulPartitionedCall#^conv3d_108/StatefulPartitionedCall#^conv3d_109/StatefulPartitionedCall#^conv3d_110/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2H
"conv3d_106/StatefulPartitionedCall"conv3d_106/StatefulPartitionedCall2H
"conv3d_107/StatefulPartitionedCall"conv3d_107/StatefulPartitionedCall2H
"conv3d_108/StatefulPartitionedCall"conv3d_108/StatefulPartitionedCall2H
"conv3d_109/StatefulPartitionedCall"conv3d_109/StatefulPartitionedCall2H
"conv3d_110/StatefulPartitionedCall"conv3d_110/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49795

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????2222
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
c
E__inference_dropout_44_layer_call_and_return_conditional_losses_48390

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:?????????2222

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:?????????2222

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49761

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
c
*__inference_dropout_42_layer_call_fn_49894

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_487492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22222
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_49_layer_call_fn_49808

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_477472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_44_layer_call_and_return_conditional_losses_50298

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_47969

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50229

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????2222
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
c
E__inference_dropout_42_layer_call_and_return_conditional_losses_49872

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:?????????2222

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:?????????2222

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_48131

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_51_layer_call_fn_50255

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_481312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_42_layer_call_and_return_conditional_losses_49884

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
(__inference_dense_61_layer_call_fn_50359

inputs
unknown:
??@
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
C__inference_dense_61_layer_call_and_return_conditional_losses_484282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_50_layer_call_fn_50061

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_483222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_51_layer_call_fn_50242

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_480712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv3d_109_layer_call_and_return_conditional_losses_48350

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

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
?
E__inference_conv3d_110_layer_call_and_return_conditional_losses_48403

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

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
?
?
6__inference_batch_normalization_51_layer_call_fn_50268

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_483752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
*__inference_conv3d_106_layer_call_fn_49687

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_106_layer_call_and_return_conditional_losses_482272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????222: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
*__inference_conv3d_107_layer_call_fn_49867

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_107_layer_call_and_return_conditional_losses_482732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

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
?
?
E__inference_conv3d_107_layer_call_and_return_conditional_losses_49858

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

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
*__inference_conv3d_108_layer_call_fn_49914

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_108_layer_call_and_return_conditional_losses_482972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

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
?
c
E__inference_dropout_43_layer_call_and_return_conditional_losses_48337

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:?????????2222

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:?????????2222

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
c
E__inference_dropout_44_layer_call_and_return_conditional_losses_50286

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:?????????2222

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:?????????2222

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
??
?
H__inference_sequential_22_layer_call_and_return_conditional_losses_49384

inputsG
)conv3d_106_conv3d_readvariableop_resource:8
*conv3d_106_biasadd_readvariableop_resource:F
8batch_normalization_49_batchnorm_readvariableop_resource:J
<batch_normalization_49_batchnorm_mul_readvariableop_resource:H
:batch_normalization_49_batchnorm_readvariableop_1_resource:H
:batch_normalization_49_batchnorm_readvariableop_2_resource:G
)conv3d_107_conv3d_readvariableop_resource:8
*conv3d_107_biasadd_readvariableop_resource:G
)conv3d_108_conv3d_readvariableop_resource:8
*conv3d_108_biasadd_readvariableop_resource:F
8batch_normalization_50_batchnorm_readvariableop_resource:J
<batch_normalization_50_batchnorm_mul_readvariableop_resource:H
:batch_normalization_50_batchnorm_readvariableop_1_resource:H
:batch_normalization_50_batchnorm_readvariableop_2_resource:G
)conv3d_109_conv3d_readvariableop_resource:8
*conv3d_109_biasadd_readvariableop_resource:F
8batch_normalization_51_batchnorm_readvariableop_resource:J
<batch_normalization_51_batchnorm_mul_readvariableop_resource:H
:batch_normalization_51_batchnorm_readvariableop_1_resource:H
:batch_normalization_51_batchnorm_readvariableop_2_resource:G
)conv3d_110_conv3d_readvariableop_resource:8
*conv3d_110_biasadd_readvariableop_resource:;
'dense_61_matmul_readvariableop_resource:
??@6
(dense_61_biasadd_readvariableop_resource:@9
'dense_62_matmul_readvariableop_resource:@6
(dense_62_biasadd_readvariableop_resource:
identity??/batch_normalization_49/batchnorm/ReadVariableOp?1batch_normalization_49/batchnorm/ReadVariableOp_1?1batch_normalization_49/batchnorm/ReadVariableOp_2?3batch_normalization_49/batchnorm/mul/ReadVariableOp?/batch_normalization_50/batchnorm/ReadVariableOp?1batch_normalization_50/batchnorm/ReadVariableOp_1?1batch_normalization_50/batchnorm/ReadVariableOp_2?3batch_normalization_50/batchnorm/mul/ReadVariableOp?/batch_normalization_51/batchnorm/ReadVariableOp?1batch_normalization_51/batchnorm/ReadVariableOp_1?1batch_normalization_51/batchnorm/ReadVariableOp_2?3batch_normalization_51/batchnorm/mul/ReadVariableOp?!conv3d_106/BiasAdd/ReadVariableOp? conv3d_106/Conv3D/ReadVariableOp?!conv3d_107/BiasAdd/ReadVariableOp? conv3d_107/Conv3D/ReadVariableOp?!conv3d_108/BiasAdd/ReadVariableOp? conv3d_108/Conv3D/ReadVariableOp?!conv3d_109/BiasAdd/ReadVariableOp? conv3d_109/Conv3D/ReadVariableOp?!conv3d_110/BiasAdd/ReadVariableOp? conv3d_110/Conv3D/ReadVariableOp?dense_61/BiasAdd/ReadVariableOp?dense_61/MatMul/ReadVariableOp?dense_62/BiasAdd/ReadVariableOp?dense_62/MatMul/ReadVariableOp?
 conv3d_106/Conv3D/ReadVariableOpReadVariableOp)conv3d_106_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_106/Conv3D/ReadVariableOp?
conv3d_106/Conv3DConv3Dinputs(conv3d_106/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_106/Conv3D?
!conv3d_106/BiasAdd/ReadVariableOpReadVariableOp*conv3d_106_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_106/BiasAdd/ReadVariableOp?
conv3d_106/BiasAddBiasAddconv3d_106/Conv3D:output:0)conv3d_106/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_106/BiasAdd?
conv3d_106/ReluReluconv3d_106/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_106/Relu?
/batch_normalization_49/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_49_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_49/batchnorm/ReadVariableOp?
&batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_49/batchnorm/add/y?
$batch_normalization_49/batchnorm/addAddV27batch_normalization_49/batchnorm/ReadVariableOp:value:0/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/add?
&batch_normalization_49/batchnorm/RsqrtRsqrt(batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/Rsqrt?
3batch_normalization_49/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_49_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_49/batchnorm/mul/ReadVariableOp?
$batch_normalization_49/batchnorm/mulMul*batch_normalization_49/batchnorm/Rsqrt:y:0;batch_normalization_49/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/mul?
&batch_normalization_49/batchnorm/mul_1Mulconv3d_106/Relu:activations:0(batch_normalization_49/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_49/batchnorm/mul_1?
1batch_normalization_49/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_49_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_49/batchnorm/ReadVariableOp_1?
&batch_normalization_49/batchnorm/mul_2Mul9batch_normalization_49/batchnorm/ReadVariableOp_1:value:0(batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/mul_2?
1batch_normalization_49/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_49_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_49/batchnorm/ReadVariableOp_2?
$batch_normalization_49/batchnorm/subSub9batch_normalization_49/batchnorm/ReadVariableOp_2:value:0*batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/sub?
&batch_normalization_49/batchnorm/add_1AddV2*batch_normalization_49/batchnorm/mul_1:z:0(batch_normalization_49/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_49/batchnorm/add_1?
 conv3d_107/Conv3D/ReadVariableOpReadVariableOp)conv3d_107_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_107/Conv3D/ReadVariableOp?
conv3d_107/Conv3DConv3D*batch_normalization_49/batchnorm/add_1:z:0(conv3d_107/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_107/Conv3D?
!conv3d_107/BiasAdd/ReadVariableOpReadVariableOp*conv3d_107_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_107/BiasAdd/ReadVariableOp?
conv3d_107/BiasAddBiasAddconv3d_107/Conv3D:output:0)conv3d_107/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_107/BiasAdd?
conv3d_107/SigmoidSigmoidconv3d_107/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_107/Sigmoid?
dropout_42/IdentityIdentityconv3d_107/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????2222
dropout_42/Identity?
 conv3d_108/Conv3D/ReadVariableOpReadVariableOp)conv3d_108_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_108/Conv3D/ReadVariableOp?
conv3d_108/Conv3DConv3Ddropout_42/Identity:output:0(conv3d_108/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_108/Conv3D?
!conv3d_108/BiasAdd/ReadVariableOpReadVariableOp*conv3d_108_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_108/BiasAdd/ReadVariableOp?
conv3d_108/BiasAddBiasAddconv3d_108/Conv3D:output:0)conv3d_108/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_108/BiasAdd?
conv3d_108/SigmoidSigmoidconv3d_108/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_108/Sigmoid?
/batch_normalization_50/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_50_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_50/batchnorm/ReadVariableOp?
&batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_50/batchnorm/add/y?
$batch_normalization_50/batchnorm/addAddV27batch_normalization_50/batchnorm/ReadVariableOp:value:0/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/add?
&batch_normalization_50/batchnorm/RsqrtRsqrt(batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/Rsqrt?
3batch_normalization_50/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_50_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_50/batchnorm/mul/ReadVariableOp?
$batch_normalization_50/batchnorm/mulMul*batch_normalization_50/batchnorm/Rsqrt:y:0;batch_normalization_50/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/mul?
&batch_normalization_50/batchnorm/mul_1Mulconv3d_108/Sigmoid:y:0(batch_normalization_50/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_50/batchnorm/mul_1?
1batch_normalization_50/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_50_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_50/batchnorm/ReadVariableOp_1?
&batch_normalization_50/batchnorm/mul_2Mul9batch_normalization_50/batchnorm/ReadVariableOp_1:value:0(batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/mul_2?
1batch_normalization_50/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_50_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_50/batchnorm/ReadVariableOp_2?
$batch_normalization_50/batchnorm/subSub9batch_normalization_50/batchnorm/ReadVariableOp_2:value:0*batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/sub?
&batch_normalization_50/batchnorm/add_1AddV2*batch_normalization_50/batchnorm/mul_1:z:0(batch_normalization_50/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_50/batchnorm/add_1?
dropout_43/IdentityIdentity*batch_normalization_50/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????2222
dropout_43/Identity?
 conv3d_109/Conv3D/ReadVariableOpReadVariableOp)conv3d_109_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_109/Conv3D/ReadVariableOp?
conv3d_109/Conv3DConv3Ddropout_43/Identity:output:0(conv3d_109/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_109/Conv3D?
!conv3d_109/BiasAdd/ReadVariableOpReadVariableOp*conv3d_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_109/BiasAdd/ReadVariableOp?
conv3d_109/BiasAddBiasAddconv3d_109/Conv3D:output:0)conv3d_109/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_109/BiasAdd?
conv3d_109/ReluReluconv3d_109/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_109/Relu?
/batch_normalization_51/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_51_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_51/batchnorm/ReadVariableOp?
&batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_51/batchnorm/add/y?
$batch_normalization_51/batchnorm/addAddV27batch_normalization_51/batchnorm/ReadVariableOp:value:0/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/add?
&batch_normalization_51/batchnorm/RsqrtRsqrt(batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/Rsqrt?
3batch_normalization_51/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_51_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_51/batchnorm/mul/ReadVariableOp?
$batch_normalization_51/batchnorm/mulMul*batch_normalization_51/batchnorm/Rsqrt:y:0;batch_normalization_51/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/mul?
&batch_normalization_51/batchnorm/mul_1Mulconv3d_109/Relu:activations:0(batch_normalization_51/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_51/batchnorm/mul_1?
1batch_normalization_51/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_51_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_51/batchnorm/ReadVariableOp_1?
&batch_normalization_51/batchnorm/mul_2Mul9batch_normalization_51/batchnorm/ReadVariableOp_1:value:0(batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/mul_2?
1batch_normalization_51/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_51_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_51/batchnorm/ReadVariableOp_2?
$batch_normalization_51/batchnorm/subSub9batch_normalization_51/batchnorm/ReadVariableOp_2:value:0*batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/sub?
&batch_normalization_51/batchnorm/add_1AddV2*batch_normalization_51/batchnorm/mul_1:z:0(batch_normalization_51/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_51/batchnorm/add_1?
dropout_44/IdentityIdentity*batch_normalization_51/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????2222
dropout_44/Identity?
 conv3d_110/Conv3D/ReadVariableOpReadVariableOp)conv3d_110_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_110/Conv3D/ReadVariableOp?
conv3d_110/Conv3DConv3Ddropout_44/Identity:output:0(conv3d_110/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_110/Conv3D?
!conv3d_110/BiasAdd/ReadVariableOpReadVariableOp*conv3d_110_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_110/BiasAdd/ReadVariableOp?
conv3d_110/BiasAddBiasAddconv3d_110/Conv3D:output:0)conv3d_110/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_110/BiasAdd?
conv3d_110/SigmoidSigmoidconv3d_110/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_110/Sigmoidu
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
flatten_22/Const?
flatten_22/ReshapeReshapeconv3d_110/Sigmoid:y:0flatten_22/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_22/Reshape?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMulflatten_22/Reshape:output:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_61/MatMul?
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_61/BiasAdd/ReadVariableOp?
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_61/BiasAdds
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_61/Relu?
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_62/MatMul/ReadVariableOp?
dense_62/MatMulMatMuldense_61/Relu:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_62/MatMul?
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_62/BiasAdd/ReadVariableOp?
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_62/BiasAdd|
dense_62/SigmoidSigmoiddense_62/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_62/Sigmoid?	
IdentityIdentitydense_62/Sigmoid:y:00^batch_normalization_49/batchnorm/ReadVariableOp2^batch_normalization_49/batchnorm/ReadVariableOp_12^batch_normalization_49/batchnorm/ReadVariableOp_24^batch_normalization_49/batchnorm/mul/ReadVariableOp0^batch_normalization_50/batchnorm/ReadVariableOp2^batch_normalization_50/batchnorm/ReadVariableOp_12^batch_normalization_50/batchnorm/ReadVariableOp_24^batch_normalization_50/batchnorm/mul/ReadVariableOp0^batch_normalization_51/batchnorm/ReadVariableOp2^batch_normalization_51/batchnorm/ReadVariableOp_12^batch_normalization_51/batchnorm/ReadVariableOp_24^batch_normalization_51/batchnorm/mul/ReadVariableOp"^conv3d_106/BiasAdd/ReadVariableOp!^conv3d_106/Conv3D/ReadVariableOp"^conv3d_107/BiasAdd/ReadVariableOp!^conv3d_107/Conv3D/ReadVariableOp"^conv3d_108/BiasAdd/ReadVariableOp!^conv3d_108/Conv3D/ReadVariableOp"^conv3d_109/BiasAdd/ReadVariableOp!^conv3d_109/Conv3D/ReadVariableOp"^conv3d_110/BiasAdd/ReadVariableOp!^conv3d_110/Conv3D/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_49/batchnorm/ReadVariableOp/batch_normalization_49/batchnorm/ReadVariableOp2f
1batch_normalization_49/batchnorm/ReadVariableOp_11batch_normalization_49/batchnorm/ReadVariableOp_12f
1batch_normalization_49/batchnorm/ReadVariableOp_21batch_normalization_49/batchnorm/ReadVariableOp_22j
3batch_normalization_49/batchnorm/mul/ReadVariableOp3batch_normalization_49/batchnorm/mul/ReadVariableOp2b
/batch_normalization_50/batchnorm/ReadVariableOp/batch_normalization_50/batchnorm/ReadVariableOp2f
1batch_normalization_50/batchnorm/ReadVariableOp_11batch_normalization_50/batchnorm/ReadVariableOp_12f
1batch_normalization_50/batchnorm/ReadVariableOp_21batch_normalization_50/batchnorm/ReadVariableOp_22j
3batch_normalization_50/batchnorm/mul/ReadVariableOp3batch_normalization_50/batchnorm/mul/ReadVariableOp2b
/batch_normalization_51/batchnorm/ReadVariableOp/batch_normalization_51/batchnorm/ReadVariableOp2f
1batch_normalization_51/batchnorm/ReadVariableOp_11batch_normalization_51/batchnorm/ReadVariableOp_12f
1batch_normalization_51/batchnorm/ReadVariableOp_21batch_normalization_51/batchnorm/ReadVariableOp_22j
3batch_normalization_51/batchnorm/mul/ReadVariableOp3batch_normalization_51/batchnorm/mul/ReadVariableOp2F
!conv3d_106/BiasAdd/ReadVariableOp!conv3d_106/BiasAdd/ReadVariableOp2D
 conv3d_106/Conv3D/ReadVariableOp conv3d_106/Conv3D/ReadVariableOp2F
!conv3d_107/BiasAdd/ReadVariableOp!conv3d_107/BiasAdd/ReadVariableOp2D
 conv3d_107/Conv3D/ReadVariableOp conv3d_107/Conv3D/ReadVariableOp2F
!conv3d_108/BiasAdd/ReadVariableOp!conv3d_108/BiasAdd/ReadVariableOp2D
 conv3d_108/Conv3D/ReadVariableOp conv3d_108/Conv3D/ReadVariableOp2F
!conv3d_109/BiasAdd/ReadVariableOp!conv3d_109/BiasAdd/ReadVariableOp2D
 conv3d_109/Conv3D/ReadVariableOp conv3d_109/Conv3D/ReadVariableOp2F
!conv3d_110/BiasAdd/ReadVariableOp!conv3d_110/BiasAdd/ReadVariableOp2D
 conv3d_110/Conv3D/ReadVariableOp conv3d_110/Conv3D/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?K
?
H__inference_sequential_22_layer_call_and_return_conditional_losses_49143
conv3d_106_input.
conv3d_106_49076:
conv3d_106_49078:*
batch_normalization_49_49081:*
batch_normalization_49_49083:*
batch_normalization_49_49085:*
batch_normalization_49_49087:.
conv3d_107_49090:
conv3d_107_49092:.
conv3d_108_49096:
conv3d_108_49098:*
batch_normalization_50_49101:*
batch_normalization_50_49103:*
batch_normalization_50_49105:*
batch_normalization_50_49107:.
conv3d_109_49111:
conv3d_109_49113:*
batch_normalization_51_49116:*
batch_normalization_51_49118:*
batch_normalization_51_49120:*
batch_normalization_51_49122:.
conv3d_110_49126:
conv3d_110_49128:"
dense_61_49132:
??@
dense_61_49134:@ 
dense_62_49137:@
dense_62_49139:
identity??.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_50/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?"conv3d_106/StatefulPartitionedCall?"conv3d_107/StatefulPartitionedCall?"conv3d_108/StatefulPartitionedCall?"conv3d_109/StatefulPartitionedCall?"conv3d_110/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall?
"conv3d_106/StatefulPartitionedCallStatefulPartitionedCallconv3d_106_inputconv3d_106_49076conv3d_106_49078*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_106_layer_call_and_return_conditional_losses_482272$
"conv3d_106/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall+conv3d_106/StatefulPartitionedCall:output:0batch_normalization_49_49081batch_normalization_49_49083batch_normalization_49_49085batch_normalization_49_49087*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_4825220
.batch_normalization_49/StatefulPartitionedCall?
"conv3d_107/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0conv3d_107_49090conv3d_107_49092*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_107_layer_call_and_return_conditional_losses_482732$
"conv3d_107/StatefulPartitionedCall?
dropout_42/PartitionedCallPartitionedCall+conv3d_107/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_482842
dropout_42/PartitionedCall?
"conv3d_108/StatefulPartitionedCallStatefulPartitionedCall#dropout_42/PartitionedCall:output:0conv3d_108_49096conv3d_108_49098*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_108_layer_call_and_return_conditional_losses_482972$
"conv3d_108/StatefulPartitionedCall?
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall+conv3d_108/StatefulPartitionedCall:output:0batch_normalization_50_49101batch_normalization_50_49103batch_normalization_50_49105batch_normalization_50_49107*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_4832220
.batch_normalization_50/StatefulPartitionedCall?
dropout_43/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_43_layer_call_and_return_conditional_losses_483372
dropout_43/PartitionedCall?
"conv3d_109/StatefulPartitionedCallStatefulPartitionedCall#dropout_43/PartitionedCall:output:0conv3d_109_49111conv3d_109_49113*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_109_layer_call_and_return_conditional_losses_483502$
"conv3d_109/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall+conv3d_109/StatefulPartitionedCall:output:0batch_normalization_51_49116batch_normalization_51_49118batch_normalization_51_49120batch_normalization_51_49122*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_4837520
.batch_normalization_51/StatefulPartitionedCall?
dropout_44/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_483902
dropout_44/PartitionedCall?
"conv3d_110/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0conv3d_110_49126conv3d_110_49128*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_110_layer_call_and_return_conditional_losses_484032$
"conv3d_110/StatefulPartitionedCall?
flatten_22/PartitionedCallPartitionedCall+conv3d_110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_484152
flatten_22/PartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_61_49132dense_61_49134*
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
C__inference_dense_61_layer_call_and_return_conditional_losses_484282"
 dense_61/StatefulPartitionedCall?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_49137dense_62_49139*
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
C__inference_dense_62_layer_call_and_return_conditional_losses_484452"
 dense_62/StatefulPartitionedCall?
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall#^conv3d_106/StatefulPartitionedCall#^conv3d_107/StatefulPartitionedCall#^conv3d_108/StatefulPartitionedCall#^conv3d_109/StatefulPartitionedCall#^conv3d_110/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2H
"conv3d_106/StatefulPartitionedCall"conv3d_106/StatefulPartitionedCall2H
"conv3d_107/StatefulPartitionedCall"conv3d_107/StatefulPartitionedCall2H
"conv3d_108/StatefulPartitionedCall"conv3d_108/StatefulPartitionedCall2H
"conv3d_109/StatefulPartitionedCall"conv3d_109/StatefulPartitionedCall2H
"conv3d_110/StatefulPartitionedCall"conv3d_110/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:e a
3
_output_shapes!
:?????????222
*
_user_specified_nameconv3d_106_input
?+
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_48811

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????2222
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?P
?
H__inference_sequential_22_layer_call_and_return_conditional_losses_49213
conv3d_106_input.
conv3d_106_49146:
conv3d_106_49148:*
batch_normalization_49_49151:*
batch_normalization_49_49153:*
batch_normalization_49_49155:*
batch_normalization_49_49157:.
conv3d_107_49160:
conv3d_107_49162:.
conv3d_108_49166:
conv3d_108_49168:*
batch_normalization_50_49171:*
batch_normalization_50_49173:*
batch_normalization_50_49175:*
batch_normalization_50_49177:.
conv3d_109_49181:
conv3d_109_49183:*
batch_normalization_51_49186:*
batch_normalization_51_49188:*
batch_normalization_51_49190:*
batch_normalization_51_49192:.
conv3d_110_49196:
conv3d_110_49198:"
dense_61_49202:
??@
dense_61_49204:@ 
dense_62_49207:@
dense_62_49209:
identity??.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_50/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?"conv3d_106/StatefulPartitionedCall?"conv3d_107/StatefulPartitionedCall?"conv3d_108/StatefulPartitionedCall?"conv3d_109/StatefulPartitionedCall?"conv3d_110/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall?"dropout_42/StatefulPartitionedCall?"dropout_43/StatefulPartitionedCall?"dropout_44/StatefulPartitionedCall?
"conv3d_106/StatefulPartitionedCallStatefulPartitionedCallconv3d_106_inputconv3d_106_49146conv3d_106_49148*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_106_layer_call_and_return_conditional_losses_482272$
"conv3d_106/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall+conv3d_106/StatefulPartitionedCall:output:0batch_normalization_49_49151batch_normalization_49_49153batch_normalization_49_49155batch_normalization_49_49157*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_4881120
.batch_normalization_49/StatefulPartitionedCall?
"conv3d_107/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0conv3d_107_49160conv3d_107_49162*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_107_layer_call_and_return_conditional_losses_482732$
"conv3d_107/StatefulPartitionedCall?
"dropout_42/StatefulPartitionedCallStatefulPartitionedCall+conv3d_107/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_487492$
"dropout_42/StatefulPartitionedCall?
"conv3d_108/StatefulPartitionedCallStatefulPartitionedCall+dropout_42/StatefulPartitionedCall:output:0conv3d_108_49166conv3d_108_49168*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_108_layer_call_and_return_conditional_losses_482972$
"conv3d_108/StatefulPartitionedCall?
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall+conv3d_108/StatefulPartitionedCall:output:0batch_normalization_50_49171batch_normalization_50_49173batch_normalization_50_49175batch_normalization_50_49177*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_4870820
.batch_normalization_50/StatefulPartitionedCall?
"dropout_43/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0#^dropout_42/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_43_layer_call_and_return_conditional_losses_486562$
"dropout_43/StatefulPartitionedCall?
"conv3d_109/StatefulPartitionedCallStatefulPartitionedCall+dropout_43/StatefulPartitionedCall:output:0conv3d_109_49181conv3d_109_49183*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_109_layer_call_and_return_conditional_losses_483502$
"conv3d_109/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall+conv3d_109/StatefulPartitionedCall:output:0batch_normalization_51_49186batch_normalization_51_49188batch_normalization_51_49190batch_normalization_51_49192*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_4861520
.batch_normalization_51/StatefulPartitionedCall?
"dropout_44/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0#^dropout_43/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_485632$
"dropout_44/StatefulPartitionedCall?
"conv3d_110/StatefulPartitionedCallStatefulPartitionedCall+dropout_44/StatefulPartitionedCall:output:0conv3d_110_49196conv3d_110_49198*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_110_layer_call_and_return_conditional_losses_484032$
"conv3d_110/StatefulPartitionedCall?
flatten_22/PartitionedCallPartitionedCall+conv3d_110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_484152
flatten_22/PartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_61_49202dense_61_49204*
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
C__inference_dense_61_layer_call_and_return_conditional_losses_484282"
 dense_61/StatefulPartitionedCall?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_49207dense_62_49209*
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
C__inference_dense_62_layer_call_and_return_conditional_losses_484452"
 dense_62/StatefulPartitionedCall?
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall#^conv3d_106/StatefulPartitionedCall#^conv3d_107/StatefulPartitionedCall#^conv3d_108/StatefulPartitionedCall#^conv3d_109/StatefulPartitionedCall#^conv3d_110/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall#^dropout_42/StatefulPartitionedCall#^dropout_43/StatefulPartitionedCall#^dropout_44/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2H
"conv3d_106/StatefulPartitionedCall"conv3d_106/StatefulPartitionedCall2H
"conv3d_107/StatefulPartitionedCall"conv3d_107/StatefulPartitionedCall2H
"conv3d_108/StatefulPartitionedCall"conv3d_108/StatefulPartitionedCall2H
"conv3d_109/StatefulPartitionedCall"conv3d_109/StatefulPartitionedCall2H
"conv3d_110/StatefulPartitionedCall"conv3d_110/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2H
"dropout_42/StatefulPartitionedCall"dropout_42/StatefulPartitionedCall2H
"dropout_43/StatefulPartitionedCall"dropout_43/StatefulPartitionedCall2H
"dropout_44/StatefulPartitionedCall"dropout_44/StatefulPartitionedCall:e a
3
_output_shapes!
:?????????222
*
_user_specified_nameconv3d_106_input
?
d
E__inference_dropout_43_layer_call_and_return_conditional_losses_48656

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?

?
C__inference_dense_61_layer_call_and_return_conditional_losses_50350

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
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
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_48071

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_49_layer_call_fn_49834

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_482522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_48322

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
c
*__inference_dropout_43_layer_call_fn_50101

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_43_layer_call_and_return_conditional_losses_486562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22222
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50175

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv3d_110_layer_call_and_return_conditional_losses_50319

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

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
?
c
E__inference_dropout_43_layer_call_and_return_conditional_losses_50079

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:?????????2222

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:?????????2222

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
E__inference_conv3d_107_layer_call_and_return_conditional_losses_48273

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

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
?
?
-__inference_sequential_22_layer_call_fn_48507
conv3d_106_input%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:(

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
??@

unknown_22:@

unknown_23:@

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_106_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_22_layer_call_and_return_conditional_losses_484522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
3
_output_shapes!
:?????????222
*
_user_specified_nameconv3d_106_input
?
F
*__inference_dropout_44_layer_call_fn_50303

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_483902
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50195

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49988

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
d
E__inference_dropout_43_layer_call_and_return_conditional_losses_50091

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
F
*__inference_dropout_42_layer_call_fn_49889

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_482842
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
d
E__inference_dropout_42_layer_call_and_return_conditional_losses_48749

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_47747

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_47807

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_50_layer_call_fn_50074

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_487082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_48375

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
(__inference_dense_62_layer_call_fn_50379

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
C__inference_dense_62_layer_call_and_return_conditional_losses_484452
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
?
?
E__inference_conv3d_108_layer_call_and_return_conditional_losses_48297

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

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
ۗ
?!
__inference__traced_save_50627
file_prefix0
,savev2_conv3d_106_kernel_read_readvariableop.
*savev2_conv3d_106_bias_read_readvariableop;
7savev2_batch_normalization_49_gamma_read_readvariableop:
6savev2_batch_normalization_49_beta_read_readvariableopA
=savev2_batch_normalization_49_moving_mean_read_readvariableopE
Asavev2_batch_normalization_49_moving_variance_read_readvariableop0
,savev2_conv3d_107_kernel_read_readvariableop.
*savev2_conv3d_107_bias_read_readvariableop0
,savev2_conv3d_108_kernel_read_readvariableop.
*savev2_conv3d_108_bias_read_readvariableop;
7savev2_batch_normalization_50_gamma_read_readvariableop:
6savev2_batch_normalization_50_beta_read_readvariableopA
=savev2_batch_normalization_50_moving_mean_read_readvariableopE
Asavev2_batch_normalization_50_moving_variance_read_readvariableop0
,savev2_conv3d_109_kernel_read_readvariableop.
*savev2_conv3d_109_bias_read_readvariableop;
7savev2_batch_normalization_51_gamma_read_readvariableop:
6savev2_batch_normalization_51_beta_read_readvariableopA
=savev2_batch_normalization_51_moving_mean_read_readvariableopE
Asavev2_batch_normalization_51_moving_variance_read_readvariableop0
,savev2_conv3d_110_kernel_read_readvariableop.
*savev2_conv3d_110_bias_read_readvariableop.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop.
*savev2_dense_62_kernel_read_readvariableop,
(savev2_dense_62_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv3d_106_kernel_m_read_readvariableop5
1savev2_adam_conv3d_106_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_49_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_49_beta_m_read_readvariableop7
3savev2_adam_conv3d_107_kernel_m_read_readvariableop5
1savev2_adam_conv3d_107_bias_m_read_readvariableop7
3savev2_adam_conv3d_108_kernel_m_read_readvariableop5
1savev2_adam_conv3d_108_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_50_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_50_beta_m_read_readvariableop7
3savev2_adam_conv3d_109_kernel_m_read_readvariableop5
1savev2_adam_conv3d_109_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_51_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_51_beta_m_read_readvariableop7
3savev2_adam_conv3d_110_kernel_m_read_readvariableop5
1savev2_adam_conv3d_110_bias_m_read_readvariableop5
1savev2_adam_dense_61_kernel_m_read_readvariableop3
/savev2_adam_dense_61_bias_m_read_readvariableop5
1savev2_adam_dense_62_kernel_m_read_readvariableop3
/savev2_adam_dense_62_bias_m_read_readvariableop7
3savev2_adam_conv3d_106_kernel_v_read_readvariableop5
1savev2_adam_conv3d_106_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_49_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_49_beta_v_read_readvariableop7
3savev2_adam_conv3d_107_kernel_v_read_readvariableop5
1savev2_adam_conv3d_107_bias_v_read_readvariableop7
3savev2_adam_conv3d_108_kernel_v_read_readvariableop5
1savev2_adam_conv3d_108_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_50_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_50_beta_v_read_readvariableop7
3savev2_adam_conv3d_109_kernel_v_read_readvariableop5
1savev2_adam_conv3d_109_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_51_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_51_beta_v_read_readvariableop7
3savev2_adam_conv3d_110_kernel_v_read_readvariableop5
1savev2_adam_conv3d_110_bias_v_read_readvariableop5
1savev2_adam_dense_61_kernel_v_read_readvariableop3
/savev2_adam_dense_61_bias_v_read_readvariableop5
1savev2_adam_dense_62_kernel_v_read_readvariableop3
/savev2_adam_dense_62_bias_v_read_readvariableop
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
ShardedFilename?*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?)
value?)B?)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices? 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv3d_106_kernel_read_readvariableop*savev2_conv3d_106_bias_read_readvariableop7savev2_batch_normalization_49_gamma_read_readvariableop6savev2_batch_normalization_49_beta_read_readvariableop=savev2_batch_normalization_49_moving_mean_read_readvariableopAsavev2_batch_normalization_49_moving_variance_read_readvariableop,savev2_conv3d_107_kernel_read_readvariableop*savev2_conv3d_107_bias_read_readvariableop,savev2_conv3d_108_kernel_read_readvariableop*savev2_conv3d_108_bias_read_readvariableop7savev2_batch_normalization_50_gamma_read_readvariableop6savev2_batch_normalization_50_beta_read_readvariableop=savev2_batch_normalization_50_moving_mean_read_readvariableopAsavev2_batch_normalization_50_moving_variance_read_readvariableop,savev2_conv3d_109_kernel_read_readvariableop*savev2_conv3d_109_bias_read_readvariableop7savev2_batch_normalization_51_gamma_read_readvariableop6savev2_batch_normalization_51_beta_read_readvariableop=savev2_batch_normalization_51_moving_mean_read_readvariableopAsavev2_batch_normalization_51_moving_variance_read_readvariableop,savev2_conv3d_110_kernel_read_readvariableop*savev2_conv3d_110_bias_read_readvariableop*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop*savev2_dense_62_kernel_read_readvariableop(savev2_dense_62_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv3d_106_kernel_m_read_readvariableop1savev2_adam_conv3d_106_bias_m_read_readvariableop>savev2_adam_batch_normalization_49_gamma_m_read_readvariableop=savev2_adam_batch_normalization_49_beta_m_read_readvariableop3savev2_adam_conv3d_107_kernel_m_read_readvariableop1savev2_adam_conv3d_107_bias_m_read_readvariableop3savev2_adam_conv3d_108_kernel_m_read_readvariableop1savev2_adam_conv3d_108_bias_m_read_readvariableop>savev2_adam_batch_normalization_50_gamma_m_read_readvariableop=savev2_adam_batch_normalization_50_beta_m_read_readvariableop3savev2_adam_conv3d_109_kernel_m_read_readvariableop1savev2_adam_conv3d_109_bias_m_read_readvariableop>savev2_adam_batch_normalization_51_gamma_m_read_readvariableop=savev2_adam_batch_normalization_51_beta_m_read_readvariableop3savev2_adam_conv3d_110_kernel_m_read_readvariableop1savev2_adam_conv3d_110_bias_m_read_readvariableop1savev2_adam_dense_61_kernel_m_read_readvariableop/savev2_adam_dense_61_bias_m_read_readvariableop1savev2_adam_dense_62_kernel_m_read_readvariableop/savev2_adam_dense_62_bias_m_read_readvariableop3savev2_adam_conv3d_106_kernel_v_read_readvariableop1savev2_adam_conv3d_106_bias_v_read_readvariableop>savev2_adam_batch_normalization_49_gamma_v_read_readvariableop=savev2_adam_batch_normalization_49_beta_v_read_readvariableop3savev2_adam_conv3d_107_kernel_v_read_readvariableop1savev2_adam_conv3d_107_bias_v_read_readvariableop3savev2_adam_conv3d_108_kernel_v_read_readvariableop1savev2_adam_conv3d_108_bias_v_read_readvariableop>savev2_adam_batch_normalization_50_gamma_v_read_readvariableop=savev2_adam_batch_normalization_50_beta_v_read_readvariableop3savev2_adam_conv3d_109_kernel_v_read_readvariableop1savev2_adam_conv3d_109_bias_v_read_readvariableop>savev2_adam_batch_normalization_51_gamma_v_read_readvariableop=savev2_adam_batch_normalization_51_beta_v_read_readvariableop3savev2_adam_conv3d_110_kernel_v_read_readvariableop1savev2_adam_conv3d_110_bias_v_read_readvariableop1savev2_adam_dense_61_kernel_v_read_readvariableop/savev2_adam_dense_61_bias_v_read_readvariableop1savev2_adam_dense_62_kernel_v_read_readvariableop/savev2_adam_dense_62_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::::::::::::::::::
??@:@:@:: : : : : : : : : :::::::::::::::::
??@:@:@::::::::::::::::::
??@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0	,
*
_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
??@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::
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
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :0$,
*
_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::0(,
*
_output_shapes
:: )

_output_shapes
::0*,
*
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::0.,
*
_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::02,
*
_output_shapes
:: 3

_output_shapes
::&4"
 
_output_shapes
:
??@: 5

_output_shapes
:@:$6 

_output_shapes

:@: 7

_output_shapes
::08,
*
_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
::0<,
*
_output_shapes
:: =

_output_shapes
::0>,
*
_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
::0B,
*
_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::0F,
*
_output_shapes
:: G

_output_shapes
::&H"
 
_output_shapes
:
??@: I

_output_shapes
:@:$J 

_output_shapes

:@: K

_output_shapes
::L

_output_shapes
: 
?
d
E__inference_dropout_44_layer_call_and_return_conditional_losses_48563

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_49_layer_call_fn_49821

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_478072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49934

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_43_layer_call_fn_50096

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_43_layer_call_and_return_conditional_losses_483372
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
E__inference_conv3d_109_layer_call_and_return_conditional_losses_50112

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

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
??
?
 __inference__wrapped_model_47723
conv3d_106_inputU
7sequential_22_conv3d_106_conv3d_readvariableop_resource:F
8sequential_22_conv3d_106_biasadd_readvariableop_resource:T
Fsequential_22_batch_normalization_49_batchnorm_readvariableop_resource:X
Jsequential_22_batch_normalization_49_batchnorm_mul_readvariableop_resource:V
Hsequential_22_batch_normalization_49_batchnorm_readvariableop_1_resource:V
Hsequential_22_batch_normalization_49_batchnorm_readvariableop_2_resource:U
7sequential_22_conv3d_107_conv3d_readvariableop_resource:F
8sequential_22_conv3d_107_biasadd_readvariableop_resource:U
7sequential_22_conv3d_108_conv3d_readvariableop_resource:F
8sequential_22_conv3d_108_biasadd_readvariableop_resource:T
Fsequential_22_batch_normalization_50_batchnorm_readvariableop_resource:X
Jsequential_22_batch_normalization_50_batchnorm_mul_readvariableop_resource:V
Hsequential_22_batch_normalization_50_batchnorm_readvariableop_1_resource:V
Hsequential_22_batch_normalization_50_batchnorm_readvariableop_2_resource:U
7sequential_22_conv3d_109_conv3d_readvariableop_resource:F
8sequential_22_conv3d_109_biasadd_readvariableop_resource:T
Fsequential_22_batch_normalization_51_batchnorm_readvariableop_resource:X
Jsequential_22_batch_normalization_51_batchnorm_mul_readvariableop_resource:V
Hsequential_22_batch_normalization_51_batchnorm_readvariableop_1_resource:V
Hsequential_22_batch_normalization_51_batchnorm_readvariableop_2_resource:U
7sequential_22_conv3d_110_conv3d_readvariableop_resource:F
8sequential_22_conv3d_110_biasadd_readvariableop_resource:I
5sequential_22_dense_61_matmul_readvariableop_resource:
??@D
6sequential_22_dense_61_biasadd_readvariableop_resource:@G
5sequential_22_dense_62_matmul_readvariableop_resource:@D
6sequential_22_dense_62_biasadd_readvariableop_resource:
identity??=sequential_22/batch_normalization_49/batchnorm/ReadVariableOp??sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_1??sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_2?Asequential_22/batch_normalization_49/batchnorm/mul/ReadVariableOp?=sequential_22/batch_normalization_50/batchnorm/ReadVariableOp??sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_1??sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_2?Asequential_22/batch_normalization_50/batchnorm/mul/ReadVariableOp?=sequential_22/batch_normalization_51/batchnorm/ReadVariableOp??sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_1??sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_2?Asequential_22/batch_normalization_51/batchnorm/mul/ReadVariableOp?/sequential_22/conv3d_106/BiasAdd/ReadVariableOp?.sequential_22/conv3d_106/Conv3D/ReadVariableOp?/sequential_22/conv3d_107/BiasAdd/ReadVariableOp?.sequential_22/conv3d_107/Conv3D/ReadVariableOp?/sequential_22/conv3d_108/BiasAdd/ReadVariableOp?.sequential_22/conv3d_108/Conv3D/ReadVariableOp?/sequential_22/conv3d_109/BiasAdd/ReadVariableOp?.sequential_22/conv3d_109/Conv3D/ReadVariableOp?/sequential_22/conv3d_110/BiasAdd/ReadVariableOp?.sequential_22/conv3d_110/Conv3D/ReadVariableOp?-sequential_22/dense_61/BiasAdd/ReadVariableOp?,sequential_22/dense_61/MatMul/ReadVariableOp?-sequential_22/dense_62/BiasAdd/ReadVariableOp?,sequential_22/dense_62/MatMul/ReadVariableOp?
.sequential_22/conv3d_106/Conv3D/ReadVariableOpReadVariableOp7sequential_22_conv3d_106_conv3d_readvariableop_resource**
_output_shapes
:*
dtype020
.sequential_22/conv3d_106/Conv3D/ReadVariableOp?
sequential_22/conv3d_106/Conv3DConv3Dconv3d_106_input6sequential_22/conv3d_106/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2!
sequential_22/conv3d_106/Conv3D?
/sequential_22/conv3d_106/BiasAdd/ReadVariableOpReadVariableOp8sequential_22_conv3d_106_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_22/conv3d_106/BiasAdd/ReadVariableOp?
 sequential_22/conv3d_106/BiasAddBiasAdd(sequential_22/conv3d_106/Conv3D:output:07sequential_22/conv3d_106/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222"
 sequential_22/conv3d_106/BiasAdd?
sequential_22/conv3d_106/ReluRelu)sequential_22/conv3d_106/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
sequential_22/conv3d_106/Relu?
=sequential_22/batch_normalization_49/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_49_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_22/batch_normalization_49/batchnorm/ReadVariableOp?
4sequential_22/batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:26
4sequential_22/batch_normalization_49/batchnorm/add/y?
2sequential_22/batch_normalization_49/batchnorm/addAddV2Esequential_22/batch_normalization_49/batchnorm/ReadVariableOp:value:0=sequential_22/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_49/batchnorm/add?
4sequential_22/batch_normalization_49/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_49/batchnorm/Rsqrt?
Asequential_22/batch_normalization_49/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_49_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_22/batch_normalization_49/batchnorm/mul/ReadVariableOp?
2sequential_22/batch_normalization_49/batchnorm/mulMul8sequential_22/batch_normalization_49/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_49/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_49/batchnorm/mul?
4sequential_22/batch_normalization_49/batchnorm/mul_1Mul+sequential_22/conv3d_106/Relu:activations:06sequential_22/batch_normalization_49/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????22226
4sequential_22/batch_normalization_49/batchnorm/mul_1?
?sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_22_batch_normalization_49_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_1?
4sequential_22/batch_normalization_49/batchnorm/mul_2MulGsequential_22/batch_normalization_49/batchnorm/ReadVariableOp_1:value:06sequential_22/batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_49/batchnorm/mul_2?
?sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_22_batch_normalization_49_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_2?
2sequential_22/batch_normalization_49/batchnorm/subSubGsequential_22/batch_normalization_49/batchnorm/ReadVariableOp_2:value:08sequential_22/batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_49/batchnorm/sub?
4sequential_22/batch_normalization_49/batchnorm/add_1AddV28sequential_22/batch_normalization_49/batchnorm/mul_1:z:06sequential_22/batch_normalization_49/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????22226
4sequential_22/batch_normalization_49/batchnorm/add_1?
.sequential_22/conv3d_107/Conv3D/ReadVariableOpReadVariableOp7sequential_22_conv3d_107_conv3d_readvariableop_resource**
_output_shapes
:*
dtype020
.sequential_22/conv3d_107/Conv3D/ReadVariableOp?
sequential_22/conv3d_107/Conv3DConv3D8sequential_22/batch_normalization_49/batchnorm/add_1:z:06sequential_22/conv3d_107/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2!
sequential_22/conv3d_107/Conv3D?
/sequential_22/conv3d_107/BiasAdd/ReadVariableOpReadVariableOp8sequential_22_conv3d_107_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_22/conv3d_107/BiasAdd/ReadVariableOp?
 sequential_22/conv3d_107/BiasAddBiasAdd(sequential_22/conv3d_107/Conv3D:output:07sequential_22/conv3d_107/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222"
 sequential_22/conv3d_107/BiasAdd?
 sequential_22/conv3d_107/SigmoidSigmoid)sequential_22/conv3d_107/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222"
 sequential_22/conv3d_107/Sigmoid?
!sequential_22/dropout_42/IdentityIdentity$sequential_22/conv3d_107/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????2222#
!sequential_22/dropout_42/Identity?
.sequential_22/conv3d_108/Conv3D/ReadVariableOpReadVariableOp7sequential_22_conv3d_108_conv3d_readvariableop_resource**
_output_shapes
:*
dtype020
.sequential_22/conv3d_108/Conv3D/ReadVariableOp?
sequential_22/conv3d_108/Conv3DConv3D*sequential_22/dropout_42/Identity:output:06sequential_22/conv3d_108/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2!
sequential_22/conv3d_108/Conv3D?
/sequential_22/conv3d_108/BiasAdd/ReadVariableOpReadVariableOp8sequential_22_conv3d_108_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_22/conv3d_108/BiasAdd/ReadVariableOp?
 sequential_22/conv3d_108/BiasAddBiasAdd(sequential_22/conv3d_108/Conv3D:output:07sequential_22/conv3d_108/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222"
 sequential_22/conv3d_108/BiasAdd?
 sequential_22/conv3d_108/SigmoidSigmoid)sequential_22/conv3d_108/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222"
 sequential_22/conv3d_108/Sigmoid?
=sequential_22/batch_normalization_50/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_50_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_22/batch_normalization_50/batchnorm/ReadVariableOp?
4sequential_22/batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:26
4sequential_22/batch_normalization_50/batchnorm/add/y?
2sequential_22/batch_normalization_50/batchnorm/addAddV2Esequential_22/batch_normalization_50/batchnorm/ReadVariableOp:value:0=sequential_22/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_50/batchnorm/add?
4sequential_22/batch_normalization_50/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_50/batchnorm/Rsqrt?
Asequential_22/batch_normalization_50/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_50_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_22/batch_normalization_50/batchnorm/mul/ReadVariableOp?
2sequential_22/batch_normalization_50/batchnorm/mulMul8sequential_22/batch_normalization_50/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_50/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_50/batchnorm/mul?
4sequential_22/batch_normalization_50/batchnorm/mul_1Mul$sequential_22/conv3d_108/Sigmoid:y:06sequential_22/batch_normalization_50/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????22226
4sequential_22/batch_normalization_50/batchnorm/mul_1?
?sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_22_batch_normalization_50_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_1?
4sequential_22/batch_normalization_50/batchnorm/mul_2MulGsequential_22/batch_normalization_50/batchnorm/ReadVariableOp_1:value:06sequential_22/batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_50/batchnorm/mul_2?
?sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_22_batch_normalization_50_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_2?
2sequential_22/batch_normalization_50/batchnorm/subSubGsequential_22/batch_normalization_50/batchnorm/ReadVariableOp_2:value:08sequential_22/batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_50/batchnorm/sub?
4sequential_22/batch_normalization_50/batchnorm/add_1AddV28sequential_22/batch_normalization_50/batchnorm/mul_1:z:06sequential_22/batch_normalization_50/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????22226
4sequential_22/batch_normalization_50/batchnorm/add_1?
!sequential_22/dropout_43/IdentityIdentity8sequential_22/batch_normalization_50/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????2222#
!sequential_22/dropout_43/Identity?
.sequential_22/conv3d_109/Conv3D/ReadVariableOpReadVariableOp7sequential_22_conv3d_109_conv3d_readvariableop_resource**
_output_shapes
:*
dtype020
.sequential_22/conv3d_109/Conv3D/ReadVariableOp?
sequential_22/conv3d_109/Conv3DConv3D*sequential_22/dropout_43/Identity:output:06sequential_22/conv3d_109/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2!
sequential_22/conv3d_109/Conv3D?
/sequential_22/conv3d_109/BiasAdd/ReadVariableOpReadVariableOp8sequential_22_conv3d_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_22/conv3d_109/BiasAdd/ReadVariableOp?
 sequential_22/conv3d_109/BiasAddBiasAdd(sequential_22/conv3d_109/Conv3D:output:07sequential_22/conv3d_109/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222"
 sequential_22/conv3d_109/BiasAdd?
sequential_22/conv3d_109/ReluRelu)sequential_22/conv3d_109/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
sequential_22/conv3d_109/Relu?
=sequential_22/batch_normalization_51/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_51_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_22/batch_normalization_51/batchnorm/ReadVariableOp?
4sequential_22/batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:26
4sequential_22/batch_normalization_51/batchnorm/add/y?
2sequential_22/batch_normalization_51/batchnorm/addAddV2Esequential_22/batch_normalization_51/batchnorm/ReadVariableOp:value:0=sequential_22/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_51/batchnorm/add?
4sequential_22/batch_normalization_51/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_51/batchnorm/Rsqrt?
Asequential_22/batch_normalization_51/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_51_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_22/batch_normalization_51/batchnorm/mul/ReadVariableOp?
2sequential_22/batch_normalization_51/batchnorm/mulMul8sequential_22/batch_normalization_51/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_51/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_51/batchnorm/mul?
4sequential_22/batch_normalization_51/batchnorm/mul_1Mul+sequential_22/conv3d_109/Relu:activations:06sequential_22/batch_normalization_51/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????22226
4sequential_22/batch_normalization_51/batchnorm/mul_1?
?sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_22_batch_normalization_51_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_1?
4sequential_22/batch_normalization_51/batchnorm/mul_2MulGsequential_22/batch_normalization_51/batchnorm/ReadVariableOp_1:value:06sequential_22/batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_22/batch_normalization_51/batchnorm/mul_2?
?sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_22_batch_normalization_51_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_2?
2sequential_22/batch_normalization_51/batchnorm/subSubGsequential_22/batch_normalization_51/batchnorm/ReadVariableOp_2:value:08sequential_22/batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_22/batch_normalization_51/batchnorm/sub?
4sequential_22/batch_normalization_51/batchnorm/add_1AddV28sequential_22/batch_normalization_51/batchnorm/mul_1:z:06sequential_22/batch_normalization_51/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????22226
4sequential_22/batch_normalization_51/batchnorm/add_1?
!sequential_22/dropout_44/IdentityIdentity8sequential_22/batch_normalization_51/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????2222#
!sequential_22/dropout_44/Identity?
.sequential_22/conv3d_110/Conv3D/ReadVariableOpReadVariableOp7sequential_22_conv3d_110_conv3d_readvariableop_resource**
_output_shapes
:*
dtype020
.sequential_22/conv3d_110/Conv3D/ReadVariableOp?
sequential_22/conv3d_110/Conv3DConv3D*sequential_22/dropout_44/Identity:output:06sequential_22/conv3d_110/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2!
sequential_22/conv3d_110/Conv3D?
/sequential_22/conv3d_110/BiasAdd/ReadVariableOpReadVariableOp8sequential_22_conv3d_110_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_22/conv3d_110/BiasAdd/ReadVariableOp?
 sequential_22/conv3d_110/BiasAddBiasAdd(sequential_22/conv3d_110/Conv3D:output:07sequential_22/conv3d_110/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222"
 sequential_22/conv3d_110/BiasAdd?
 sequential_22/conv3d_110/SigmoidSigmoid)sequential_22/conv3d_110/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222"
 sequential_22/conv3d_110/Sigmoid?
sequential_22/flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2 
sequential_22/flatten_22/Const?
 sequential_22/flatten_22/ReshapeReshape$sequential_22/conv3d_110/Sigmoid:y:0'sequential_22/flatten_22/Const:output:0*
T0*)
_output_shapes
:???????????2"
 sequential_22/flatten_22/Reshape?
,sequential_22/dense_61/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_61_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02.
,sequential_22/dense_61/MatMul/ReadVariableOp?
sequential_22/dense_61/MatMulMatMul)sequential_22/flatten_22/Reshape:output:04sequential_22/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_22/dense_61/MatMul?
-sequential_22/dense_61/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_22/dense_61/BiasAdd/ReadVariableOp?
sequential_22/dense_61/BiasAddBiasAdd'sequential_22/dense_61/MatMul:product:05sequential_22/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_22/dense_61/BiasAdd?
sequential_22/dense_61/ReluRelu'sequential_22/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_22/dense_61/Relu?
,sequential_22/dense_62/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_62_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential_22/dense_62/MatMul/ReadVariableOp?
sequential_22/dense_62/MatMulMatMul)sequential_22/dense_61/Relu:activations:04sequential_22/dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_22/dense_62/MatMul?
-sequential_22/dense_62/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_22/dense_62/BiasAdd/ReadVariableOp?
sequential_22/dense_62/BiasAddBiasAdd'sequential_22/dense_62/MatMul:product:05sequential_22/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_22/dense_62/BiasAdd?
sequential_22/dense_62/SigmoidSigmoid'sequential_22/dense_62/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_22/dense_62/Sigmoid?
IdentityIdentity"sequential_22/dense_62/Sigmoid:y:0>^sequential_22/batch_normalization_49/batchnorm/ReadVariableOp@^sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_1@^sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_2B^sequential_22/batch_normalization_49/batchnorm/mul/ReadVariableOp>^sequential_22/batch_normalization_50/batchnorm/ReadVariableOp@^sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_1@^sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_2B^sequential_22/batch_normalization_50/batchnorm/mul/ReadVariableOp>^sequential_22/batch_normalization_51/batchnorm/ReadVariableOp@^sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_1@^sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_2B^sequential_22/batch_normalization_51/batchnorm/mul/ReadVariableOp0^sequential_22/conv3d_106/BiasAdd/ReadVariableOp/^sequential_22/conv3d_106/Conv3D/ReadVariableOp0^sequential_22/conv3d_107/BiasAdd/ReadVariableOp/^sequential_22/conv3d_107/Conv3D/ReadVariableOp0^sequential_22/conv3d_108/BiasAdd/ReadVariableOp/^sequential_22/conv3d_108/Conv3D/ReadVariableOp0^sequential_22/conv3d_109/BiasAdd/ReadVariableOp/^sequential_22/conv3d_109/Conv3D/ReadVariableOp0^sequential_22/conv3d_110/BiasAdd/ReadVariableOp/^sequential_22/conv3d_110/Conv3D/ReadVariableOp.^sequential_22/dense_61/BiasAdd/ReadVariableOp-^sequential_22/dense_61/MatMul/ReadVariableOp.^sequential_22/dense_62/BiasAdd/ReadVariableOp-^sequential_22/dense_62/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 2~
=sequential_22/batch_normalization_49/batchnorm/ReadVariableOp=sequential_22/batch_normalization_49/batchnorm/ReadVariableOp2?
?sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_1?sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_12?
?sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_2?sequential_22/batch_normalization_49/batchnorm/ReadVariableOp_22?
Asequential_22/batch_normalization_49/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_49/batchnorm/mul/ReadVariableOp2~
=sequential_22/batch_normalization_50/batchnorm/ReadVariableOp=sequential_22/batch_normalization_50/batchnorm/ReadVariableOp2?
?sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_1?sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_12?
?sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_2?sequential_22/batch_normalization_50/batchnorm/ReadVariableOp_22?
Asequential_22/batch_normalization_50/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_50/batchnorm/mul/ReadVariableOp2~
=sequential_22/batch_normalization_51/batchnorm/ReadVariableOp=sequential_22/batch_normalization_51/batchnorm/ReadVariableOp2?
?sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_1?sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_12?
?sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_2?sequential_22/batch_normalization_51/batchnorm/ReadVariableOp_22?
Asequential_22/batch_normalization_51/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_51/batchnorm/mul/ReadVariableOp2b
/sequential_22/conv3d_106/BiasAdd/ReadVariableOp/sequential_22/conv3d_106/BiasAdd/ReadVariableOp2`
.sequential_22/conv3d_106/Conv3D/ReadVariableOp.sequential_22/conv3d_106/Conv3D/ReadVariableOp2b
/sequential_22/conv3d_107/BiasAdd/ReadVariableOp/sequential_22/conv3d_107/BiasAdd/ReadVariableOp2`
.sequential_22/conv3d_107/Conv3D/ReadVariableOp.sequential_22/conv3d_107/Conv3D/ReadVariableOp2b
/sequential_22/conv3d_108/BiasAdd/ReadVariableOp/sequential_22/conv3d_108/BiasAdd/ReadVariableOp2`
.sequential_22/conv3d_108/Conv3D/ReadVariableOp.sequential_22/conv3d_108/Conv3D/ReadVariableOp2b
/sequential_22/conv3d_109/BiasAdd/ReadVariableOp/sequential_22/conv3d_109/BiasAdd/ReadVariableOp2`
.sequential_22/conv3d_109/Conv3D/ReadVariableOp.sequential_22/conv3d_109/Conv3D/ReadVariableOp2b
/sequential_22/conv3d_110/BiasAdd/ReadVariableOp/sequential_22/conv3d_110/BiasAdd/ReadVariableOp2`
.sequential_22/conv3d_110/Conv3D/ReadVariableOp.sequential_22/conv3d_110/Conv3D/ReadVariableOp2^
-sequential_22/dense_61/BiasAdd/ReadVariableOp-sequential_22/dense_61/BiasAdd/ReadVariableOp2\
,sequential_22/dense_61/MatMul/ReadVariableOp,sequential_22/dense_61/MatMul/ReadVariableOp2^
-sequential_22/dense_62/BiasAdd/ReadVariableOp-sequential_22/dense_62/BiasAdd/ReadVariableOp2\
,sequential_22/dense_62/MatMul/ReadVariableOp,sequential_22/dense_62/MatMul/ReadVariableOp:e a
3
_output_shapes!
:?????????222
*
_user_specified_nameconv3d_106_input
?
a
E__inference_flatten_22_layer_call_and_return_conditional_losses_50334

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49707

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49968

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv3d_106_layer_call_and_return_conditional_losses_49678

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????222: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_49278
conv3d_106_input%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:(

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
??@

unknown_22:@

unknown_23:@

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_106_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_477232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
3
_output_shapes!
:?????????222
*
_user_specified_nameconv3d_106_input
??
?
H__inference_sequential_22_layer_call_and_return_conditional_losses_49553

inputsG
)conv3d_106_conv3d_readvariableop_resource:8
*conv3d_106_biasadd_readvariableop_resource:L
>batch_normalization_49_assignmovingavg_readvariableop_resource:N
@batch_normalization_49_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_49_batchnorm_mul_readvariableop_resource:F
8batch_normalization_49_batchnorm_readvariableop_resource:G
)conv3d_107_conv3d_readvariableop_resource:8
*conv3d_107_biasadd_readvariableop_resource:G
)conv3d_108_conv3d_readvariableop_resource:8
*conv3d_108_biasadd_readvariableop_resource:L
>batch_normalization_50_assignmovingavg_readvariableop_resource:N
@batch_normalization_50_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_50_batchnorm_mul_readvariableop_resource:F
8batch_normalization_50_batchnorm_readvariableop_resource:G
)conv3d_109_conv3d_readvariableop_resource:8
*conv3d_109_biasadd_readvariableop_resource:L
>batch_normalization_51_assignmovingavg_readvariableop_resource:N
@batch_normalization_51_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_51_batchnorm_mul_readvariableop_resource:F
8batch_normalization_51_batchnorm_readvariableop_resource:G
)conv3d_110_conv3d_readvariableop_resource:8
*conv3d_110_biasadd_readvariableop_resource:;
'dense_61_matmul_readvariableop_resource:
??@6
(dense_61_biasadd_readvariableop_resource:@9
'dense_62_matmul_readvariableop_resource:@6
(dense_62_biasadd_readvariableop_resource:
identity??&batch_normalization_49/AssignMovingAvg?5batch_normalization_49/AssignMovingAvg/ReadVariableOp?(batch_normalization_49/AssignMovingAvg_1?7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_49/batchnorm/ReadVariableOp?3batch_normalization_49/batchnorm/mul/ReadVariableOp?&batch_normalization_50/AssignMovingAvg?5batch_normalization_50/AssignMovingAvg/ReadVariableOp?(batch_normalization_50/AssignMovingAvg_1?7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_50/batchnorm/ReadVariableOp?3batch_normalization_50/batchnorm/mul/ReadVariableOp?&batch_normalization_51/AssignMovingAvg?5batch_normalization_51/AssignMovingAvg/ReadVariableOp?(batch_normalization_51/AssignMovingAvg_1?7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_51/batchnorm/ReadVariableOp?3batch_normalization_51/batchnorm/mul/ReadVariableOp?!conv3d_106/BiasAdd/ReadVariableOp? conv3d_106/Conv3D/ReadVariableOp?!conv3d_107/BiasAdd/ReadVariableOp? conv3d_107/Conv3D/ReadVariableOp?!conv3d_108/BiasAdd/ReadVariableOp? conv3d_108/Conv3D/ReadVariableOp?!conv3d_109/BiasAdd/ReadVariableOp? conv3d_109/Conv3D/ReadVariableOp?!conv3d_110/BiasAdd/ReadVariableOp? conv3d_110/Conv3D/ReadVariableOp?dense_61/BiasAdd/ReadVariableOp?dense_61/MatMul/ReadVariableOp?dense_62/BiasAdd/ReadVariableOp?dense_62/MatMul/ReadVariableOp?
 conv3d_106/Conv3D/ReadVariableOpReadVariableOp)conv3d_106_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_106/Conv3D/ReadVariableOp?
conv3d_106/Conv3DConv3Dinputs(conv3d_106/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_106/Conv3D?
!conv3d_106/BiasAdd/ReadVariableOpReadVariableOp*conv3d_106_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_106/BiasAdd/ReadVariableOp?
conv3d_106/BiasAddBiasAddconv3d_106/Conv3D:output:0)conv3d_106/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_106/BiasAdd?
conv3d_106/ReluReluconv3d_106/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_106/Relu?
5batch_normalization_49/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_49/moments/mean/reduction_indices?
#batch_normalization_49/moments/meanMeanconv3d_106/Relu:activations:0>batch_normalization_49/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2%
#batch_normalization_49/moments/mean?
+batch_normalization_49/moments/StopGradientStopGradient,batch_normalization_49/moments/mean:output:0*
T0**
_output_shapes
:2-
+batch_normalization_49/moments/StopGradient?
0batch_normalization_49/moments/SquaredDifferenceSquaredDifferenceconv3d_106/Relu:activations:04batch_normalization_49/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????22222
0batch_normalization_49/moments/SquaredDifference?
9batch_normalization_49/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_49/moments/variance/reduction_indices?
'batch_normalization_49/moments/varianceMean4batch_normalization_49/moments/SquaredDifference:z:0Bbatch_normalization_49/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2)
'batch_normalization_49/moments/variance?
&batch_normalization_49/moments/SqueezeSqueeze,batch_normalization_49/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_49/moments/Squeeze?
(batch_normalization_49/moments/Squeeze_1Squeeze0batch_normalization_49/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_49/moments/Squeeze_1?
,batch_normalization_49/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_49/AssignMovingAvg/decay?
5batch_normalization_49/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_49_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_49/AssignMovingAvg/ReadVariableOp?
*batch_normalization_49/AssignMovingAvg/subSub=batch_normalization_49/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_49/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_49/AssignMovingAvg/sub?
*batch_normalization_49/AssignMovingAvg/mulMul.batch_normalization_49/AssignMovingAvg/sub:z:05batch_normalization_49/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2,
*batch_normalization_49/AssignMovingAvg/mul?
&batch_normalization_49/AssignMovingAvgAssignSubVariableOp>batch_normalization_49_assignmovingavg_readvariableop_resource.batch_normalization_49/AssignMovingAvg/mul:z:06^batch_normalization_49/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_49/AssignMovingAvg?
.batch_normalization_49/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_49/AssignMovingAvg_1/decay?
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_49_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_49/AssignMovingAvg_1/subSub?batch_normalization_49/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_49/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_49/AssignMovingAvg_1/sub?
,batch_normalization_49/AssignMovingAvg_1/mulMul0batch_normalization_49/AssignMovingAvg_1/sub:z:07batch_normalization_49/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2.
,batch_normalization_49/AssignMovingAvg_1/mul?
(batch_normalization_49/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_49_assignmovingavg_1_readvariableop_resource0batch_normalization_49/AssignMovingAvg_1/mul:z:08^batch_normalization_49/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_49/AssignMovingAvg_1?
&batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_49/batchnorm/add/y?
$batch_normalization_49/batchnorm/addAddV21batch_normalization_49/moments/Squeeze_1:output:0/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/add?
&batch_normalization_49/batchnorm/RsqrtRsqrt(batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/Rsqrt?
3batch_normalization_49/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_49_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_49/batchnorm/mul/ReadVariableOp?
$batch_normalization_49/batchnorm/mulMul*batch_normalization_49/batchnorm/Rsqrt:y:0;batch_normalization_49/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/mul?
&batch_normalization_49/batchnorm/mul_1Mulconv3d_106/Relu:activations:0(batch_normalization_49/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_49/batchnorm/mul_1?
&batch_normalization_49/batchnorm/mul_2Mul/batch_normalization_49/moments/Squeeze:output:0(batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/mul_2?
/batch_normalization_49/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_49_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_49/batchnorm/ReadVariableOp?
$batch_normalization_49/batchnorm/subSub7batch_normalization_49/batchnorm/ReadVariableOp:value:0*batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/sub?
&batch_normalization_49/batchnorm/add_1AddV2*batch_normalization_49/batchnorm/mul_1:z:0(batch_normalization_49/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_49/batchnorm/add_1?
 conv3d_107/Conv3D/ReadVariableOpReadVariableOp)conv3d_107_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_107/Conv3D/ReadVariableOp?
conv3d_107/Conv3DConv3D*batch_normalization_49/batchnorm/add_1:z:0(conv3d_107/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_107/Conv3D?
!conv3d_107/BiasAdd/ReadVariableOpReadVariableOp*conv3d_107_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_107/BiasAdd/ReadVariableOp?
conv3d_107/BiasAddBiasAddconv3d_107/Conv3D:output:0)conv3d_107/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_107/BiasAdd?
conv3d_107/SigmoidSigmoidconv3d_107/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_107/Sigmoidy
dropout_42/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_42/dropout/Const?
dropout_42/dropout/MulMulconv3d_107/Sigmoid:y:0!dropout_42/dropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout_42/dropout/Mulz
dropout_42/dropout/ShapeShapeconv3d_107/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_42/dropout/Shape?
/dropout_42/dropout/random_uniform/RandomUniformRandomUniform!dropout_42/dropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype021
/dropout_42/dropout/random_uniform/RandomUniform?
!dropout_42/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_42/dropout/GreaterEqual/y?
dropout_42/dropout/GreaterEqualGreaterEqual8dropout_42/dropout/random_uniform/RandomUniform:output:0*dropout_42/dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222!
dropout_42/dropout/GreaterEqual?
dropout_42/dropout/CastCast#dropout_42/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout_42/dropout/Cast?
dropout_42/dropout/Mul_1Muldropout_42/dropout/Mul:z:0dropout_42/dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout_42/dropout/Mul_1?
 conv3d_108/Conv3D/ReadVariableOpReadVariableOp)conv3d_108_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_108/Conv3D/ReadVariableOp?
conv3d_108/Conv3DConv3Ddropout_42/dropout/Mul_1:z:0(conv3d_108/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_108/Conv3D?
!conv3d_108/BiasAdd/ReadVariableOpReadVariableOp*conv3d_108_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_108/BiasAdd/ReadVariableOp?
conv3d_108/BiasAddBiasAddconv3d_108/Conv3D:output:0)conv3d_108/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_108/BiasAdd?
conv3d_108/SigmoidSigmoidconv3d_108/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_108/Sigmoid?
5batch_normalization_50/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_50/moments/mean/reduction_indices?
#batch_normalization_50/moments/meanMeanconv3d_108/Sigmoid:y:0>batch_normalization_50/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2%
#batch_normalization_50/moments/mean?
+batch_normalization_50/moments/StopGradientStopGradient,batch_normalization_50/moments/mean:output:0*
T0**
_output_shapes
:2-
+batch_normalization_50/moments/StopGradient?
0batch_normalization_50/moments/SquaredDifferenceSquaredDifferenceconv3d_108/Sigmoid:y:04batch_normalization_50/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????22222
0batch_normalization_50/moments/SquaredDifference?
9batch_normalization_50/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_50/moments/variance/reduction_indices?
'batch_normalization_50/moments/varianceMean4batch_normalization_50/moments/SquaredDifference:z:0Bbatch_normalization_50/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2)
'batch_normalization_50/moments/variance?
&batch_normalization_50/moments/SqueezeSqueeze,batch_normalization_50/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_50/moments/Squeeze?
(batch_normalization_50/moments/Squeeze_1Squeeze0batch_normalization_50/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_50/moments/Squeeze_1?
,batch_normalization_50/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_50/AssignMovingAvg/decay?
5batch_normalization_50/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_50_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_50/AssignMovingAvg/ReadVariableOp?
*batch_normalization_50/AssignMovingAvg/subSub=batch_normalization_50/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_50/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_50/AssignMovingAvg/sub?
*batch_normalization_50/AssignMovingAvg/mulMul.batch_normalization_50/AssignMovingAvg/sub:z:05batch_normalization_50/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2,
*batch_normalization_50/AssignMovingAvg/mul?
&batch_normalization_50/AssignMovingAvgAssignSubVariableOp>batch_normalization_50_assignmovingavg_readvariableop_resource.batch_normalization_50/AssignMovingAvg/mul:z:06^batch_normalization_50/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_50/AssignMovingAvg?
.batch_normalization_50/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_50/AssignMovingAvg_1/decay?
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_50_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_50/AssignMovingAvg_1/subSub?batch_normalization_50/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_50/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_50/AssignMovingAvg_1/sub?
,batch_normalization_50/AssignMovingAvg_1/mulMul0batch_normalization_50/AssignMovingAvg_1/sub:z:07batch_normalization_50/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2.
,batch_normalization_50/AssignMovingAvg_1/mul?
(batch_normalization_50/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_50_assignmovingavg_1_readvariableop_resource0batch_normalization_50/AssignMovingAvg_1/mul:z:08^batch_normalization_50/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_50/AssignMovingAvg_1?
&batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_50/batchnorm/add/y?
$batch_normalization_50/batchnorm/addAddV21batch_normalization_50/moments/Squeeze_1:output:0/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/add?
&batch_normalization_50/batchnorm/RsqrtRsqrt(batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/Rsqrt?
3batch_normalization_50/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_50_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_50/batchnorm/mul/ReadVariableOp?
$batch_normalization_50/batchnorm/mulMul*batch_normalization_50/batchnorm/Rsqrt:y:0;batch_normalization_50/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/mul?
&batch_normalization_50/batchnorm/mul_1Mulconv3d_108/Sigmoid:y:0(batch_normalization_50/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_50/batchnorm/mul_1?
&batch_normalization_50/batchnorm/mul_2Mul/batch_normalization_50/moments/Squeeze:output:0(batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/mul_2?
/batch_normalization_50/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_50_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_50/batchnorm/ReadVariableOp?
$batch_normalization_50/batchnorm/subSub7batch_normalization_50/batchnorm/ReadVariableOp:value:0*batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/sub?
&batch_normalization_50/batchnorm/add_1AddV2*batch_normalization_50/batchnorm/mul_1:z:0(batch_normalization_50/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_50/batchnorm/add_1y
dropout_43/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_43/dropout/Const?
dropout_43/dropout/MulMul*batch_normalization_50/batchnorm/add_1:z:0!dropout_43/dropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout_43/dropout/Mul?
dropout_43/dropout/ShapeShape*batch_normalization_50/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_43/dropout/Shape?
/dropout_43/dropout/random_uniform/RandomUniformRandomUniform!dropout_43/dropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype021
/dropout_43/dropout/random_uniform/RandomUniform?
!dropout_43/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_43/dropout/GreaterEqual/y?
dropout_43/dropout/GreaterEqualGreaterEqual8dropout_43/dropout/random_uniform/RandomUniform:output:0*dropout_43/dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222!
dropout_43/dropout/GreaterEqual?
dropout_43/dropout/CastCast#dropout_43/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout_43/dropout/Cast?
dropout_43/dropout/Mul_1Muldropout_43/dropout/Mul:z:0dropout_43/dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout_43/dropout/Mul_1?
 conv3d_109/Conv3D/ReadVariableOpReadVariableOp)conv3d_109_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_109/Conv3D/ReadVariableOp?
conv3d_109/Conv3DConv3Ddropout_43/dropout/Mul_1:z:0(conv3d_109/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_109/Conv3D?
!conv3d_109/BiasAdd/ReadVariableOpReadVariableOp*conv3d_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_109/BiasAdd/ReadVariableOp?
conv3d_109/BiasAddBiasAddconv3d_109/Conv3D:output:0)conv3d_109/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_109/BiasAdd?
conv3d_109/ReluReluconv3d_109/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_109/Relu?
5batch_normalization_51/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_51/moments/mean/reduction_indices?
#batch_normalization_51/moments/meanMeanconv3d_109/Relu:activations:0>batch_normalization_51/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2%
#batch_normalization_51/moments/mean?
+batch_normalization_51/moments/StopGradientStopGradient,batch_normalization_51/moments/mean:output:0*
T0**
_output_shapes
:2-
+batch_normalization_51/moments/StopGradient?
0batch_normalization_51/moments/SquaredDifferenceSquaredDifferenceconv3d_109/Relu:activations:04batch_normalization_51/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????22222
0batch_normalization_51/moments/SquaredDifference?
9batch_normalization_51/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_51/moments/variance/reduction_indices?
'batch_normalization_51/moments/varianceMean4batch_normalization_51/moments/SquaredDifference:z:0Bbatch_normalization_51/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2)
'batch_normalization_51/moments/variance?
&batch_normalization_51/moments/SqueezeSqueeze,batch_normalization_51/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_51/moments/Squeeze?
(batch_normalization_51/moments/Squeeze_1Squeeze0batch_normalization_51/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_51/moments/Squeeze_1?
,batch_normalization_51/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_51/AssignMovingAvg/decay?
5batch_normalization_51/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_51_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_51/AssignMovingAvg/ReadVariableOp?
*batch_normalization_51/AssignMovingAvg/subSub=batch_normalization_51/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_51/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_51/AssignMovingAvg/sub?
*batch_normalization_51/AssignMovingAvg/mulMul.batch_normalization_51/AssignMovingAvg/sub:z:05batch_normalization_51/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2,
*batch_normalization_51/AssignMovingAvg/mul?
&batch_normalization_51/AssignMovingAvgAssignSubVariableOp>batch_normalization_51_assignmovingavg_readvariableop_resource.batch_normalization_51/AssignMovingAvg/mul:z:06^batch_normalization_51/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_51/AssignMovingAvg?
.batch_normalization_51/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_51/AssignMovingAvg_1/decay?
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_51_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_51/AssignMovingAvg_1/subSub?batch_normalization_51/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_51/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_51/AssignMovingAvg_1/sub?
,batch_normalization_51/AssignMovingAvg_1/mulMul0batch_normalization_51/AssignMovingAvg_1/sub:z:07batch_normalization_51/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2.
,batch_normalization_51/AssignMovingAvg_1/mul?
(batch_normalization_51/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_51_assignmovingavg_1_readvariableop_resource0batch_normalization_51/AssignMovingAvg_1/mul:z:08^batch_normalization_51/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_51/AssignMovingAvg_1?
&batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_51/batchnorm/add/y?
$batch_normalization_51/batchnorm/addAddV21batch_normalization_51/moments/Squeeze_1:output:0/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/add?
&batch_normalization_51/batchnorm/RsqrtRsqrt(batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/Rsqrt?
3batch_normalization_51/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_51_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_51/batchnorm/mul/ReadVariableOp?
$batch_normalization_51/batchnorm/mulMul*batch_normalization_51/batchnorm/Rsqrt:y:0;batch_normalization_51/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/mul?
&batch_normalization_51/batchnorm/mul_1Mulconv3d_109/Relu:activations:0(batch_normalization_51/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_51/batchnorm/mul_1?
&batch_normalization_51/batchnorm/mul_2Mul/batch_normalization_51/moments/Squeeze:output:0(batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/mul_2?
/batch_normalization_51/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_51_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_51/batchnorm/ReadVariableOp?
$batch_normalization_51/batchnorm/subSub7batch_normalization_51/batchnorm/ReadVariableOp:value:0*batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/sub?
&batch_normalization_51/batchnorm/add_1AddV2*batch_normalization_51/batchnorm/mul_1:z:0(batch_normalization_51/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222(
&batch_normalization_51/batchnorm/add_1y
dropout_44/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_44/dropout/Const?
dropout_44/dropout/MulMul*batch_normalization_51/batchnorm/add_1:z:0!dropout_44/dropout/Const:output:0*
T0*3
_output_shapes!
:?????????2222
dropout_44/dropout/Mul?
dropout_44/dropout/ShapeShape*batch_normalization_51/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_44/dropout/Shape?
/dropout_44/dropout/random_uniform/RandomUniformRandomUniform!dropout_44/dropout/Shape:output:0*
T0*3
_output_shapes!
:?????????222*
dtype021
/dropout_44/dropout/random_uniform/RandomUniform?
!dropout_44/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_44/dropout/GreaterEqual/y?
dropout_44/dropout/GreaterEqualGreaterEqual8dropout_44/dropout/random_uniform/RandomUniform:output:0*dropout_44/dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:?????????2222!
dropout_44/dropout/GreaterEqual?
dropout_44/dropout/CastCast#dropout_44/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:?????????2222
dropout_44/dropout/Cast?
dropout_44/dropout/Mul_1Muldropout_44/dropout/Mul:z:0dropout_44/dropout/Cast:y:0*
T0*3
_output_shapes!
:?????????2222
dropout_44/dropout/Mul_1?
 conv3d_110/Conv3D/ReadVariableOpReadVariableOp)conv3d_110_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02"
 conv3d_110/Conv3D/ReadVariableOp?
conv3d_110/Conv3DConv3Ddropout_44/dropout/Mul_1:z:0(conv3d_110/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
conv3d_110/Conv3D?
!conv3d_110/BiasAdd/ReadVariableOpReadVariableOp*conv3d_110_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv3d_110/BiasAdd/ReadVariableOp?
conv3d_110/BiasAddBiasAddconv3d_110/Conv3D:output:0)conv3d_110/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222
conv3d_110/BiasAdd?
conv3d_110/SigmoidSigmoidconv3d_110/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
conv3d_110/Sigmoidu
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
flatten_22/Const?
flatten_22/ReshapeReshapeconv3d_110/Sigmoid:y:0flatten_22/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_22/Reshape?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMulflatten_22/Reshape:output:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_61/MatMul?
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_61/BiasAdd/ReadVariableOp?
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_61/BiasAdds
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_61/Relu?
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_62/MatMul/ReadVariableOp?
dense_62/MatMulMatMuldense_61/Relu:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_62/MatMul?
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_62/BiasAdd/ReadVariableOp?
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_62/BiasAdd|
dense_62/SigmoidSigmoiddense_62/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_62/Sigmoid?
IdentityIdentitydense_62/Sigmoid:y:0'^batch_normalization_49/AssignMovingAvg6^batch_normalization_49/AssignMovingAvg/ReadVariableOp)^batch_normalization_49/AssignMovingAvg_18^batch_normalization_49/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_49/batchnorm/ReadVariableOp4^batch_normalization_49/batchnorm/mul/ReadVariableOp'^batch_normalization_50/AssignMovingAvg6^batch_normalization_50/AssignMovingAvg/ReadVariableOp)^batch_normalization_50/AssignMovingAvg_18^batch_normalization_50/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_50/batchnorm/ReadVariableOp4^batch_normalization_50/batchnorm/mul/ReadVariableOp'^batch_normalization_51/AssignMovingAvg6^batch_normalization_51/AssignMovingAvg/ReadVariableOp)^batch_normalization_51/AssignMovingAvg_18^batch_normalization_51/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_51/batchnorm/ReadVariableOp4^batch_normalization_51/batchnorm/mul/ReadVariableOp"^conv3d_106/BiasAdd/ReadVariableOp!^conv3d_106/Conv3D/ReadVariableOp"^conv3d_107/BiasAdd/ReadVariableOp!^conv3d_107/Conv3D/ReadVariableOp"^conv3d_108/BiasAdd/ReadVariableOp!^conv3d_108/Conv3D/ReadVariableOp"^conv3d_109/BiasAdd/ReadVariableOp!^conv3d_109/Conv3D/ReadVariableOp"^conv3d_110/BiasAdd/ReadVariableOp!^conv3d_110/Conv3D/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_49/AssignMovingAvg&batch_normalization_49/AssignMovingAvg2n
5batch_normalization_49/AssignMovingAvg/ReadVariableOp5batch_normalization_49/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_49/AssignMovingAvg_1(batch_normalization_49/AssignMovingAvg_12r
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_49/batchnorm/ReadVariableOp/batch_normalization_49/batchnorm/ReadVariableOp2j
3batch_normalization_49/batchnorm/mul/ReadVariableOp3batch_normalization_49/batchnorm/mul/ReadVariableOp2P
&batch_normalization_50/AssignMovingAvg&batch_normalization_50/AssignMovingAvg2n
5batch_normalization_50/AssignMovingAvg/ReadVariableOp5batch_normalization_50/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_50/AssignMovingAvg_1(batch_normalization_50/AssignMovingAvg_12r
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_50/batchnorm/ReadVariableOp/batch_normalization_50/batchnorm/ReadVariableOp2j
3batch_normalization_50/batchnorm/mul/ReadVariableOp3batch_normalization_50/batchnorm/mul/ReadVariableOp2P
&batch_normalization_51/AssignMovingAvg&batch_normalization_51/AssignMovingAvg2n
5batch_normalization_51/AssignMovingAvg/ReadVariableOp5batch_normalization_51/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_51/AssignMovingAvg_1(batch_normalization_51/AssignMovingAvg_12r
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_51/batchnorm/ReadVariableOp/batch_normalization_51/batchnorm/ReadVariableOp2j
3batch_normalization_51/batchnorm/mul/ReadVariableOp3batch_normalization_51/batchnorm/mul/ReadVariableOp2F
!conv3d_106/BiasAdd/ReadVariableOp!conv3d_106/BiasAdd/ReadVariableOp2D
 conv3d_106/Conv3D/ReadVariableOp conv3d_106/Conv3D/ReadVariableOp2F
!conv3d_107/BiasAdd/ReadVariableOp!conv3d_107/BiasAdd/ReadVariableOp2D
 conv3d_107/Conv3D/ReadVariableOp conv3d_107/Conv3D/ReadVariableOp2F
!conv3d_108/BiasAdd/ReadVariableOp!conv3d_108/BiasAdd/ReadVariableOp2D
 conv3d_108/Conv3D/ReadVariableOp conv3d_108/Conv3D/ReadVariableOp2F
!conv3d_109/BiasAdd/ReadVariableOp!conv3d_109/BiasAdd/ReadVariableOp2D
 conv3d_109/Conv3D/ReadVariableOp conv3d_109/Conv3D/ReadVariableOp2F
!conv3d_110/BiasAdd/ReadVariableOp!conv3d_110/BiasAdd/ReadVariableOp2D
 conv3d_110/Conv3D/ReadVariableOp conv3d_110/Conv3D/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
c
*__inference_dropout_44_layer_call_fn_50308

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_485632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22222
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?

?
C__inference_dense_62_layer_call_and_return_conditional_losses_50370

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
?
?
-__inference_sequential_22_layer_call_fn_49667

inputs%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:(

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
??@

unknown_22:@

unknown_23:@

unknown_24:
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_22_layer_call_and_return_conditional_losses_489612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_48615

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????2222
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_49_layer_call_fn_49847

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_488112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
c
E__inference_dropout_42_layer_call_and_return_conditional_losses_48284

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:?????????2222

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:?????????2222

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_48252

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
E__inference_conv3d_108_layer_call_and_return_conditional_losses_49905

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

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
??
?1
!__inference__traced_restore_50862
file_prefix@
"assignvariableop_conv3d_106_kernel:0
"assignvariableop_1_conv3d_106_bias:=
/assignvariableop_2_batch_normalization_49_gamma:<
.assignvariableop_3_batch_normalization_49_beta:C
5assignvariableop_4_batch_normalization_49_moving_mean:G
9assignvariableop_5_batch_normalization_49_moving_variance:B
$assignvariableop_6_conv3d_107_kernel:0
"assignvariableop_7_conv3d_107_bias:B
$assignvariableop_8_conv3d_108_kernel:0
"assignvariableop_9_conv3d_108_bias:>
0assignvariableop_10_batch_normalization_50_gamma:=
/assignvariableop_11_batch_normalization_50_beta:D
6assignvariableop_12_batch_normalization_50_moving_mean:H
:assignvariableop_13_batch_normalization_50_moving_variance:C
%assignvariableop_14_conv3d_109_kernel:1
#assignvariableop_15_conv3d_109_bias:>
0assignvariableop_16_batch_normalization_51_gamma:=
/assignvariableop_17_batch_normalization_51_beta:D
6assignvariableop_18_batch_normalization_51_moving_mean:H
:assignvariableop_19_batch_normalization_51_moving_variance:C
%assignvariableop_20_conv3d_110_kernel:1
#assignvariableop_21_conv3d_110_bias:7
#assignvariableop_22_dense_61_kernel:
??@/
!assignvariableop_23_dense_61_bias:@5
#assignvariableop_24_dense_62_kernel:@/
!assignvariableop_25_dense_62_bias:'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: #
assignvariableop_31_total: #
assignvariableop_32_count: %
assignvariableop_33_total_1: %
assignvariableop_34_count_1: J
,assignvariableop_35_adam_conv3d_106_kernel_m:8
*assignvariableop_36_adam_conv3d_106_bias_m:E
7assignvariableop_37_adam_batch_normalization_49_gamma_m:D
6assignvariableop_38_adam_batch_normalization_49_beta_m:J
,assignvariableop_39_adam_conv3d_107_kernel_m:8
*assignvariableop_40_adam_conv3d_107_bias_m:J
,assignvariableop_41_adam_conv3d_108_kernel_m:8
*assignvariableop_42_adam_conv3d_108_bias_m:E
7assignvariableop_43_adam_batch_normalization_50_gamma_m:D
6assignvariableop_44_adam_batch_normalization_50_beta_m:J
,assignvariableop_45_adam_conv3d_109_kernel_m:8
*assignvariableop_46_adam_conv3d_109_bias_m:E
7assignvariableop_47_adam_batch_normalization_51_gamma_m:D
6assignvariableop_48_adam_batch_normalization_51_beta_m:J
,assignvariableop_49_adam_conv3d_110_kernel_m:8
*assignvariableop_50_adam_conv3d_110_bias_m:>
*assignvariableop_51_adam_dense_61_kernel_m:
??@6
(assignvariableop_52_adam_dense_61_bias_m:@<
*assignvariableop_53_adam_dense_62_kernel_m:@6
(assignvariableop_54_adam_dense_62_bias_m:J
,assignvariableop_55_adam_conv3d_106_kernel_v:8
*assignvariableop_56_adam_conv3d_106_bias_v:E
7assignvariableop_57_adam_batch_normalization_49_gamma_v:D
6assignvariableop_58_adam_batch_normalization_49_beta_v:J
,assignvariableop_59_adam_conv3d_107_kernel_v:8
*assignvariableop_60_adam_conv3d_107_bias_v:J
,assignvariableop_61_adam_conv3d_108_kernel_v:8
*assignvariableop_62_adam_conv3d_108_bias_v:E
7assignvariableop_63_adam_batch_normalization_50_gamma_v:D
6assignvariableop_64_adam_batch_normalization_50_beta_v:J
,assignvariableop_65_adam_conv3d_109_kernel_v:8
*assignvariableop_66_adam_conv3d_109_bias_v:E
7assignvariableop_67_adam_batch_normalization_51_gamma_v:D
6assignvariableop_68_adam_batch_normalization_51_beta_v:J
,assignvariableop_69_adam_conv3d_110_kernel_v:8
*assignvariableop_70_adam_conv3d_110_bias_v:>
*assignvariableop_71_adam_dense_61_kernel_v:
??@6
(assignvariableop_72_adam_dense_61_bias_v:@<
*assignvariableop_73_adam_dense_62_kernel_v:@6
(assignvariableop_74_adam_dense_62_bias_v:
identity_76??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_8?AssignVariableOp_9?*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?)
value?)B?)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_conv3d_106_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv3d_106_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_49_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_49_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_49_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_49_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv3d_107_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv3d_107_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_conv3d_108_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv3d_108_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_50_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_50_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_50_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_50_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv3d_109_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv3d_109_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_51_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_51_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_51_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_51_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_conv3d_110_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv3d_110_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_61_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_61_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_62_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_62_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_iterIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_beta_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_learning_rateIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_conv3d_106_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv3d_106_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_49_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_49_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_conv3d_107_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv3d_107_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_conv3d_108_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv3d_108_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_50_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_50_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_conv3d_109_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_conv3d_109_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_batch_normalization_51_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_batch_normalization_51_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_conv3d_110_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_conv3d_110_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_61_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_61_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_62_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_62_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_conv3d_106_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv3d_106_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_49_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_49_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_conv3d_107_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_conv3d_107_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv3d_108_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv3d_108_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_50_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_50_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv3d_109_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv3d_109_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_51_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_51_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp,assignvariableop_69_adam_conv3d_110_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_conv3d_110_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_61_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_61_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_62_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_62_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75?
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_47909

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_22_layer_call_fn_50339

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_484152
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????222:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
-__inference_sequential_22_layer_call_fn_49073
conv3d_106_input%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:(

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
??@

unknown_22:@

unknown_23:@

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_106_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_22_layer_call_and_return_conditional_losses_489612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
3
_output_shapes!
:?????????222
*
_user_specified_nameconv3d_106_input
?

?
C__inference_dense_62_layer_call_and_return_conditional_losses_48445

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
?P
?
H__inference_sequential_22_layer_call_and_return_conditional_losses_48961

inputs.
conv3d_106_48894:
conv3d_106_48896:*
batch_normalization_49_48899:*
batch_normalization_49_48901:*
batch_normalization_49_48903:*
batch_normalization_49_48905:.
conv3d_107_48908:
conv3d_107_48910:.
conv3d_108_48914:
conv3d_108_48916:*
batch_normalization_50_48919:*
batch_normalization_50_48921:*
batch_normalization_50_48923:*
batch_normalization_50_48925:.
conv3d_109_48929:
conv3d_109_48931:*
batch_normalization_51_48934:*
batch_normalization_51_48936:*
batch_normalization_51_48938:*
batch_normalization_51_48940:.
conv3d_110_48944:
conv3d_110_48946:"
dense_61_48950:
??@
dense_61_48952:@ 
dense_62_48955:@
dense_62_48957:
identity??.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_50/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?"conv3d_106/StatefulPartitionedCall?"conv3d_107/StatefulPartitionedCall?"conv3d_108/StatefulPartitionedCall?"conv3d_109/StatefulPartitionedCall?"conv3d_110/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall?"dropout_42/StatefulPartitionedCall?"dropout_43/StatefulPartitionedCall?"dropout_44/StatefulPartitionedCall?
"conv3d_106/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_106_48894conv3d_106_48896*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_106_layer_call_and_return_conditional_losses_482272$
"conv3d_106/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall+conv3d_106/StatefulPartitionedCall:output:0batch_normalization_49_48899batch_normalization_49_48901batch_normalization_49_48903batch_normalization_49_48905*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_4881120
.batch_normalization_49/StatefulPartitionedCall?
"conv3d_107/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0conv3d_107_48908conv3d_107_48910*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_107_layer_call_and_return_conditional_losses_482732$
"conv3d_107/StatefulPartitionedCall?
"dropout_42/StatefulPartitionedCallStatefulPartitionedCall+conv3d_107/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_487492$
"dropout_42/StatefulPartitionedCall?
"conv3d_108/StatefulPartitionedCallStatefulPartitionedCall+dropout_42/StatefulPartitionedCall:output:0conv3d_108_48914conv3d_108_48916*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_108_layer_call_and_return_conditional_losses_482972$
"conv3d_108/StatefulPartitionedCall?
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall+conv3d_108/StatefulPartitionedCall:output:0batch_normalization_50_48919batch_normalization_50_48921batch_normalization_50_48923batch_normalization_50_48925*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_4870820
.batch_normalization_50/StatefulPartitionedCall?
"dropout_43/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0#^dropout_42/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_43_layer_call_and_return_conditional_losses_486562$
"dropout_43/StatefulPartitionedCall?
"conv3d_109/StatefulPartitionedCallStatefulPartitionedCall+dropout_43/StatefulPartitionedCall:output:0conv3d_109_48929conv3d_109_48931*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_109_layer_call_and_return_conditional_losses_483502$
"conv3d_109/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall+conv3d_109/StatefulPartitionedCall:output:0batch_normalization_51_48934batch_normalization_51_48936batch_normalization_51_48938batch_normalization_51_48940*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_4861520
.batch_normalization_51/StatefulPartitionedCall?
"dropout_44/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0#^dropout_43/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_485632$
"dropout_44/StatefulPartitionedCall?
"conv3d_110/StatefulPartitionedCallStatefulPartitionedCall+dropout_44/StatefulPartitionedCall:output:0conv3d_110_48944conv3d_110_48946*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_110_layer_call_and_return_conditional_losses_484032$
"conv3d_110/StatefulPartitionedCall?
flatten_22/PartitionedCallPartitionedCall+conv3d_110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_22_layer_call_and_return_conditional_losses_484152
flatten_22/PartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_61_48950dense_61_48952*
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
C__inference_dense_61_layer_call_and_return_conditional_losses_484282"
 dense_61/StatefulPartitionedCall?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_48955dense_62_48957*
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
C__inference_dense_62_layer_call_and_return_conditional_losses_484452"
 dense_62/StatefulPartitionedCall?
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall#^conv3d_106/StatefulPartitionedCall#^conv3d_107/StatefulPartitionedCall#^conv3d_108/StatefulPartitionedCall#^conv3d_109/StatefulPartitionedCall#^conv3d_110/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall#^dropout_42/StatefulPartitionedCall#^dropout_43/StatefulPartitionedCall#^dropout_44/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2H
"conv3d_106/StatefulPartitionedCall"conv3d_106/StatefulPartitionedCall2H
"conv3d_107/StatefulPartitionedCall"conv3d_107/StatefulPartitionedCall2H
"conv3d_108/StatefulPartitionedCall"conv3d_108/StatefulPartitionedCall2H
"conv3d_109/StatefulPartitionedCall"conv3d_109/StatefulPartitionedCall2H
"conv3d_110/StatefulPartitionedCall"conv3d_110/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2H
"dropout_42/StatefulPartitionedCall"dropout_42/StatefulPartitionedCall2H
"dropout_43/StatefulPartitionedCall"dropout_43/StatefulPartitionedCall2H
"dropout_44/StatefulPartitionedCall"dropout_44/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
*__inference_conv3d_110_layer_call_fn_50328

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_110_layer_call_and_return_conditional_losses_484032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

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
?
?
-__inference_sequential_22_layer_call_fn_49610

inputs%
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:(

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
??@

unknown_22:@

unknown_23:@

unknown_24:
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_22_layer_call_and_return_conditional_losses_484522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????222: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?,
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49741

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_50_layer_call_fn_50048

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_479692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_50022

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????2222
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?
?
*__inference_conv3d_109_layer_call_fn_50121

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????222*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3d_109_layer_call_and_return_conditional_losses_483502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????2222

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
?
?
E__inference_conv3d_106_layer_call_and_return_conditional_losses_48227

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????222*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????2222	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????2222
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????222: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs
?+
?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_48708

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????2222
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????2222
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????2222

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????222: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????222
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Y
conv3d_106_inputE
"serving_default_conv3d_106_input:0?????????222<
dense_620
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
ف
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?|
_tf_keras_sequential?|{"name": "sequential_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_106_input"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_106", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 1]}, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv3D", "config": {"name": "conv3d_107", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv3D", "config": {"name": "conv3d_108", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_43", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv3D", "config": {"name": "conv3d_109", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_51", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_44", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv3D", "config": {"name": "conv3d_110", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_62", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 50, 50, 50, 1]}, "float32", "conv3d_106_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_106_input"}, "shared_object_id": 0}, {"class_name": "Conv3D", "config": {"name": "conv3d_106", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 1]}, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv3D", "config": {"name": "conv3d_107", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 12}, {"class_name": "Conv3D", "config": {"name": "conv3d_108", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 20}, {"class_name": "Dropout", "config": {"name": "dropout_43", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 21}, {"class_name": "Conv3D", "config": {"name": "conv3d_109", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_51", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 29}, {"class_name": "Dropout", "config": {"name": "dropout_44", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 30}, {"class_name": "Conv3D", "config": {"name": "conv3d_110", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 33}, {"class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 34}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37}, {"class_name": "Dense", "config": {"name": "dense_62", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 43}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv3d_106", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_106", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 50, 1]}, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 1]}}
?

axis
	gamma
beta
moving_mean
moving_variance
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_49", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 4}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?


$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3d_107", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_107", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_42", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 12}
?


.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3d_108", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_108", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?

4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:regularization_losses
;trainable_variables
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_50", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 4}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_43", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_43", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 21}
?


Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3d_109", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_109", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?

Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_51", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_51", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 4}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_44", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 30}
?


Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3d_110", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_110", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 50, 4]}}
?
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 51}}
?

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500000}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500000]}}
?

dkernel
ebias
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_62", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_62", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
jiter

kbeta_1

lbeta_2
	mdecay
nlearning_ratem?m?m?m?$m?%m?.m?/m?5m?6m?Am?Bm?Hm?Im?Tm?Um?^m?_m?dm?em?v?v?v?v?$v?%v?.v?/v?5v?6v?Av?Bv?Hv?Iv?Tv?Uv?^v?_v?dv?ev?"
	optimizer
?
0
1
2
3
4
5
$6
%7
.8
/9
510
611
712
813
A14
B15
H16
I17
J18
K19
T20
U21
^22
_23
d24
e25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
$4
%5
.6
/7
58
69
A10
B11
H12
I13
T14
U15
^16
_17
d18
e19"
trackable_list_wrapper
?
olayer_regularization_losses
player_metrics
qnon_trainable_variables
	variables
regularization_losses

rlayers
trainable_variables
smetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
/:-2conv3d_106/kernel
:2conv3d_106/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
tlayer_regularization_losses
ulayer_metrics
vnon_trainable_variables
	variables
regularization_losses

wlayers
trainable_variables
xmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_49/gamma
):'2batch_normalization_49/beta
2:0 (2"batch_normalization_49/moving_mean
6:4 (2&batch_normalization_49/moving_variance
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
ylayer_regularization_losses
zlayer_metrics
{non_trainable_variables
 	variables
!regularization_losses

|layers
"trainable_variables
}metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv3d_107/kernel
:2conv3d_107/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
~layer_regularization_losses
layer_metrics
?non_trainable_variables
&	variables
'regularization_losses
?layers
(trainable_variables
?metrics
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
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
*	variables
+regularization_losses
?layers
,trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv3d_108/kernel
:2conv3d_108/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
0	variables
1regularization_losses
?layers
2trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_50/gamma
):'2batch_normalization_50/beta
2:0 (2"batch_normalization_50/moving_mean
6:4 (2&batch_normalization_50/moving_variance
<
50
61
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
9	variables
:regularization_losses
?layers
;trainable_variables
?metrics
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
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
=	variables
>regularization_losses
?layers
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv3d_109/kernel
:2conv3d_109/bias
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
C	variables
Dregularization_losses
?layers
Etrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_51/gamma
):'2batch_normalization_51/beta
2:0 (2"batch_normalization_51/moving_mean
6:4 (2&batch_normalization_51/moving_variance
<
H0
I1
J2
K3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
L	variables
Mregularization_losses
?layers
Ntrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
P	variables
Qregularization_losses
?layers
Rtrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv3d_110/kernel
:2conv3d_110/bias
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
V	variables
Wregularization_losses
?layers
Xtrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Z	variables
[regularization_losses
?layers
\trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??@2dense_61/kernel
:@2dense_61/bias
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
`	variables
aregularization_losses
?layers
btrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_62/kernel
:2dense_62/bias
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
f	variables
gregularization_losses
?layers
htrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
72
83
J4
K5"
trackable_list_wrapper
?
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
12
13"
trackable_list_wrapper
0
?0
?1"
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
.
0
1"
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
.
70
81"
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
.
J0
K1"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 54}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 43}
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
4:22Adam/conv3d_106/kernel/m
": 2Adam/conv3d_106/bias/m
/:-2#Adam/batch_normalization_49/gamma/m
.:,2"Adam/batch_normalization_49/beta/m
4:22Adam/conv3d_107/kernel/m
": 2Adam/conv3d_107/bias/m
4:22Adam/conv3d_108/kernel/m
": 2Adam/conv3d_108/bias/m
/:-2#Adam/batch_normalization_50/gamma/m
.:,2"Adam/batch_normalization_50/beta/m
4:22Adam/conv3d_109/kernel/m
": 2Adam/conv3d_109/bias/m
/:-2#Adam/batch_normalization_51/gamma/m
.:,2"Adam/batch_normalization_51/beta/m
4:22Adam/conv3d_110/kernel/m
": 2Adam/conv3d_110/bias/m
(:&
??@2Adam/dense_61/kernel/m
 :@2Adam/dense_61/bias/m
&:$@2Adam/dense_62/kernel/m
 :2Adam/dense_62/bias/m
4:22Adam/conv3d_106/kernel/v
": 2Adam/conv3d_106/bias/v
/:-2#Adam/batch_normalization_49/gamma/v
.:,2"Adam/batch_normalization_49/beta/v
4:22Adam/conv3d_107/kernel/v
": 2Adam/conv3d_107/bias/v
4:22Adam/conv3d_108/kernel/v
": 2Adam/conv3d_108/bias/v
/:-2#Adam/batch_normalization_50/gamma/v
.:,2"Adam/batch_normalization_50/beta/v
4:22Adam/conv3d_109/kernel/v
": 2Adam/conv3d_109/bias/v
/:-2#Adam/batch_normalization_51/gamma/v
.:,2"Adam/batch_normalization_51/beta/v
4:22Adam/conv3d_110/kernel/v
": 2Adam/conv3d_110/bias/v
(:&
??@2Adam/dense_61/kernel/v
 :@2Adam/dense_61/bias/v
&:$@2Adam/dense_62/kernel/v
 :2Adam/dense_62/bias/v
?2?
H__inference_sequential_22_layer_call_and_return_conditional_losses_49384
H__inference_sequential_22_layer_call_and_return_conditional_losses_49553
H__inference_sequential_22_layer_call_and_return_conditional_losses_49143
H__inference_sequential_22_layer_call_and_return_conditional_losses_49213?
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
-__inference_sequential_22_layer_call_fn_48507
-__inference_sequential_22_layer_call_fn_49610
-__inference_sequential_22_layer_call_fn_49667
-__inference_sequential_22_layer_call_fn_49073?
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
 __inference__wrapped_model_47723?
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
annotations? *;?8
6?3
conv3d_106_input?????????222
?2?
E__inference_conv3d_106_layer_call_and_return_conditional_losses_49678?
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
*__inference_conv3d_106_layer_call_fn_49687?
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
?2?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49707
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49741
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49761
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49795?
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
6__inference_batch_normalization_49_layer_call_fn_49808
6__inference_batch_normalization_49_layer_call_fn_49821
6__inference_batch_normalization_49_layer_call_fn_49834
6__inference_batch_normalization_49_layer_call_fn_49847?
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
E__inference_conv3d_107_layer_call_and_return_conditional_losses_49858?
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
*__inference_conv3d_107_layer_call_fn_49867?
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
E__inference_dropout_42_layer_call_and_return_conditional_losses_49872
E__inference_dropout_42_layer_call_and_return_conditional_losses_49884?
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
?2?
*__inference_dropout_42_layer_call_fn_49889
*__inference_dropout_42_layer_call_fn_49894?
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
E__inference_conv3d_108_layer_call_and_return_conditional_losses_49905?
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
*__inference_conv3d_108_layer_call_fn_49914?
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
?2?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49934
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49968
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49988
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_50022?
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
6__inference_batch_normalization_50_layer_call_fn_50035
6__inference_batch_normalization_50_layer_call_fn_50048
6__inference_batch_normalization_50_layer_call_fn_50061
6__inference_batch_normalization_50_layer_call_fn_50074?
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
?2?
E__inference_dropout_43_layer_call_and_return_conditional_losses_50079
E__inference_dropout_43_layer_call_and_return_conditional_losses_50091?
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
?2?
*__inference_dropout_43_layer_call_fn_50096
*__inference_dropout_43_layer_call_fn_50101?
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
E__inference_conv3d_109_layer_call_and_return_conditional_losses_50112?
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
*__inference_conv3d_109_layer_call_fn_50121?
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
?2?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50141
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50175
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50195
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50229?
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
6__inference_batch_normalization_51_layer_call_fn_50242
6__inference_batch_normalization_51_layer_call_fn_50255
6__inference_batch_normalization_51_layer_call_fn_50268
6__inference_batch_normalization_51_layer_call_fn_50281?
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
?2?
E__inference_dropout_44_layer_call_and_return_conditional_losses_50286
E__inference_dropout_44_layer_call_and_return_conditional_losses_50298?
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
?2?
*__inference_dropout_44_layer_call_fn_50303
*__inference_dropout_44_layer_call_fn_50308?
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
E__inference_conv3d_110_layer_call_and_return_conditional_losses_50319?
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
*__inference_conv3d_110_layer_call_fn_50328?
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
E__inference_flatten_22_layer_call_and_return_conditional_losses_50334?
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
*__inference_flatten_22_layer_call_fn_50339?
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
C__inference_dense_61_layer_call_and_return_conditional_losses_50350?
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
(__inference_dense_61_layer_call_fn_50359?
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
C__inference_dense_62_layer_call_and_return_conditional_losses_50370?
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
(__inference_dense_62_layer_call_fn_50379?
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
#__inference_signature_wrapper_49278conv3d_106_input"?
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
 __inference__wrapped_model_47723?$%./8576ABKHJITU^_deE?B
;?8
6?3
conv3d_106_input?????????222
? "3?0
.
dense_62"?
dense_62??????????
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49707?Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49741?Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49761z??<
5?2
,?)
inputs?????????222
p 
? "1?.
'?$
0?????????222
? ?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_49795z??<
5?2
,?)
inputs?????????222
p
? "1?.
'?$
0?????????222
? ?
6__inference_batch_normalization_49_layer_call_fn_49808?Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "??<8?????????????????????????????????????
6__inference_batch_normalization_49_layer_call_fn_49821?Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "??<8?????????????????????????????????????
6__inference_batch_normalization_49_layer_call_fn_49834m??<
5?2
,?)
inputs?????????222
p 
? "$?!?????????222?
6__inference_batch_normalization_49_layer_call_fn_49847m??<
5?2
,?)
inputs?????????222
p
? "$?!?????????222?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49934?8576Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49968?7856Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_49988z8576??<
5?2
,?)
inputs?????????222
p 
? "1?.
'?$
0?????????222
? ?
Q__inference_batch_normalization_50_layer_call_and_return_conditional_losses_50022z7856??<
5?2
,?)
inputs?????????222
p
? "1?.
'?$
0?????????222
? ?
6__inference_batch_normalization_50_layer_call_fn_50035?8576Z?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "??<8?????????????????????????????????????
6__inference_batch_normalization_50_layer_call_fn_50048?7856Z?W
P?M
G?D
inputs8????????????????????????????????????
p
? "??<8?????????????????????????????????????
6__inference_batch_normalization_50_layer_call_fn_50061m8576??<
5?2
,?)
inputs?????????222
p 
? "$?!?????????222?
6__inference_batch_normalization_50_layer_call_fn_50074m7856??<
5?2
,?)
inputs?????????222
p
? "$?!?????????222?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50141?KHJIZ?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50175?JKHIZ?W
P?M
G?D
inputs8????????????????????????????????????
p
? "L?I
B??
08????????????????????????????????????
? ?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50195zKHJI??<
5?2
,?)
inputs?????????222
p 
? "1?.
'?$
0?????????222
? ?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_50229zJKHI??<
5?2
,?)
inputs?????????222
p
? "1?.
'?$
0?????????222
? ?
6__inference_batch_normalization_51_layer_call_fn_50242?KHJIZ?W
P?M
G?D
inputs8????????????????????????????????????
p 
? "??<8?????????????????????????????????????
6__inference_batch_normalization_51_layer_call_fn_50255?JKHIZ?W
P?M
G?D
inputs8????????????????????????????????????
p
? "??<8?????????????????????????????????????
6__inference_batch_normalization_51_layer_call_fn_50268mKHJI??<
5?2
,?)
inputs?????????222
p 
? "$?!?????????222?
6__inference_batch_normalization_51_layer_call_fn_50281mJKHI??<
5?2
,?)
inputs?????????222
p
? "$?!?????????222?
E__inference_conv3d_106_layer_call_and_return_conditional_losses_49678t;?8
1?.
,?)
inputs?????????222
? "1?.
'?$
0?????????222
? ?
*__inference_conv3d_106_layer_call_fn_49687g;?8
1?.
,?)
inputs?????????222
? "$?!?????????222?
E__inference_conv3d_107_layer_call_and_return_conditional_losses_49858t$%;?8
1?.
,?)
inputs?????????222
? "1?.
'?$
0?????????222
? ?
*__inference_conv3d_107_layer_call_fn_49867g$%;?8
1?.
,?)
inputs?????????222
? "$?!?????????222?
E__inference_conv3d_108_layer_call_and_return_conditional_losses_49905t./;?8
1?.
,?)
inputs?????????222
? "1?.
'?$
0?????????222
? ?
*__inference_conv3d_108_layer_call_fn_49914g./;?8
1?.
,?)
inputs?????????222
? "$?!?????????222?
E__inference_conv3d_109_layer_call_and_return_conditional_losses_50112tAB;?8
1?.
,?)
inputs?????????222
? "1?.
'?$
0?????????222
? ?
*__inference_conv3d_109_layer_call_fn_50121gAB;?8
1?.
,?)
inputs?????????222
? "$?!?????????222?
E__inference_conv3d_110_layer_call_and_return_conditional_losses_50319tTU;?8
1?.
,?)
inputs?????????222
? "1?.
'?$
0?????????222
? ?
*__inference_conv3d_110_layer_call_fn_50328gTU;?8
1?.
,?)
inputs?????????222
? "$?!?????????222?
C__inference_dense_61_layer_call_and_return_conditional_losses_50350^^_1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????@
? }
(__inference_dense_61_layer_call_fn_50359Q^_1?.
'?$
"?
inputs???????????
? "??????????@?
C__inference_dense_62_layer_call_and_return_conditional_losses_50370\de/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_dense_62_layer_call_fn_50379Ode/?,
%?"
 ?
inputs?????????@
? "???????????
E__inference_dropout_42_layer_call_and_return_conditional_losses_49872t??<
5?2
,?)
inputs?????????222
p 
? "1?.
'?$
0?????????222
? ?
E__inference_dropout_42_layer_call_and_return_conditional_losses_49884t??<
5?2
,?)
inputs?????????222
p
? "1?.
'?$
0?????????222
? ?
*__inference_dropout_42_layer_call_fn_49889g??<
5?2
,?)
inputs?????????222
p 
? "$?!?????????222?
*__inference_dropout_42_layer_call_fn_49894g??<
5?2
,?)
inputs?????????222
p
? "$?!?????????222?
E__inference_dropout_43_layer_call_and_return_conditional_losses_50079t??<
5?2
,?)
inputs?????????222
p 
? "1?.
'?$
0?????????222
? ?
E__inference_dropout_43_layer_call_and_return_conditional_losses_50091t??<
5?2
,?)
inputs?????????222
p
? "1?.
'?$
0?????????222
? ?
*__inference_dropout_43_layer_call_fn_50096g??<
5?2
,?)
inputs?????????222
p 
? "$?!?????????222?
*__inference_dropout_43_layer_call_fn_50101g??<
5?2
,?)
inputs?????????222
p
? "$?!?????????222?
E__inference_dropout_44_layer_call_and_return_conditional_losses_50286t??<
5?2
,?)
inputs?????????222
p 
? "1?.
'?$
0?????????222
? ?
E__inference_dropout_44_layer_call_and_return_conditional_losses_50298t??<
5?2
,?)
inputs?????????222
p
? "1?.
'?$
0?????????222
? ?
*__inference_dropout_44_layer_call_fn_50303g??<
5?2
,?)
inputs?????????222
p 
? "$?!?????????222?
*__inference_dropout_44_layer_call_fn_50308g??<
5?2
,?)
inputs?????????222
p
? "$?!?????????222?
E__inference_flatten_22_layer_call_and_return_conditional_losses_50334f;?8
1?.
,?)
inputs?????????222
? "'?$
?
0???????????
? ?
*__inference_flatten_22_layer_call_fn_50339Y;?8
1?.
,?)
inputs?????????222
? "?????????????
H__inference_sequential_22_layer_call_and_return_conditional_losses_49143?$%./8576ABKHJITU^_deM?J
C?@
6?3
conv3d_106_input?????????222
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_22_layer_call_and_return_conditional_losses_49213?$%./7856ABJKHITU^_deM?J
C?@
6?3
conv3d_106_input?????????222
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_22_layer_call_and_return_conditional_losses_49384?$%./8576ABKHJITU^_deC?@
9?6
,?)
inputs?????????222
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_22_layer_call_and_return_conditional_losses_49553?$%./7856ABJKHITU^_deC?@
9?6
,?)
inputs?????????222
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_22_layer_call_fn_48507?$%./8576ABKHJITU^_deM?J
C?@
6?3
conv3d_106_input?????????222
p 

 
? "???????????
-__inference_sequential_22_layer_call_fn_49073?$%./7856ABJKHITU^_deM?J
C?@
6?3
conv3d_106_input?????????222
p

 
? "???????????
-__inference_sequential_22_layer_call_fn_49610{$%./8576ABKHJITU^_deC?@
9?6
,?)
inputs?????????222
p 

 
? "???????????
-__inference_sequential_22_layer_call_fn_49667{$%./7856ABJKHITU^_deC?@
9?6
,?)
inputs?????????222
p

 
? "???????????
#__inference_signature_wrapper_49278?$%./8576ABKHJITU^_deY?V
? 
O?L
J
conv3d_106_input6?3
conv3d_106_input?????????222"3?0
.
dense_62"?
dense_62?????????