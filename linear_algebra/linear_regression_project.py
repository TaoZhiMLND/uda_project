
# coding: utf-8

# # 欢迎来到线性回归项目
# 
# 若项目中的题目有困难没完成也没关系，我们鼓励你带着问题提交项目，评审人会给予你诸多帮助。
# 
# 所有选做题都可以不做，不影响项目通过。如果你做了，那么项目评审会帮你批改，也会因为选做部分做错而判定为不通过。
# 
# 其中非代码题可以提交手写后扫描的 pdf 文件，或使用 Latex 在文档中直接回答。

# ### 目录:
# [1 矩阵运算](#1-矩阵运算)  
# [2 Gaussian Jordan 消元法](#2-Gaussian-Jordan-消元法)  
# [3  线性回归](#3-线性回归)  

# In[1]:


# 任意选一个你喜欢的整数，这能帮你得到稳定的结果
seed = 8


# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[2]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何NumPy以及相关的科学计算库来完成作业


# 本项目要求矩阵统一使用二维列表表示，如下：
A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

# 向量也用二维列表表示
C = [[1],
     [2],
     [3]]

#TODO 创建一个 4*4 单位矩阵
I = [[1,0,0,0], 
     [0,1,0,0], 
     [0,0,1,0],
     [0,0,0,1]]

len(tuple(zip(*I)))


# ## 1.2 返回矩阵的行数和列数

# In[3]:


# DONE 返回矩阵的行数和列数
def shape(M):
    return (len(M),len(M[0]))


# In[4]:


# 运行以下代码测试你的 shape 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 每个元素四舍五入到特定小数数位

# In[5]:


# DONE 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(0, shape(M)[0]):
        for j in range(0, shape(M)[1]):
            M[i][j] = round(M[i][j], decPts)
    pass


# In[6]:


# 运行以下代码测试你的 matxRound 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 计算矩阵的转置

# In[7]:


# DONE 计算矩阵的转置
def transpose(M):
    return [list(col) for col in zip(*M)]


# In[8]:


# 运行以下代码测试你的 transpose 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 计算矩阵乘法 AB

# In[9]:


# DONE 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):
    if (shape(A)[1] != shape(B)[0]):
        raise ValueError
    result = list()
    B_t = transpose(B)
    for j in range(0, shape(A)[0]):
        row = list()
        for i in range(0, shape(B_t)[0]):
            node = 0
            for c in range(0, shape(A)[1]):
                node += B_t[i][c] * A[j][c]
            row.append(node)
        result.append(row)
    return result


# In[10]:


# 运行以下代码测试你的 matxMultiply 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
# # 2 Gaussian Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[11]:


# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    return [ra + rb for ra, rb in zip(A, b)]


# In[12]:


# 运行以下代码测试你的 augmentMatrix 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[13]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]


# In[14]:


# 运行以下代码测试你的 swapRows 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_swapRows')


# In[32]:


# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if (scale == 0):
        raise ValueError
    M[r] = [row * scale for row in M[r]]


# In[33]:


# 运行以下代码测试你的 scaleRow 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[17]:


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    M[r1] = [row1 + row2 * scale for row1, row2 in zip(M[r1], M[r2])]


# In[18]:


# 运行以下代码测试你的 addScaledRow 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_addScaledRow')


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 2.3.1 算法
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None (你可以在选做问题2.4中证明为什么这里A一定是奇异矩阵)
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# **注：** 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# ### 2.3.2 算法推演
# 
# 为了充分了解Gaussian Jordan消元法的计算流程，请根据Gaussian Jordan消元法，分别手动推演矩阵A为***可逆矩阵***，矩阵A为***奇异矩阵***两种情况。

# #### 推演示例 
# 
# 
# $Ab = \begin{bmatrix}
#     -7 & 5 & -1 & 1\\
#     1 & -3 & -8 & 1\\
#     -10 & -2 & 9 & 1\end{bmatrix}$
# 
# $ --> $
# $\begin{bmatrix}
#     1 & \frac{1}{5} & -\frac{9}{10} & -\frac{1}{10}\\
#     0 & -\frac{16}{5} & -\frac{71}{10} & \frac{11}{10}\\
#     0 & \frac{32}{5} & -\frac{73}{10} & \frac{3}{10}\end{bmatrix}$
# 
# $ --> $
# $\begin{bmatrix}
#     1 & 0 & -\frac{43}{64} & -\frac{7}{64}\\
#     0 & 1 & -\frac{73}{64} & \frac{3}{64}\\
#     0 & 0 & -\frac{43}{4} & \frac{5}{4}\end{bmatrix}$
# 
# $ --> $
# $\begin{bmatrix}
#     1 & 0 & 0 & -\frac{3}{16}\\
#     0 & 1 & 0 & -\frac{59}{688}\\
#     0 & 0 & 1 & -\frac{5}{43}\end{bmatrix}$
#     
# 
# #### 推演有以下要求:
# 1. 展示每一列的消元结果, 比如3*3的矩阵, 需要写三步
# 2. 用分数来表示
# 3. 分数不能再约分
# 4. 我们已经给出了latex的语法,你只要把零改成你要的数字(或分数)即可
# 5. 可以用[这个页面](http://www.math.odu.edu/~bogacki/cgi-bin/lat.cgi?c=sys)检查你的答案(注意只是答案, 推演步骤两者算法不一致)
# 
# _你可以用python的 [fractions](https://docs.python.org/2/library/fractions.html) 模块辅助你的约分_

# #### 分数的输入方法
# (双击这个区域就能看到语法啦)  
#   
# 示例一: $\frac{n}{m}$  
# 
# 示例二: $-\frac{a}{b}$  

# #### 以下开始你的尝试吧!

# In[19]:


# 不要修改这里！
from helper import *
A = generateMatrix(3,seed,singular=False)
b = np.ones(shape=(3,1),dtype=int) # it doesn't matter
Ab = augmentMatrix(A.tolist(),b.tolist()) # 请确保你的增广矩阵已经写好了
printInMatrixFormat(Ab,padding=3,truncating=0)


# 请按照算法的步骤3，逐步推演***可逆矩阵***的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵(注意使用[分数语法](#分数的输入方法))
# 
# $ Ab = \begin{bmatrix}
#     1 & -1 & \frac{1}{7} & -\frac{1}{7} \\
#     0 & -7 & \frac{68}{7} & \frac{2}{7} \\
#     0 & 4 & \frac{23}{7} & \frac{5}{7} \end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     1 & 0 & -\frac{61}{49} & -\frac{9}{49} \\
#     0 & 1 & -\frac{68}{49} & -\frac{2}{49} \\
#     0 & 0 & \frac{433}{49} & \frac{43}{49} \end{bmatrix}$
#     
# $ --> \begin{bmatrix}
#     1 & 0 & 0 & -\frac{26}{433} \\
#     0 & 1 & 0 & \frac{42}{433} \\
#     0 & 0 & 1 & \frac{43}{433} \end{bmatrix}$

# In[20]:


# 不要修改这里！
A = generateMatrix(3,seed,singular=True)
b = np.ones(shape=(3,1),dtype=int)
Ab = augmentMatrix(A.tolist(),b.tolist()) # 请确保你的增广矩阵已经写好了
printInMatrixFormat(Ab,padding=3,truncating=0)


# 请按照算法的步骤3，逐步推演***奇异矩阵***的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵(注意使用[分数语法](#分数的输入方法))
# 
# $ Ab = \begin{bmatrix}
#     1 & -\frac{3}{4} & -\frac{3}{8} & -\frac{1}{8} \\
#     0 & 0 & 0 & 0 \\
#     0 & -\frac{15}{4} & -\frac{63}{8} & -\frac{3}{8} \end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     1 & 0 & \frac{6}{5} & -\frac{1}{5} \\
#     0 & 1 & \frac{21}{10} & -\frac{1}{10} \\
#     0 & 0 & 0 & 0 \end{bmatrix}$

# ### 2.3.3 实现 Gaussian Jordan 消元法

# In[21]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""
def find_max_value_row_under_diagonal(A, col):
    row_index = col
    max_value = abs(A[col][col])
    for i in range(col+1, len(A)):
        if (abs(A[i][col]) > max_value):
            max_value = abs(A[i][col])
            row_index = i
    return row_index
    

def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if (len(A) != len(b)):
        return None
    Ab = augmentMatrix(A, b)
    for i in range(0, len(Ab)):
        #寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值所在行
        row_index = find_max_value_row_under_diagonal(Ab, i)
        #如果绝对值最大值为0 那么A为奇异矩阵，返回None
        if (abs(Ab[row_index][i]) <= epsilon):
            return None
        # 使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
        swapRows(Ab, i, row_index)
        #使用第二个行变换，将列c的对角线元素缩放为1
        scaleRow(Ab, i, 1/Ab[i][i])
        #多次使用第三个行变换，将列c的其他元素消为0
        for j in range(0, len(Ab)):
            if (j == i):
                continue
            addScaledRow(Ab, j, i, -Ab[j][i])
    matxRound(Ab, decPts)
    result = list()
    for i in range(0, len(Ab)):
        res = list([Ab[i][len(Ab[i])-1]])
        result.append(res)
    
    return result


# In[22]:


# 运行以下代码测试你的 gj_Solve 函数
get_ipython().run_line_magic('run', '-i -e test.py LinearRegressionTestCase.test_gj_Solve')


# ## (选做) 2.4 算法正确判断了奇异矩阵：
# 
# 在算法的步骤3 中，如果发现某一列对角线和对角线以下所有元素都为0，那么则断定这个矩阵为奇异矩阵。
# 
# 我们用正式的语言描述这个命题，并证明为真。
# 
# 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# DONE 证明：
# 
# $
# \\
# 设矩阵Y的第一列为A的第t列，因为I 为单位矩阵，Z 为全0矩阵，I为n维方阵;\\
# 设A的第t列元素为t_1,t_2…t_n,0,0…\\
# 则A_t = A_1*t_1+A_2*t_2+…+A_n*t_n\\
# 矩阵A的第t列,是前n列的线性组合，所以可以断定A为奇异矩阵\\
# $

# # 3 线性回归

# ## 3.1 随机生成样本点

# In[23]:


# 不要修改这里！
get_ipython().run_line_magic('matplotlib', 'notebook')
from helper import *

X,Y = generatePoints2D(seed)
vs_scatter_2d(X, Y)


# ## 3.2 拟合一条直线
# 
# ### 3.2.1 猜测一条直线

# In[24]:


#TODO 请选择最适合的直线 y = mx + b
m1 = -5
b1 = 8
# 不要修改这里！
vs_scatter_2d(X, Y, m1, b1)


# ### 3.2.2 计算平均平方误差 (MSE)

# 我们要编程计算所选直线的平均平方误差(MSE), 即数据集中每个点到直线的Y方向距离的平方的平均数，表达式如下：
# $$
# MSE = \frac{1}{n}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$

# In[25]:


# DONE 实现以下函数并输出所选直线的MSE
def calculateMSE2D(X,Y,m,b):
    n = 0;
    mse_sum = 0
    for (x ,y) in zip(X, Y):
        n += 1
        mse_sum += pow(y - m*x - b, 2)
    return mse_sum / n

# DONE 检查这里的结果, 如果你上面猜测的直线准确, 这里的输出会在1.5以内
print(calculateMSE2D(X,Y,m1,b1))


# ### 3.2.3 调整参数 $m, b$ 来获得最小的平方平均误差
# 
# 你可以调整3.2.1中的参数 $m1,b1$ 让蓝点均匀覆盖在红线周围，然后微调 $m1, b1$ 让MSE最小。

# ## 3.3 (选做) 找到参数 $m, b$ 使得平方平均误差最小
# 
# **这一部分需要简单的微积分知识(  $ (x^2)' = 2x $ )。因为这是一个线性代数项目，所以设为选做。**
# 
# 刚刚我们手动调节参数，尝试找到最小的平方平均误差。下面我们要精确得求解 $m, b$ 使得平方平均误差最小。
# 
# 定义目标函数 $E$ 为
# $$
# E = \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 因为 $E = \frac{n}{2}MSE$, 所以 $E$ 取到最小值时，$MSE$ 也取到最小值。要找到 $E$ 的最小值，即要找到 $m, b$ 使得 $E$ 相对于 $m$, $E$ 相对于 $b$ 的偏导数等于0. 
# 
# 因此我们要解下面的方程组。
# 
# $$
# \begin{cases}
# \displaystyle
# \frac{\partial E}{\partial m} =0 \\
# \\
# \displaystyle
# \frac{\partial E}{\partial b} =0 \\
# \end{cases}
# $$
# 
# ### 3.3.1 计算目标函数相对于参数的导数
# 首先我们计算两个式子左边的值
# 
# 证明/计算：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-(y_i - mx_i - b)}
# $$

# DONE 证明:
# 
# $
# \\
# E = \frac{1}{2}((y_1-mx_1-b)^2 + (y_2-mx_2-b)^2 + … + (y_n-mx_n-b)^2)\\
# \frac{\partial E}{\partial m} = \frac{1}{2}(-2x_1(y_1-mx_1-b) - 2x_2(y_2-mx_2-b) + … -2x_n(y_n-mx_n-b))\\
# \frac{\partial E}{\partial m} = -x_1(y_1-mx_1-b) - x_2(y_2-mx_2-b) + … -x_n(y_n-mx_n-b)\\
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} \\
# \frac{\partial E}{\partial b} = \frac{1}{2}(-2(y_1-mx_1-b) - 2(y_2-mx_2-b) + … -2(y_n-mx_n-b))\\
# \frac{\partial E}{\partial b} = -(y_1-mx_1-b) - (y_2-mx_2-b) + … -(y_n-mx_n-b)\\
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-(y_i - mx_i - b)}\\
# $

# ### 3.3.2 实例推演
# 
# 现在我们有了一个二元二次方程组
# 
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# $$
# 
# 为了加强理解，我们用一个实际例子演练。
# 
# 我们要用三个点 $(1,1), (2,2), (3,2)$ 来拟合一条直线 y = m*x + b, 请写出
# 
# - 目标函数 $E$, 
# - 二元二次方程组，
# - 并求解最优参数 $m, b$

# DONE 写出目标函数，方程组和最优参数
# $
# \\
# E = \frac{1}{2}((1-m-b)^2 + (2-2m-b)^2 + (2-3m-b)^2)\\
# \begin{cases}
# \displaystyle
# -(1-m-b)-2(2-2m-b)-3(2-3m-b)=0 \\
# \\
# \displaystyle
# -(1-m-b)-(2-2m-b)-(2-3m-b)=0 \\
# \end{cases}\\
# m=\frac{1}{2},b=\frac{2}{3}
# $

# ### 3.3.3 将方程组写成矩阵形式
# 
# 我们的二元二次方程组可以用更简洁的矩阵形式表达，将方程组写成矩阵形式更有利于我们使用 Gaussian Jordan 消元法求解。
# 
# 请证明 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = X^TXh - X^TY
# $$
# 
# 其中向量 $Y$, 矩阵 $X$ 和 向量 $h$ 分别为 :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# DONE 证明:
# 
# $
# \\
# X^TXh=
# \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1 & 1 & ...& 1\\
# \end{bmatrix}
# \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix}
# \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}\\
# =\begin{bmatrix}
#     x_1^2 +x_2^2 +...xn^2 & x_1 + x_2 + ... + x_n \\
#       x_1 + x_2 + ... + x_n & n\\
# \end{bmatrix}
# \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}\\
# =\begin{bmatrix}
#     m(x_1^2 +x_2^2 +...xn^2) + b(x_1 + x_2 + ... + x_n) \\
#     m(x_1 + x_2 + ... + x_n) +bn \\
# \end{bmatrix}\\
# X^TY=
# \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1 & 1 & ...& 1\\
# \end{bmatrix}
# \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}\\
# =\begin{bmatrix}
#     x_1y_1 + x_2y_2 + ... +x_ny_n \\
#     y_1 + y_2 + ... +y_n \\
# \end{bmatrix}\\
# X^TXh - X^TY = 
# \begin{bmatrix}
#     -x_1(y_1-mx_1-b) - x_2(y_2-mx_2-b) + … -x_n(y_n-mx_n-b) \\
#     -(y_1-mx_1-b) - (y_2-mx_2-b) + … -(y_n-mx_n-b) \\
# \end{bmatrix}\\
# =\begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}
# $

# 至此我们知道，通过求解方程 $X^TXh = X^TY$ 来找到最优参数。这个方程十分重要，他有一个名字叫做 **Normal Equation**，也有直观的几何意义。你可以在 [子空间投影](http://open.163.com/movie/2010/11/J/U/M6V0BQC4M_M6V2AJLJU.html) 和 [投影矩阵与最小二乘](http://open.163.com/movie/2010/11/P/U/M6V0BQC4M_M6V2AOJPU.html) 看到更多关于这个方程的内容。

# ### 3.4 求解 $X^TXh = X^TY$ 
# 
# 在3.3 中，我们知道线性回归问题等价于求解 $X^TXh = X^TY$ (如果你选择不做3.3，就勇敢的相信吧，哈哈)

# In[26]:


# DONE 实现线性回归
'''
参数：X, Y 存储着一一对应的横坐标与纵坐标的两个一维数组
返回：线性回归的系数(如上面所说的 m, b)
'''
def linearRegression2D(X,Y):
    X = [[x, 1] for x in X]
    Y = [[y] for y in Y]
    X_t = transpose(X)
    A = matxMultiply(X_t, X)
    b = matxMultiply(X_t, Y)
    result = gj_Solve(A, b)
    return transpose(result)[0]


# In[27]:


# 请不要修改下面的代码
m2,b2 = linearRegression2D(X,Y)
assert isinstance(m2,float),"m is not a float"
assert isinstance(b2,float),"b is not a float"
print(m2,b2)


# 你求得的回归结果是什么？
# 请使用运行以下代码将它画出来。

# In[28]:


## 请不要修改下面的代码
vs_scatter_2d(X, Y, m2, b2)
print(calculateMSE2D(X,Y,m2,b2))


# ## Bonus !!!
# 如果你的高斯约当消元法通过了单元测试, 那么它将能够解决多维的回归问题  
# 你将会在更高维度考验你的线性回归实现

# In[29]:


# 生成三维的数据点
X_3d, Y_3d = generatePoints3D(seed)
vs_scatter_3d(X_3d, Y_3d)


# 你的线性回归是否能够对付三维的情况?

# In[30]:


def linearRegression(X,Y):
    X = [[*x, 1] for x in X]
    Y = [[y] for y in Y]
    X_t = transpose(X)
    A = matxMultiply(X_t, X)
    b = matxMultiply(X_t, Y)
    result = gj_Solve(A, b)
    return transpose(result)[0]


# In[31]:


coeff = linearRegression(X_3d, Y_3d)
vs_scatter_3d(X_3d, Y_3d, coeff)

