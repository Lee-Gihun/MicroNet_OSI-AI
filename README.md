{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MicroNet Challenge (Team: OSI AI)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Our team build a network having `80.06%` accuracy on cifar-100 with `0.5322(Mbyte)` parameters and `30.4957(FLOPs)` multiply-add operations, achieveing the MicroNet Challenge score of `0.006552`."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Overview\n",
    "The below figure is our proposed architecture for the cifar-100 dataset. The numbers described above the arrows are the shape of each input and output.  \n",
    "Our architecture consists of:  \n",
    "1. Upsample Layer\n",
    "2. Stem_Conv\n",
    "3. 10 $\\times$ MobileNet V2 Convolution Block (MBConvBlock)\n",
    "4. Head_Conv\n",
    "5. Global Average Pooling\n",
    "6. Fully Connected Layer  \n",
    "\n",
    "The details of Stem_Conv, Head_Conv, and MBConvBlock are described below the 'Main network'.\n",
    "* In addition, in MBConvBlock\\[0\\], there is no the first three layers (Expansion_Conv, BatchNorm, Activation Function) in a block since there is no expansion when $e=1$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"./src/overview_v1.png\" width=\"1200\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Our Approach Detail"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2-0. Configuration (Please refer to `Config/main.json`)\n",
    "* <b>Data & Model precision</b>\n",
    "    * 16 bits\n",
    "* <b>Data</b>\n",
    "    * Dataset: cifar-100\n",
    "    * Batch size: 128\n",
    "    * Train size/Valid size: 50000/0\n",
    "    * Augmentation: \\[random crop 32*32 with padding of 4, random horizontal flip(p=0.5), normalization\\] \\+ (custom) auto augmentation for cifar-100 \\+ Mixup\n",
    "* <b>Model</b>\n",
    "    * Architecture: See `figure 1`\n",
    "    * Activation function: swish (beta=1)\n",
    "    * Batch normalization: ghost batch normalization (splits=4)\n",
    "    * Optimizer: sgd (lr=0.13, weight_decay=1e-5, momentum=0.9)\n",
    "    * Loss function: cross entropy loss with label smoothing (smoothing factor=0.3)\n",
    "    * Learning rate scheduler: cosine annealing scheduler (T_max=1200, without restart)\n",
    "    * Epochs #: 1200\n",
    "* <b>Pruning</b>\n",
    "    * Pruning method(one shot/iterative): iterative\n",
    "    * Desired sparsity/Pruning ratio per iteration: 50%/10%\n",
    "    * Epochs # per pruning iteration: 600\n",
    "    * Optimizer: sgd (lr=0.13, weight_decay=1e-5, momuntum=0.9)\n",
    "    * Loss function: cross entropy loss with label smoothing (smoothing factor=0.3)\n",
    "    * Learning rate scheduler: cosine annealing scheduler (T_max=600, without restart)\n",
    "    * Weight reset: False\n",
    "    * Normalization: Layer-wise magitude normalization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2-1. Architecture Search\n",
    "First of all, we search for a baseline architecture suitable for cifar-100 data set based on the [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) architecture using autoML. The search process is as follows:\n",
    "1. <b>Block arguments search</b>: In this step, we search the number of MBConvBlock, and kernel size(k), stride(s), expansion ratio(e), input channels(i), output channels(o), and squeeze-expansion ratio(se) in each block. From the results of the block arguments search, we find out that the position of the convolutional layer which serves to reduce resolution, or convolutional layer with stride of 2, is a sensitive factor to accuracy. With this inference, after several hand-made experiments, the above architecture is chosen.\n",
    "\n",
    "2. <b>Scaling coefficients search</b>: In this step, after block aurgments are decided, we search three coefficients by adjusting available resources: width, depth, and resolution. Actually, we set the depth coefficient as 1 since its slight change gets even worse in terms of score. Therefore, a resolution coefficient is set randomly within a given range according to the available resources, and then a width coefficient is calculated by \\[available resources / resolution coefficient$^2$\\].  From the results of the scaling coefficients search, we find out that a large resolution coefficient make a greater performance improvement than a large width coefficient under our circumstance. As a result, when we set available resources as 2, we get a resolution coefficient of 1.4. Finally, to lighten this model, we decide a width coefficient as 0.9, and adapt these coefficients to the model we've got via block arguments search."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2-2. Techniques for Improvement\n",
    "* <b>[Auto augmentation](https://arxiv.org/pdf/1805.09501.pdf)</b>: We search 25 sub-policies for cifar-100 data set based on the augmentation search space in `AutoAugment` except `Cutout` and `SamplePairing`. Please refer to `AutoML_autoaug.py` for the process and `data_utils/autoaugment.py` for the policy we've got.\n",
    "* <b>[Mixup](https://arxiv.org/pdf/1710.09412.pdf)</b>: We add a Mixup technique with $\\alpha$ of 1, which is the hyperparameter for beta-distribution, after auto augmentation. We thought that this augmentation can help inter-exploration between arbitrary two classes.\n",
    "* <b>[No bias decay](https://arxiv.org/pdf/1812.01187.pdf)</b>: We do not apply weight decay regularizer to biases. Since these part has a small percentage of the total, it can make underfitting.\n",
    "* <b>[Swish activation function](https://arxiv.org/pdf/1710.05941.pdf)</b>: We use a <i>Swish</i> activation function with $\\beta$ of 1, which is $x\\times sigmoid(x)$. This activation function is usually interpreted as a self-gate activation.\n",
    "* <b>[Ghost batch normalization](https://arxiv.org/pdf/1705.08741.pdf)</b>: We use ghost batch normalization, where batch is divided into four smaller ghost batch in our case to match the splited batch size to 32, instead of plain batch normalization.\n",
    "* <b>[Label smoothing](https://arxiv.org/pdf/1512.00567.pdf)</b>: We use a label smoothing technique through which the probability of the correct label is assinged as 0.7, and $\\frac{0.3}{99}$ for the others.\n",
    "* <b>[Cosine annealing scheduler](https://arxiv.org/pdf/1608.03983.pdf)</b>: We use cosine annealing scheduler for adaptive learning rate, and set a period of one cycle as the number of epochs. Hence, there is no restart process."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2-3. Pruning\n",
    "After training the main network, we adapt layer-wise normalized magnitude-based iterative pruning method. We prune 10% from whole weights and repeat 5 times in the same way, and hence 50% of whole parameters are pruned."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Scoring metric\n",
    "The table below describes the number of parameters and the number of operations of our model on a 32-bit basis, which is obtained by hand. \n",
    "- Before pruning:\n",
    "    - Parameter Storage (Numbers): 0.47M / 36.5M * (1/2) = 0.006438\n",
    "    - Math Operation (Numbers): 117.5M / 10490M * (1/2) = 0.005601\n",
    "    - Therefore, score is 0.012039  \n",
    "- After pruning:\n",
    "    - 50% pruning\n",
    "    - Parameter Storage (Numbers): 0.003645\n",
    "    - Math Operation (Numbers): 0.002907\n",
    "    - Therefore, score is 0.006552\n",
    "    \n",
    "Here, (1/2) means 16-bit quantization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "| <div style=\"width:70px\">Input</div> | Operator         |  k  |  s  |  e  |  i  |  o  |  se  | Parameter Storage    | MULTI      |  ADD       | Math Operations |\n",
    "| :---:                               | :---:            |:---:|:---:|:---:|:---:|:---:| :---:| :---:                | :---:      | :---:      | :---:           |\n",
    "| $32^{2}\\times3$                     | Upsample(nearest)| -   | -   | -   | -   | -   | -    | 0                    | 11,907     | 0          | 11,907          |\n",
    "| $63^{2}\\times3$                     | Stem\\_Conv2d     | 3   | 2   | -   | 3   | 24  | -    | 648                  | 691,920    | 622,728    | 1,314,648       |\n",
    "| $31^{2}\\times24$                    | MBConvBlock\\[0\\] | 3   | 1   | 1   | 24  | 16  | 0.20 | 820                  | 669,132    | 584,484    | 1,253,616       |\n",
    "| $31^{2}\\times16$                    | MBConvBlock\\[1\\] | 3   | 1   | 6   | 16  | 24  | 0.20 | 5,379                | 5,167,209  | 4,590,315  | 9,757,524       |\n",
    "| $31^{2}\\times24$                    | MBConvBlock\\[2\\] | 3   | 2   | 6   | 24  | 40  | 0.20 | 11,812               | 5,455,164  | 4,933,372  | 10,388,536      |\n",
    "| $15^{2}\\times40$                    | MBConvBlock\\[3\\] | 3   | 1   | 6   | 40  | 40  | 0.20 | 25,448               | 5,188,584  | 4,908,848  | 10,097,432      |\n",
    "| $15^{2}\\times40$                    | MBConvBlock\\[4\\] | 3   | 1   | 6   | 40  | 48  | 0.20 | 27,368               | 5,620,584  | 5,285,048  | 10,905,632      |\n",
    "| $15^{2}\\times48$                    | MBConvBlock\\[5\\] | 3   | 1   | 6   | 48  | 64  | 0.20 | 40,329               | 8,300,475  | 7,896,393  | 16,196,868      |\n",
    "| $15^{2}\\times64$                    | MBConvBlock\\[6\\] | 3   | 1   | 6   | 64  | 64  | 0.20 | 62,220               | 12,452,004 | 12,004,428 | 24,456,432      |\n",
    "| $15^{2}\\times64$                    | MBConvBlock\\[7\\] | 3   | 2   | 6   | 64  | 80  | 0.20 | 68,364               | 7,549,092  | 7,228,348  | 14,777,440      |\n",
    "| $7^{2}\\times80$                     | MBConvBlock\\[8\\] | 3   | 1   | 6   | 80  | 80  | 0.20 | 96,976               | 4,156,368  | 4,033,376  | 8,189,744       |\n",
    "| $7^{2}\\times80$                     | MBConvBlock\\[9\\] | 3   | 1   | 6   | 80  | 96  | 0.20 | 104,456              | 4,532,688  | 4,385,392  | 8,918,080       |\n",
    "| $7^{2}\\times96$                     | Head\\_Conv2d     | 1   | 1   | -   | 96  | 136 | -    | 13,056               | 659,736    | 639,744    | 1,299,480       |\n",
    "| $7^{2}\\times136$                    | AveragePool      | 7   | -   | -   | -   | -   | -    | 0                    | 136        | 6,528      | 6,664           |\n",
    "| $136$                               | FullyConnected   | -   | -   | -   | -   | -   | _    | 13,700               | 13,600     | 13,600     | 27,200          |\n",
    "| $100$                               | -                | -   | -   | -   | -   | -   | -    | -                    | -          | -          | -               |\n",
    "| Total                               | -                | -   | -   | -   | -   | -   | -    | 470,776              | 60,456,692 | 57,132,604 | 117,589,296     |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3-1. Parameter Storage"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Upsample(nearest): $0$\n",
    "* Stem_Conv2d:\n",
    "    * $3 \\times 3 \\times 3 \\times 24$ (Conv2d)\n",
    "    * $0$ (BatchNorm)\n",
    "    * $0$ (Activation)\n",
    "* MBConvBlock(k,s,e,i,o,se):\n",
    "    * $1 \\times 1 \\times i \\times (i \\times e)$ (Expansion_Conv)\n",
    "    * $k \\times k \\times (i \\times e)$ (Depthwise_Conv)\n",
    "    * $1 \\times 1 \\times (i \\times e) \\times int(i \\times e \\times se)$ (SE_squeeze weight) + $int(i \\times e \\times se)$ (SE_squeeze bias)\n",
    "    * $1 \\times 1 \\times int(i \\times e \\times se) \\times (i \\times e)$ (SE_expand weight) + $(i \\times e)$ (SE_squeeze bias)\n",
    "    * $1 \\times 1 \\times (i \\times e) \\times o$ (Projection_conv) \n",
    "        * when $e=1$, Expansion_Conv is omitted. (i.e., $- 1 \\times 1 \\times i \\times (i \\times e)$)\n",
    "* Head_Conv2d:\n",
    "    * $1 \\times 1 \\times 96 \\times 136$ (Conv2d)\n",
    "    * $0$ (BatchNorm)\n",
    "    * $0$ (Activation)\n",
    "* Global Average Pooling: $0$\n",
    "* FullyConnected: $136 \\times 100$ (Weight) + $100$ (Bias)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3-2. Math Operations"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Upsample(nearest): \\[ $63\\times63\\times3$ \\<multi\\> + $0$ \\<add\\> \\]\n",
    "* Stem_Conv2d:\n",
    "    * \\[ $(3 \\times 3 \\times 3) \\times (31 \\times 31 \\times 24)$ \\<multi\\> + $(3 \\times 3 \\times 3 - 1) \\times (31 \\times 31 \\times 24)$ \\<add\\> \\] (Conv2d)\n",
    "    * $0$ (BatchNorm) \n",
    "    * \\[ $(3) \\times (31 \\times 31 \\times 24)$ \\<multi\\> + $(1) \\times (31 \\times 31 \\times 24)$ \\<add\\> \\] (Swish Activation)\n",
    "* MBConvBlock(k,s,e,i,o,se):\n",
    "    * \\[ $(1 \\times 1 \\times i) \\times (w_{in} \\times h_{in} \\times (i \\times e))$ \\<multi\\> + $(1 \\times 1 \\times i - 1) \\times (w_{in} \\times h_{in} \\times (i \\times e))$ \\<add\\> \\] (Expansion_Conv)\n",
    "    * $0$ (BatchNorm) \n",
    "    * \\[ $(3) \\times (w_{in} \\times h_{in} \\times (i \\times e))$ \\<multi\\> + $(1) \\times (w_{in} \\times h_{in} \\times (i \\times e))$ \\<add\\> \\] (Swish Activation)\n",
    "    * \\[ $(k \\times k) \\times (w_{out} \\times h_{out} \\times (i \\times e))$ \\<multi\\> + $(k \\times k - 1) \\times (w_{out} \\times h_{out} \\times (i \\times e))$ \\<add\\> \\] (Depthwise_Conv)\n",
    "    * $0$ (BatchNorm) \n",
    "    * \\[ $(3) \\times (w_{out} \\times h_{out} \\times (i \\times e))$ \\<multi\\> + $(1) \\times (w_{out} \\times h_{out} \\times (i \\times e))$ \\<add\\> \\] (Swish Activation)\n",
    "    * \\[ $(i \\times e)$ \\<multi\\> + $(w_{out} \\times h_{out} -1 ) \\times (i \\times e)$ \\<add\\> \\] (Global Average Pooling)\n",
    "    * \\[ $(1 \\times 1 \\times (i \\times e)) \\times (1 \\times 1 \\times int(i \\times e \\times se))$ \\<multi\\> + $(1 \\times 1 \\times (i \\times e) -1) \\times (1 \\times 1 \\times int(i \\times e \\times se))$ \\<add\\> \\] (SE_squeee weight)\n",
    "    * \\[ $0$ \\<multi\\> + $(1 \\times 1 \\times int(i \\times e \\times se))$ \\<add\\> \\] (SE_squeee bias)\n",
    "    * $0$ (BatchNorm) \n",
    "    * \\[ $(3) \\times (1 \\times 1 \\times int(i \\times e \\times se))$ \\<multi\\> + $(1) \\times (1 \\times 1 \\times int(i \\times e \\times se))$ \\<add\\> \\] (Swish Activation)\n",
    "    * \\[ $(1 \\times 1 \\times int(i \\times e \\times se)) \\times (1 \\times 1 \\times (i \\times e))$ \\<multi\\> + $(1 \\times 1 \\times int(i \\times e \\times se) -1) \\times (1 \\times 1 \\times (i \\times e))$ \\<add\\> \\] (SE_expand weight)\n",
    "    * \\[ $0$ \\<multi\\> + $(1 \\times 1 \\times (i \\times e))$ \\<add\\> \\] (SE_expand bias)\n",
    "    * $0$ (BatchNorm) \n",
    "    * \\[ $(2) \\times (1 \\times 1 \\times (i \\times e))$ \\<multi\\> + $(1) \\times (1 \\times 1 \\times (i \\times e))$ \\<add\\> \\] (Sigmoid)\n",
    "    * \\[ $(w_{out} \\times h_{out} \\times (i \\times e))$ \\<multi\\> + $0$ \\<add\\> \\] (Scale)\n",
    "    * \\[ $(1 \\times 1 \\times (i \\times e)) \\times (w_{out} \\times h_{out} \\times o)$ \\<multi\\> + $(1 \\times 1 \\times (i \\times e) - 1) \\times (w_{out} \\times h_{out} \\times o)$ \\<add\\> \\] (Projection)\n",
    "        * when $e=1$, Expansion_Conv is omitted. (i.e., $ - 1 \\times 1 \\times i \\times (i \\times e)$)\n",
    "        * when stride equals to 1, and # of input channels and # of output channels are the same, skip connection happens. (i.e., $ + w_{in} \\times h_{in} \\times c) $ (Add)\n",
    "* Head_Conv2d:\n",
    "    * \\[ $(1 \\times 1 \\times 96) \\times (7 \\times 7 \\times 136)$ \\<multi\\> + $(1 \\times 1 \\times 96 - 1) \\times (7 \\times 7 \\times 136)$ \\<add\\> \\] (Conv2d)\n",
    "    * $0$ (BatchNorm) \n",
    "    * \\[ $(3) \\times (7 \\times 7 \\times 136)$ \\<multi\\> + $(1) \\times (7 \\times 7 \\times 136)$ \\<add\\> \\] (Swish Activation)\n",
    "* AveragePool:\n",
    "    * \\[ $(136)$ \\<multi\\> + $(7 \\times 7 - 1 ) \\times 136$ \\<add\\> \\] (Global Average Pooling)\n",
    "* FullyConnected:\n",
    "    * \\[ $(136) \\times (100)$ \\<multi\\> + $(135) \\times (100) $ \\<add\\> \\] (Weight) + \\[ $0$ \\<multi\\> + $(100) $ \\<add\\> \\] (Bias)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Reproduce Process\n",
    "* `python main.py ./Config/main.json`\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
