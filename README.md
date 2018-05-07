# One Cycle Learning Rate Policy for Keras
Implementation of One-Cycle Learning rate policy from the papers by Leslie N. Smith.

- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
- [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
        
Contains two Keras callbacks, `LRFinder` and `OneCycleLR` which are ported from the PyTorch *Fast.ai* library.

# What is One Cycle Learning Rate
It is the combination of gradually increasing learning rate, and optionally, gradually decreasing the momentum during the first half of the cycle, then gradually decreasing the learning rate and optionally increasing the momentum during the latter half of the cycle. 

Finally, in a certain percentage of the end of the cycle, the learning rate is sharply reduced every epoch. 

The Learning rate schedule is visualized as : 

<img src="https://github.com/titu1994/keras-one-cycle/blob/master/images/one_cycle_lr.png?raw=true" height=50% width=100%> 

The Optional Momentum schedule is visualized as : 

<img src="https://github.com/titu1994/keras-one-cycle/blob/master/images/one_cycle_momentum.png?raw=true" height=50% width=100%>

# Usage

## Finding a good learning rate
Use `LRFinder` to obtain a loss plot, and visually inspect it to determine the initial loss plot. Provided below is an example, used for the `MiniMobileNetV2` model.

An example script has been provided in `find_schedule_cifar_10.py`.

Essentially,

```python
from clr import LRFinder

lr_callback = LRFinder(num_samples, batch_size,
                       minimum_lr, maximum_lr,
                       lr_scale='exp', save_dir='path/to/save/directory')

# Ensure that number of epochs = 1 when calling fit()
model.fit(X, Y, epochs=1, batch_size=batch_size, callbacks=[lr_callback])
```
The above callback does a few things. 

- Must supply number of samples in the dataset (here, 50k from CIFAR 10) and the batch size that will be used during training.
- `lr_scale` is set to `exp` - useful when searching over a large range of learning rates. Set to `linear` to search a smaller space.
- `save_dir` - Automatic saving of the results of LRFinder on some directory path specified. This is highly encouraged.

To visualize the plot, there are two ways - 

- Use `lr_callback.plot_schedule()` after the fit() call. This uses the current training session results.
- Use class method `LRFinder.plot_schedule_from_file('path/to/save/directory')` to visualize the plot separately from the training session. This only works if you used the `save_dir` argument to save the results of the search to some location.

## Interpreting the plot

<centre>
<img src="https://github.com/titu1994/keras-one-cycle/blob/master/images/lr.png?raw=true" width="100%" height="50%">
</centre>

Consider the above plot from using the `LRFinder` on the MiniMobileNetV2 model. In particular, there are a few regions above that we need to carefully interpret. 

**Note : The values are in log 10 scale (since `exp` was used for `lr_scale`)** ; All values discussed will be based on the x-axis (learning rate) : 

- After the -1.5 point on the graph, the loss becomes erratic
- After the 0.5 point on the graph, the loss is noisy but doesn't decrease any further.
- **-1.7** is the last relatively smooth portion before the **-1.5** region. To be safe, we can choose to move a little more to the left, closer to -1.8, but this will reduce the performance. 
- It is usually important to visualize the first 2-3 epochs of `OneCycleLR` training with values close to these edges to determine which is the best. 

## Training with `OneCycleLR`
Once we find the maximum learning rate, we can then move onto using the `OneCycleLR` callback with SGD to train our model.

```python
from clr import OneCycleLR

lr_manager = OneCycleLR(num_samples, num_epoch, batch_size, max_lr
                        end_percentage=0.1, scale_precentage=None,
                        maximum_momentum=0.95, minimum_momentum=0.85)
                        
model.fit(X, Y, epochs=EPOCHS, batch_size=batch_size, callbacks=[model_checkpoint, lr_manager], 
          ...)
```

There are many parameters, but a few of the important ones : 
- Must provide a lot of training information - `number of samples`, `number of epochs`, `batch size` and `max learning rate`
- `end_percentage` is used to determine what percentage of the training epochs will be used for steep reduction in the learning rate. At its miminum, the lowest learning rate will be calculated as 1/1000th of the `max_lr` provided.
- `scale_precentage` is a confusing parameter. It dictates the scaling factor of the learning rate in the second half of the training cycle. **It is best to test this out visually using the `plot_clr.py` script to ensure there are no mistakes**. Leaving it as None defaults to using the same percentage as the provided `end_percentage`.
- `maximum/minimum_momentum` are preset according to the paper and `Fast.ai`. However, if you don't wish to scale it, set both to the same value, generally `0.9` is preferred as the momentum value for SGD. If you don't want to update the momentum / are not using SGD (not adviseable) - set both to None to ignore the momentum updates.

# Results

**-1.7** is therefore chosen to be the maximum learning rate (in log10 space) for the `OneCycleLR` schedule. Since this is in log10 scale, we use `10 ^ (x)` to get the actual learning maximum learning rate. Here, `10 ^ -1.7 ~ 0.019999`. Therefore, we round up to a **maximum learning rate of 0.02**

For the MiniMobileNetV2 model, 2 passes of the OneCycle LR with SGD (40 epochs - max lr = 0.02, 30 epochs - max lr = 0.005) obtained 90.33%. This may not seem like much, but this is a model with only 650k parameters, and in comparison, the same model trained on Adam with initial learning rate 2e-3 did not converge to the same score in over 100 epochs (89.14%). 

# Requirements
- Keras 2.1.6+
- Tensorflow (tested) / Theano / CNTK for the backend
- matplotlib to visualize the plots.
