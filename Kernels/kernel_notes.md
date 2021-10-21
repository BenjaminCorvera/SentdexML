# Kernels
## What are Kernels
- a kernal transforms a feature set to have more dimensions without incuring procesing cost
  - We do this so that feature set has some semblance of linear seperability that can be used to draw a decision boundary (using svm)
- recall that svm classification is y = sign(w.x+b), where x is feature set of unknown dimensions
- since w.x returns a scalar value, x can be infinitely large. x can be subbed for z (projected output of kernel algorithm) 
- there are two major contrains
- yi (xi.w+b) - 1 >= 0
  - w = sum((alphai).(yi).(xi))
  - Every interaction is a dot product
  - full formula is L = sum(alphai) - (1/2)(sum(alphai.alphaj.yi.yj).(xi.xj))

## Why Kernels
$$
K(x,x') = {{z}\cdot {z'}} 
\\
z = function(x)
\\
z' = funtion(x')
$$
- a dot product produces a scalar value
- we need to convert X = [x<sub>1</sub>,x<sub>2</sub>] to a 2nd order polynomial
$$
\begin{aligned}
  & X = [x_1,x_2] \\
  & Z = [1, x_1, x_2, x_1^2, x_1^2,x_1 , x_2]  \\
  &Z' = [1, x_1', x_2', {x'}_1^2, {x'}_1^2,x'_1 , x'_2]  \\
  &K(x,x') = z \cdot z' = 1 + x_1x'_1 + x_2x'_2 + x_1^2{x'}_1^2 +  x_2^2{x'}_2^2 + x_1{x'}_1^1x_2x'_2
\end{aligned}
$$
- this is unfeasibly complex
- can we do this kernel without having to visit the z space
  - yes, with the polynomial kernel
$$
\begin{aligned}
&K(X,X') = (1+x \cdot x')^p \\
&p = 2, n = 2 \\
&(1+x,x_1'+...x_nx_n')^p
\end{aligned}
$$
- even if p and n where 100 and 15 respectively, this is not much more complex
$$
[1,x_1^2,x_2^2,\sqrt{2}x,\sqrt{2}x_2,\sqrt{2}x_1x_2]
$$
- If this was a vector multiplied by itself, you get the same exact scalar without visiting z space
### Radial Basis Kernel (default)
- This works most of the time and the one that is provided out of the box
- You can have a dataset that cannot be linearly seperated
$$
K(X,X') = \exp(-\gamma||x-x'||^2)
$$
## Soft Margin SVM
- In reality, you may find that you either cannot find a linearly separable dimension for your dataset for machine learning, or you may find that your support vector machine has significant overfitment to your data. 
- You know you have over-fitment if you have a large percentage of your dataset as support vectors. 
- The soft-margin SVM allows for some "wiggle room" with separation
  - Allows for "slack"
    $$\xi$$
  - equation becomes : 
     $$y_i(x_i \cdot w + b) >= 1 - \xi$$
  - In contrast to "hard margin" where every + or - must sit in their respective hyperplane
  - Slack should be minimized
- minimize = $\frac{1}{2} ||\overline{w}||^2+c\sum\limits_{i} \xi_i$
  - c is basically a toggle on how important slack becomes in the equation
