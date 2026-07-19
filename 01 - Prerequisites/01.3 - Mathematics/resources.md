# Mathematics for AI — Learning Resources

## Books

### Linear Algebra
- **"Linear Algebra and Its Applications"** — Gilbert Strang. The gold standard. Clear intuition with practical applications.
- **"Linear Algebra Done Right"** — Sheldon Axler. More theoretical, excellent for deep understanding of vector spaces.
- **"Matrix Analysis and Applied Linear Algebra"** — Carl D. Meyer. Comprehensive reference with applications.

### Calculus
- **"Calculus"** — James Stewart. Standard university calculus text covering single and multivariable.
- **"Calculus on Manifolds"** — Michael Spivak. Concise, rigorous treatment for advanced readers.
- **"The Calculus of Several Variables"** — Serge Lang. Good for multivariate analysis needed in ML.

### Probability & Statistics
- **"Introduction to Probability"** — Joseph K. Blitzstein & Jessica Hwang. Excellent intuition, Harvard's Stat 110 text.
- **"Pattern Recognition and Machine Learning"** — Christopher Bishop. Chapters 1-2 give outstanding coverage of probability for ML.
- **"The Elements of Statistical Learning"** — Hastie, Tibshirani, Friedman. A definitive reference for statistical learning.
- **"Probability Theory: The Logic of Science"** — E. T. Jaynes. Bayesian perspective, deeply insightful.

### Optimization
- **"Convex Optimization"** — Stephen Boyd & Lieven Vandenberghe. The definitive text on convex optimization.
- **"Numerical Optimization"** — Jorge Nocedal & Stephen J. Wright. Comprehensive coverage of optimization algorithms.
- **"Optimization for Machine Learning"** — Suvrit Sra, Sebastian Nowozin, Stephen J. Wright. ML-specific optimization techniques.

### Information Theory
- **"Elements of Information Theory"** — Thomas M. Cover & Joy A. Thomas. The canonical text on information theory.
- **"Information Theory, Inference, and Learning Algorithms"** — David J. C. MacKay. Connects information theory to ML. Free online: https://www.inference.org.uk/mackay/itila/
- **"A Mathematical Theory of Communication"** — Claude Shannon (1948). The original paper that founded information theory.

---

## Online Courses

### Linear Algebra
- **MIT 18.06 — Linear Algebra** (Gilbert Strang) — https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/
- **Imperial College — Mathematics for Machine Learning: Linear Algebra** (Coursera) — https://www.coursera.org/learn/linear-algebra-machine-learning

### Calculus
- **MIT 18.01/18.02 — Single/Multivariable Calculus** — https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/
- **3Blue1Brown — Essence of Calculus** (YouTube) — Outstanding visual intuition for calculus concepts.

### Probability & Statistics
- **Harvard Stat 110 — Probability** (Joe Blitzstein) — https://projects.iq.harvard.edu/stat110
- **Stanford CS109 — Probability for Computer Scientists** — https://web.stanford.edu/class/cs109/
- **MIT 18.05 — Introduction to Probability and Statistics** — https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/

### Optimization
- **Stanford EE364A — Convex Optimization** (Boyd) — https://stanford.edu/class/ee364a/
- **DeepLearning.AI — Optimization for Machine Learning** (Coursera)

### Information Theory
- **Stanford EE376A — Information Theory** — https://web.stanford.edu/class/ee376a/
- **Cambridge — Information Theory** (MacKay) — Video lectures available online

### Integrated Math + ML Courses
- **Stanford CS229 — Machine Learning** (lecture notes cover all math prerequisites) — https://cs229.stanford.edu/
- **NYU DS-GA 1003 — Machine Learning** (Goodfellow/Bengio)
- **fast.ai — Computational Linear Algebra** — https://github.com/fastai/numerical-linear-algebra
- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong) — Book + companion course: https://mml-book.github.io/

---

## Research Papers (Foundational)

- **Shannon, C. E. (1948).** "A Mathematical Theory of Communication." — *The Bell System Technical Journal.*
- **Pearson, K. (1901).** "On Lines and Planes of Closest Fit to Systems of Points in Space." — *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science.* (PCA origin)
- **Hotelling, H. (1933).** "Analysis of a complex of statistical variables into principal components." — *Journal of Educational Psychology.*
- **Eckart, C. & Young, G. (1936).** "The approximation of one matrix by another of lower rank." — *Psychometrika.* (Eckart-Young theorem for SVD)
- **Robbins, H. & Monro, S. (1951).** "A Stochastic Approximation Method." — *The Annals of Mathematical Statistics.* (SGD origin)
- **Kingma, D. P. & Ba, J. (2015).** "Adam: A Method for Stochastic Optimization." — *ICLR.*
- **Cover, T. M. & Van Campenhout, J. (1981).** "On the possible orderings in the measurement selection problem." — *IEEE Transactions on Information Theory.*

---

## Interactive Tools & Visualizations

- **3Blue1Brown** (YouTube) — "Essence of Linear Algebra", "Essence of Calculus", "Neural Networks" series
- **Seeing Theory** (Brown University) — https://seeing-theory.brown.edu/ — Interactive probability & statistics visualizations
- **Setosa.io** — Interactive visual explanations of PCA, SVD, Markov chains
- **Distill.pub** — High-quality interactive ML explanations
- **Immersive Linear Algebra** — http://immersivemath.com/ila/
- **Matrix Calculus for Deep Learning** (Parr & Howard) — https://arxiv.org/abs/1802.01528

---

## Reference & Cheat Sheets

- **The Matrix Cookbook** (Petersen & Pedersen) — Comprehensive matrix identities and derivatives
- **Probability Cheatsheet** (Joe Blitzstein) — Concise probability reference
- **Stanford CS229 Probability & Statistics Refresher** — https://cs229.stanford.edu/section/cs229-prob.pdf
- **Stanford CS229 Linear Algebra Refresher** — https://cs229.stanford.edu/section/cs229-linalg.pdf
- **NumPy Documentation** — https://numpy.org/doc/stable/

---

## Tips for Self-Study

1. **Work through derivations by hand** — The key formulas (MLE, gradient descent, backpropagation) are best understood by deriving them yourself.
2. **Implement from scratch** — Before using high-level libraries, implement linear regression, logistic regression, and PCA in NumPy.
3. **Visualize concepts** — Plot loss surfaces, gradient descent paths, probability distributions. Visualization builds intuition.
4. **Connect theory to code** — For each formula in this module, write a corresponding NumPy function.
5. **Spaced repetition** — Review the cheatsheet weekly. Math for ML is about familiarity, not memorization.
6. **Start with Strang + Blitzstein + Boyd** — These three authors provide the clearest exposition for linear algebra, probability, and optimization respectively.
