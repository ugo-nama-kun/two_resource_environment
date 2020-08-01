# Two Reource Problem 
## Background and Motivation
One of the fundamental feature of the animal behavior is the stability of the internal state of the body. 
This feature, called "Homeostasis" (Feedback mechanism) or "Allostasis" (Model-Predictive mechanism), have been received attentions since the early stage og the artificial intelligence. One of the origin of these insight in the artificial intelligence context would be the Ashby's Homeostat and the concept of the "Ultra-stable system". He rigorously treated the problem of survival in the general environment as a problem of the regulation of the interoceptive control through the behavior selection. This is a concept of the classical cybernetics.

<p align="center">
  <img width="300" height="300" src="https://user-images.githubusercontent.com/1684732/89105612-78aef380-d45d-11ea-8ba6-e739c9c16774.png">
</p>

Because of the universality and the generality of the problem, the optimal control of the internal state of the agent offers the theoretically-grounded treatment of the survival of the "natural agent" (animals). Fortunately, Dawkins says the survival of individuals can be seen as an approximation of the objective of the animals in [The Selfish Gene](https://en.wikipedia.org/wiki/The_Selfish_Gene) Recent progress of the theoretical neuroscience is starting to discuss a "regulator" perspective of the animal behavior, integrating the Bayesian view of the control problem. 

<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/1684732/89105835-b0b73600-d45f-11ea-8c81-45c99d4a1af8.png">
</p>

(From [Seth](https://open-mind.net/DOI?isbn=9783958570108))
  
Animals regulate multiple resources through behavior control. This feature is treated in the field of the [food selection](https://science.sciencemag.org/content/307/5706/111.abstract) research or the nutrient selection. This behavior can be [observed in insects too](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2011.2410). Researchers of the theoretical animal behavior suggested the "two-resource problem" as a simplest but concrete form of the nutrient selection problem (image below). The agent has sensor for nutrient detection, interoceptive nutrient level sensors, and manually implemented high-level foraging behaviors for two nutrient resources.

<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/1684732/89105594-469d9180-d45d-11ea-944c-367bab8b7c68.png">
</p>

(Two-resource problem overview, From  [McFarland & Spiery](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.6775&rep=rep1&type=pdf))

In this project of my research, I treated the problem of the homeostasis and the survival as the stochastic optimal control problem. The agent receives interoceptive signals from the body (red & blue resource levels) and two-steps RGB vision inputs. Agent has only primitive actions like "go forward", "turn left" or "eat it". Behavior optimization from the motor control level will be done in the future research. 

<p align="center">
  <img width="640" height="434" src="https://user-images.githubusercontent.com/1684732/89106581-07c00980-d466-11ea-9c67-6fcb01762be2.gif">
</p>

For simplicity, we used the "vanilla" [Deep Q network](https://www.nature.com/articles/nature14236?wm=book_wap_0005) for the optimization. Recent more advanced optimization algorithms will optimize faster and better than my realization. This experiment is a replication of my previous research of the [general homeostatic agent](https://content.sciendo.com/view/journals/jagi/8/1/article-p1.xml).

### Youtube
https://www.youtube.com/watch?v=_xhMq272wbE

### Technical Issues
- Agent has NONE, LEFT TURN, RIGHT TURN, FORWARD and EAT behaviors. 
- For enhancing the learning speed, I used the technique of the [shaping reward](https://www.jair.org/index.php/jair/article/view/10338) for adding the initial bias of the value function.

### Some Work Environment is Something Like...
Windows 10 + Anaconda + Python 3.6+ + Unity ML-Agent 1.0+

